#!/usr/bin/env python3
r"""
Comprehensive mountain weather reports -> three PDFs:
  reports/alta_weather_report_YYYYMMDD_HHMM.pdf   (Alta & Little Cottonwood)
  reports/pc_weather_report_YYYYMMDD_HHMM.pdf     (Park City & Deer Valley)
  reports/bcc_weather_report_YYYYMMDD_HHMM.pdf    (Big Cottonwood Canyon)
Locations, forecast points and stations for each report are set in SITES below.

Reliability: every download retries (see get(): 3 more attempts after 10/30/60 s, bad
pages such as NWS "Runtime Error" included). If a forecast.weather.gov table still
fails, that point's forecast comes from the NWS API instead (same values; the page says
so). Each run's per-section results go to status.json next to the web page, which the
workflow's catch-up runs use to rebuild only when something failed.

Forecast sections (scraped; the avalanche and AFD sections are shared by all reports):
  * Utah Avalanche Center - Salt Lake forecast (text + danger rose + problem graphics)
  * NWS Salt Lake City Area Forecast Discussion (AFD)
  * NWS point forecasts (Alta Base + Mount Superior / PC Base + PC Jupiter / Kessler +
    Spruces + Flagstaff): 7-day and next-24-hour charts plus a daily summary, built
    from the tabular ("digital") forecast pages
  * University of Utah GEFS downscaled ensemble plume (Alta-Collins / PCMR Summit /
    Silver Lake Brighton)
  * UAC avalanches reported in the Salt Lake region over the past 7 days

History sections:
  * NCEI co-op station (Alta: ALTA, GHCND:USC00420072; Park City: PARK CITY,
    GHCND:USC00426644; Big Cottonwood: SILVER LAKE BRIGHTON, GHCND:USC00427846):
    the station is looked up with the HOMR API
    (https://www.ncei.noaa.gov/access/homr/api, station *metadata* only) and its daily
    data comes from the NCEI Access Data Service: max/min temperature, new snow and
    snow depth (no wind). NCEI data lag days to a couple of weeks.
  * Mesonet stations (Alta: ALTA - BASE, COLLINS, TOP OF COLLINS, MOUNT BALDY,
    LCC - CARDIFF PEAK; Park City: DV - BURNS, DV - EMPIRE PEAK; Big Cottonwood:
    SOLITUDE HONEYCOMB PEAK, REYNOLDS PEAK, SPRUCES): 30-day daily
    temperature, wind speed and wind direction candlesticks, plus the last 24 hours of
    hourly temperature and wind/gusts and a 24-hour wind rose.
    NCEI/HOMR doesn't carry these stations. Sources, tried in order per station:
      1. the weather.gov station timeseries page (https://www.weather.gov/wrh/timeseries),
         read with headless Chromium via Playwright, since the table is drawn by
         JavaScript. Full hourly data for up to 30 days (the page's limit is 720 h).
      2. Synoptic Data (MesoWest), when --synoptic-token / SYNOPTIC_TOKEN is set and
         works (7 days on the free plan).
      3. the public NWS API (api.weather.gov): ~7 days, roughly half the hours.
    Each run also saves what it gets in data/, and the charts merge it in.
    Setup for (1):  pip install playwright, then
        $env:PLAYWRIGHT_BROWSERS_PATH = "$HOME\.playwright-browsers"
        python -m playwright install chromium-headless-shell
    (the script uses ~/.playwright-browsers automatically when it exists)

Usage:
    python weather_report.py                 # both reports, into reports/
    python weather_report.py --site pc       # just Park City (or --site alta)
    python weather_report.py --days 30 --outdir somewhere
    $env:SYNOPTIC_TOKEN = "xxxx"             # optional fallback data source

Requirements: requests, beautifulsoup4, pandas, numpy, matplotlib, pillow
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import os
import re
import shutil
import sys
import textwrap
import time
import traceback
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from matplotlib.backends.backend_pdf import PdfFile, PdfPages, RendererPdf
from matplotlib.colors import LinearSegmentedColormap, Normalize
from PIL import Image

# --------------------------------------------------------------------------- config
TZ = ZoneInfo("America/Denver")
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; AltaWeatherReport/1.0)"}
TIMEOUT = 45

# Shared by both reports
UAC_URL = "https://utahavalanchecenter.org/forecast/salt-lake"
AFD_URL = ("https://forecast.weather.gov/product.php?site=SLC&issuedby=SLC&product=AFD"
           "&format=CI&version=1&glossary=1")
PLUME_PAGE = "https://weather.utah.edu/index.php"

HOMR_SEARCH = "https://www.ncei.noaa.gov/access/homr/services/station/search"
NCEI_DATA = "https://www.ncei.noaa.gov/access/services/data/v1"

SYNOPTIC_BASE = "https://api.synopticdata.com/v2"
NWS_API = "https://api.weather.gov"
NWS_API_HEADERS = {"User-Agent": "(alta-weather-report, personal use)",
                   "Accept": "application/geo+json"}
WRH_TIMESERIES = "https://www.weather.gov/wrh/timeseries"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
# After a full run (both reports), the PDFs are copied here and every other file
# directly in this folder is deleted. Use --publish-dir "" to turn this off.
PUBLISH_DIR = r"G:\My Drive\02 Ski Patrol\01 WeatherReports"


@dataclass
class Point:
    """An NWS point forecast location. elev is the label shown in titles; the NWS
    page's own grid-cell elevation is read from the page and shown in the subtitle."""
    label: str
    elev: int
    lat: float
    lon: float

    @property
    def url(self) -> str:
        return (f"https://forecast.weather.gov/MapClick.php?lat={self.lat}&lon={self.lon}"
                "&unit=0&lg=english&FcstType=digital")


@dataclass
class Site:
    key: str
    title: str                  # title page heading
    file_prefix: str            # <prefix>_YYYYMMDD_HHMM.pdf
    points: list[Point]         # NWS point forecasts
    plume_station: str          # weather.utah.edu plume site code
    plume_name: str
    stations: list[tuple]       # (display name, elevation ft, station ID) for history
    history_label: str
    ncei_ghcnd: str | None = None  # NCEI co-op station (GHCN-Daily ID) for the NCEI page
    snow_station: tuple | None = None  # (name, elev ft, ID) whose snow joins the NCEI page


SITES = {
    "alta": Site(
        key="alta", title="Alta & Little Cottonwood", file_prefix="alta_weather_report",
        points=[Point("Alta Base", 8560, 40.5846, -111.6213),
                Point("Alta-Collins Top", 9400, 40.572, -111.631),
                Point("Mount Superior", 10318, 40.592, -111.667)],
        plume_station="CLN", plume_name="Alta-Collins",
        # Station IDs are the same on weather.gov, Synoptic/MesoWest and the NWS API.
        stations=[("ALTA - BASE", 8560, "ATB"),
                  ("ALTA - COLLINS", 9662, "CLN"),
                  ("ALTA - TOP OF COLLINS", 10443, "ALT"),
                  ("ALTA - MOUNT BALDY", 11066, "AMB"),
                  ("LCC - CARDIFF PEAK", 10059, "IFF")],
        history_label="Alta stations", ncei_ghcnd="USC00420072",   # ALTA co-op
        snow_station=("ALTA - COLLINS", 9662, "CLN")),
    "pc": Site(
        key="pc", title="Park City & Deer Valley", file_prefix="pc_weather_report",
        points=[Point("PC Base", 6697, 40.6461, -111.498),
                Point("PC Jupiter", 9406, 40.613, -111.54)],
        plume_station="PKC", plume_name="PCMR Summit",
        stations=[("DV - BURNS", 7333, "DVE"),
                  ("DV - EMPIRE PEAK", 9570, "EMPUT")],
        history_label="Park City / Deer Valley stations",
        ncei_ghcnd="USC00426644",   # PARK CITY co-op, 6,900 ft
        snow_station=("DV - BURNS", 7333, "DVE")),
    "bcc": Site(
        key="bcc", title="Big Cottonwood Canyon", file_prefix="bcc_weather_report",
        # forecasts and stations in increasing elevation
        points=[Point("Spruces", 7400, 40.643, -111.637),
                Point("Kessler", 9400, 40.626, -111.669),
                Point("Flagstaff", 10318, 40.599, -111.67)],
        plume_station="PKCU1", plume_name="Silver Lake Brighton",
        stations=[("SPRUCES", 7400, "SPC"),
                  ("REYNOLDS PEAK", 9400, "REY"),
                  ("SOLITUDE HONEYCOMB PEAK", 10488, "SOLHP")],
        history_label="Big Cottonwood stations",
        ncei_ghcnd="USC00427846",   # SILVER LAKE BRIGHTON co-op
        snow_station=("SPRUCES", 7400, "SPC")),
}

PAGE = (8.5, 11)  # portrait letter
LAND = (11, 8.5)  # landscape letter

# Palette (colorblind-safe)
C_WARM = "#d1495b"
C_COOL = "#2e86ab"
C_MEAN = "#1b3b5a"
C_GUST = "#e07a1f"
C_GRID = "#d9d9d9"
C_TEXT = "#222222"
C_MUTED = "#666666"

plt.rcParams.update({
    "font.size": 9,
    "axes.edgecolor": "#888888",
    "axes.grid": True,
    "grid.color": C_GRID,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


RETRY_WAITS = (10, 30, 60)   # seconds to wait before each retry
_HTTP_CACHE: dict = {}       # good responses only, shared by every report and both passes
_DOWN_HOSTS: set = set()     # hosts that already failed every attempt during this run


class FetchError(RuntimeError):
    pass


def get(url: str, params: dict | None = None, validate=None, headers: dict | None = None,
        **kw) -> requests.Response:
    """GET with retries and a per-run cache.

    Retries timeouts, connection errors, HTTP 5xx/429 and responses that fail
    `validate(response)` (e.g. an NWS "Runtime Error" page served with status 200),
    waiting RETRY_WAITS between attempts. Other 4xx errors aren't retried. Only good
    responses are cached, so a bad page is never reused. Once a host has failed
    completely, later requests to it get a single quick retry so an outage can't stall
    the whole run."""
    key = (url, tuple(sorted((params or {}).items())))
    if key in _HTTP_CACHE:
        return _HTTP_CACHE[key]
    host = requests.utils.urlparse(url).netloc
    waits = (0,) + ((10,) if host in _DOWN_HOSTS else RETRY_WAITS)
    last = None
    for wait in waits:
        if wait:
            print(f"  retrying {url} in {wait}s ({last})", file=sys.stderr, flush=True)
            time.sleep(wait)
        try:
            r = requests.get(url, params=params, headers={**HEADERS, **(headers or {})},
                             timeout=TIMEOUT, **kw)
            if r.status_code >= 500 or r.status_code == 429:
                last = f"HTTP {r.status_code}"
                continue
            r.raise_for_status()  # other 4xx: a real error, not worth retrying
            if validate is not None and not validate(r):
                last = "unexpected page content"
                continue
            _HTTP_CACHE[key] = r
            return r
        except (requests.ConnectionError, requests.Timeout) as e:
            last = type(e).__name__
    _DOWN_HOSTS.add(host)
    raise FetchError(f"{url} failed after {len(waits)} attempts ({last})")


def soup_of(url: str, **kw) -> BeautifulSoup:
    r = get(url, **kw)
    r.encoding = r.apparent_encoding if not r.encoding or r.encoding.lower() == "iso-8859-1" else r.encoding
    return BeautifulSoup(r.text, "html.parser")


def fetch_image(url: str) -> Image.Image:
    return Image.open(io.BytesIO(get(url).content)).convert("RGB")


# --------------------------------------------------------------------------- PDF helpers
@dataclass
class Report:
    pdf: PdfPages
    toc: list = field(default_factory=list)
    page_no: int = 0
    generated: dt.datetime = field(default_factory=lambda: dt.datetime.now(TZ))
    label: str = "Weather report"

    def save(self, fig, section: str | None = None):
        self.page_no += 1
        if section:
            self.toc.append((section, self.page_no))
        fig.text(0.5, 0.012, f"{self.label}  ·  generated {self.generated:%a %b %d %Y %H:%M %Z}"
                              f"  ·  page {self.page_no}", ha="center", fontsize=7, color=C_MUTED)
        self.pdf.savefig(fig)
        plt.close(fig)


def header(fig, title: str, subtitle: str = ""):
    fig.text(0.06, 0.965, title, fontsize=15, weight="bold", color=C_TEXT, va="top")
    if subtitle:
        fig.text(0.06, 0.937, subtitle, fontsize=8.5, color=C_MUTED, va="top")


class Flow:
    """Lays text and images down portrait pages top to bottom, measuring each block as it
    renders and starting a new page when the next block won't fit above the footer."""
    TOP, BOTTOM, LEFT, RIGHT = 0.905, 0.05, 0.06, 0.94

    def __init__(self, rep: Report, title: str, source: str = "", section: str | None = None,
                 mono: bool = False):
        self.rep, self.title, self.source, self.section, self.mono = rep, title, source, section, mono
        self.fig, self.pages = None, 0
        self._new_page()

    def _new_page(self):
        if self.fig is not None:
            self._save()
        self.pages += 1
        self.fig = plt.figure(figsize=PAGE, dpi=72)
        first = self.pages == 1
        header(self.fig, self.title + ("" if first else " (cont.)"), self.source if first else "")
        self.y = self.TOP if first else 0.915
        # measure with the PDF renderer itself: the screen (Agg) renderer reports text about
        # 8% shorter than the PDF draws it, which let long pages run into the footer
        self.renderer = RendererPdf(PdfFile(io.BytesIO()), 72, PAGE[0] * 72, PAGE[1] * 72)

    def _save(self):
        self.rep.save(self.fig, self.section if self.pages == 1 else None)

    def _height(self, artist) -> float:
        return artist.get_window_extent(self.renderer).height / self.fig.bbox.height

    def space(self, frac: float):
        self.y -= frac

    def text(self, s: str, size: float = 7.8, weight: str = "normal", color: str = C_TEXT,
             width: int | None = None, gap: float = 0.004):
        """A paragraph (or preformatted block when mono). Splits across pages if needed."""
        width = width or (98 if self.mono else int(105 * 7.6 / size))
        lines = []
        for para in s.splitlines() or [""]:
            if self.mono:
                lines += textwrap.wrap(para, width, replace_whitespace=False,
                                       drop_whitespace=False) or [""]
            else:
                lines += textwrap.wrap(para, width) or [""]
        self.y -= gap
        while lines:
            t = self.fig.text(self.LEFT, self.y, "\n".join(lines), fontsize=size, weight=weight,
                              color=color, va="top", ha="left", linespacing=1.32,
                              family="monospace" if self.mono else "sans-serif", parse_math=False)
            h = self._height(t)
            if self.y - h >= self.BOTTOM:
                self.y -= h
                return
            t.remove()
            fit = int((self.y - self.BOTTOM) / (h / len(lines)))
            while fit >= 2 or (fit >= 1 and len(lines) == 1):
                part = self.fig.text(self.LEFT, self.y, "\n".join(lines[:fit]), fontsize=size,
                                     weight=weight, color=color, va="top", ha="left",
                                     linespacing=1.32, parse_math=False,
                                     family="monospace" if self.mono else "sans-serif")
                ph = self._height(part)
                if self.y - ph >= self.BOTTOM:  # estimate verified: it fits
                    self.y -= ph
                    lines = lines[fit:]
                    break
                part.remove()
                fit -= 1
            self._new_page()

    def heading(self, s: str):
        # keep a heading with at least a few lines of what follows
        if self.y - 0.08 < self.BOTTOM:
            self._new_page()
        self.text(s, size=9.5, weight="bold", gap=0.012)

    def image(self, img: Image.Image, max_h: float, caption: str = ""):
        aspect = img.height / img.width * PAGE[0] / PAGE[1]  # height per unit width, in fig fractions
        w = min(self.RIGHT - self.LEFT, max_h / aspect)
        h = w * aspect
        if self.y - h - 0.02 < self.BOTTOM:
            self._new_page()
        if caption:
            self.text(caption, size=7.5, color=C_MUTED, gap=0.006)
        ax = self.fig.add_axes([self.LEFT, self.y - h - 0.004, w, h])
        ax.imshow(img)
        ax.axis("off")
        self.y -= h + 0.01

    def finish(self):
        self._save()


def text_pages(rep: Report, title: str, text: str, source: str, mono: bool = False,
               section: str | None = None):
    """Flow long text over as many portrait pages as needed."""
    flow = Flow(rep, title, source, section, mono=mono)
    cleaned = re.sub(r"\n{3,}", "\n\n", text.strip("\n"))
    flow.text(cleaned, size=7.6, gap=0.0)
    flow.finish()


def image_page(rep: Report, title: str, images: list[Image.Image], source: str,
               section: str | None = None, landscape: bool = False, captions=None):
    fig = plt.figure(figsize=LAND if landscape else PAGE)
    header(fig, title, source)
    n = len(images)
    top, bottom = 0.90, 0.04
    h = (top - bottom) / n
    for i, img in enumerate(images):
        ax = fig.add_axes([0.04, top - (i + 1) * h + 0.01, 0.92, h - 0.02])
        ax.imshow(img)
        ax.axis("off")
        if captions and captions[i]:
            ax.set_title(captions[i], fontsize=8, color=C_MUTED, loc="left")
    rep.save(fig, section)


def error_page(rep: Report, title: str, err: str, section: str | None = None):
    fig = plt.figure(figsize=PAGE)
    header(fig, title, "This section couldn't be built")
    fig.text(0.06, 0.88, "\n".join(textwrap.wrap(err, 100)[:60]), fontsize=8, va="top",
             family="monospace", color=C_WARM, parse_math=False)
    rep.save(fig, section)


# --------------------------------------------------------------------------- forecast: UAC
def avalanche_section(rep: Report):
    s = soup_of(UAC_URL)
    main = s.find("main") or s.body
    text = main.get_text("\n", strip=True)

    start = text.find("Forecast for the")
    end = text.find("General Announcements")
    body = text[start:end if end > 0 else None] if start >= 0 else text
    # drop sponsor blurb
    body = re.sub(r"\nSponsor\n.*?(?=\nAvalanche Problem)", "\n", body, flags=re.S)
    # The UAC site serves U+FFFD in place of some punctuation: after a digit it is a
    # foot mark, elsewhere usually an em dash.
    body = re.sub("(?<=\\d)\ufffd", "'", body).replace("\ufffd", " – ")

    # page text is one token per line; rejoin into ("h", heading) / ("p", paragraph) items
    headings = {"Special Announcements", "Weather and Snow", "Recent Avalanches",
                "Additional Information", "Description"}
    skip_headings = {"Type", "Location", "Likelihood", "Size"}
    items, buf = [], []

    def flush():
        if buf:
            items.append(("p", re.sub(r"\s+([.,;:!?])", r"\1", " ".join(buf))))
            buf.clear()

    for ln in body.split("\n"):
        if ln in headings or ln in skip_headings or ln.startswith("Avalanche Problem #"):
            flush()
            if ln not in skip_headings:
                items.append(("h", ln))
        elif ln in {"Low", "Moderate", "Considerable", "High", "Extreme",
                    "Learn how to read the forecast", "here"}:
            continue
        else:
            buf.append(ln)
            if re.search(r"[.!?:]$", ln) and len(" ".join(buf)) > 60:
                flush()
    flush()

    # images: overall danger rose first, then one location rose per avalanche problem
    imgs = []
    for img in main.find_all("img"):
        src = img.get("src") or ""
        if "/files/forecast/" in src:
            try:
                imgs.append(fetch_image(requests.compat.urljoin(UAC_URL, src)))
            except Exception:
                imgs.append(None)
    rose, problem_imgs = (imgs[0] if imgs else None), imgs[1:]

    flow = Flow(rep, "Avalanche Forecast — Salt Lake Mountains",
                f"Utah Avalanche Center  ·  {UAC_URL}", section="Avalanche forecast (UAC)")
    if rose is not None:
        flow.image(rose, max_h=0.24, caption="Overall danger rose")
    problem, awaiting_type = 0, False
    for kind, txt in items:
        if kind == "h" and txt.startswith("Avalanche Problem #"):
            problem += 1
            flow.heading(f"Avalanche Problem #{problem}")
            awaiting_type = True
        elif kind == "h":
            flow.text(txt, size=8.2, weight="bold", gap=0.008) if txt == "Description" \
                else flow.heading(txt)
        elif awaiting_type:
            # the line right after "Avalanche Problem #N" is its type, e.g. "Persistent Weak
            # Layer": show it, then that problem's location rose, then the description
            flow.text(txt, size=9, weight="bold", color=C_WARM, gap=0.002)
            img = problem_imgs[problem - 1] if problem - 1 < len(problem_imgs) else None
            if img is not None:
                flow.image(img, max_h=0.16)
            awaiting_type = False
        else:
            flow.text(txt)
    flow.finish()


# --------------------------------------------------------------------------- avalanches past 7 days
UAC_EXPLORER = "https://utahavalanchecenter.org/observations-explorer"
UAC_AVY_API = "https://utahavalanchecenter.org/api/v2/avalanches"
AVY_REGION = "Salt Lake"

AVY_TYPES = ["Dry Loose", "Soft Slab", "Hard Slab", "Wet Loose", "Wet Slab", "Cornice Fall",
             "Glide", "Roof Avalanche"]
AVY_PROBLEMS = ["New Snow (NS)", "Persistent Weak Layer (PWL)", "Wind Drifted Snow", "Cornice",
                "Glide", "Normal Caution"]
AVY_LAYERS = ["New Snow (NS)", "Density Change", "Old Snow Interface", "Facets", "Depth Hoar",
              "Surface Hoar", "Graupel", "Wet Grains", "Ground Interface"]
AVY_TRIGGERS = ["Unknown", "Natural", "Skier", "Snowboarder", "Snowmobile", "Snow Bike",
                "Snow Shoe", "Explosive", "Hiker", "Snow Cat", "Vehicle", "Other"]
ASPECT_DEG = {"North": 0, "Northeast": 45, "East": 90, "Southeast": 135, "South": 180,
              "Southwest": 225, "West": 270, "Northwest": 315}
DEPTH_CMAP = LinearSegmentedColormap.from_list("depth", ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DEPTH_NORM = Normalize(vmin=0, vmax=36, clip=True)  # crown depth, inches; 36"+ = darkest


def fetch_avalanches(days: int = 7, region: str = AVY_REGION):
    """Avalanches reported to the UAC in `region` over the past `days` days: the same data
    the observations explorer shows (and exports as CSV). Avalanches carry numeric location
    IDs, so the explorer page's region list maps "Salt Lake" to all the IDs inside it."""
    html = get(UAC_EXPLORER).text
    i = html.index('"regions":[') + len('"regions":')
    regions, _ = json.JSONDecoder().raw_decode(html, i)
    tids = set(next(g for g in regions if g["name"] == region and g["depth"] == 0)["tids"])
    end = dt.datetime.now(TZ).date()
    start = end - dt.timedelta(days=days)
    allav = get(UAC_AVY_API, params={"start_date": start.isoformat(),
                                     "end_date": end.isoformat()}).json()
    return [a for a in allav if set(a.get("region") or []) & tids], start, end


def _avy_label(value: str, options: list[str]) -> str:
    """Map a UAC value onto one of the listed labels, e.g. "Persistent Weak Layer" ->
    "Persistent Weak Layer (PWL)", "New Snow/Old Snow Interface" -> "Old Snow Interface"."""
    v = value.lower()
    for o in options:
        if v in (o.lower(), re.sub(r"\s*\(.*\)", "", o).lower()):
            return o
    for o in options:
        base = re.sub(r"\s*\(.*\)", "", o).lower()
        if base != "new snow" and base in v:
            return o
    if "new snow" in v:
        return next(o for o in options if o.startswith("New Snow"))
    return "Other" if "Other" in options else f"Other: {value}"


def _avy_tally(values, options):
    counts, unreported = {}, 0
    for v in values:
        vs = [x for x in (v if isinstance(v, list) else [v]) if x]
        if not vs:
            unreported += 1
        for x in vs:
            k = _avy_label(x, options)
            counts[k] = counts.get(k, 0) + 1
    extra = [k for k in counts if k not in options]
    return [(o, counts.get(o, 0)) for o in options + extra], unreported


# Problem colors for the per-avalanche plot. Fixed per problem (not by this week's
# ranking) so a color always means the same thing. These six were chosen so every pair
# stays distinct for normal vision (OKLab Delta E >= 15.6); under color blindness some
# pairs get close (worst ~7), so each problem also has its own marker shape. No seventh
# color clears that bar, so Normal Caution and any problem name not listed here share
# dark gray and are told apart by shape.
PROBLEM_STYLE = {
    "Persistent Weak Layer": ("#2a78d6", "o"),   # blue circle
    "Wind Drifted Snow": ("#e34948", "s"),       # red square
    "New Snow": ("#1baf7a", "^"),                # aqua triangle
    "Cornice": ("#eda100", "p"),                 # yellow pentagon
    "Glide": ("#008300", "v"),                   # green down-triangle
    "Normal Caution": ("#555555", "D"),          # dark gray diamond
    "Other problem": ("#555555", "X"),           # dark gray X
    "No problem reported": ("#8f8f8f", "o"),     # drawn as an unfilled ring
}
E_TOP, E_BOTTOM = 13000, 4000


def _aspect_axes(fig, rect, title):
    """Polar aspect/elevation axes: N at top, clockwise, 13,000 ft at the centre."""
    ax = fig.add_axes(rect, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, E_TOP - E_BOTTOM)
    ring_elevs = list(range(12000, E_BOTTOM - 1, -1000))
    ax.set_yticks([E_TOP - e for e in ring_elevs])
    ax.set_yticklabels([f"{e // 1000}k" if e % 2000 == 0 else "" for e in ring_elevs],
                       fontsize=6, color=C_MUTED)
    ax.set_rlabel_position(202.5)
    ax.set_xticks(np.deg2rad(list(ASPECT_DEG.values())),
                  ["N", "NE", "E", "SE", "S", "SW", "W", "NW"], fontsize=8.5)
    ax.tick_params(axis="x", pad=-1)
    ax.grid(color="#cfcfcf", lw=0.7)
    for e in (8000, 10000, 12000):  # emphasise every 2,000 ft
        ax.plot(np.linspace(0, 2 * np.pi, 200), [E_TOP - e] * 200, color="#a8a8a8", lw=1.1, zorder=1)
    ax.spines["polar"].set_color("#8c8c8c")
    ax.set_title(title, fontsize=10, weight="bold", pad=12)
    return ax


def _problem_of(a) -> str:
    probs = [p for p in (a.get("avalancheProblem") or []) if p]
    if not probs:
        return "No problem reported"
    first = probs[0]  # an avalanche listing several problems is shown by the first
    for name in PROBLEM_STYLE:  # tolerate small wording differences, e.g. "Cornice Fall"
        if name.lower() in first.lower() or first.lower() in name.lower():
            return name
    return "Other problem"


def avalanche_activity_section(rep: Report, days: int = 7, title: str | None = None,
                               section: str | None = None):
    """Avalanche page for the past `days` days (title/section override the wording, e.g.
    for the season report)."""
    av, start, end = fetch_avalanches(days)
    n = len(av)
    fig = plt.figure(figsize=LAND)
    header(fig, f"{title or f'Avalanches Past {days} Days'} — {AVY_REGION}",
           f"{n} avalanche{'s' if n != 1 else ''} reported to the Utah Avalanche Center, "
           f"{start:%b %d} – {end:%b %d %Y}  ·  {UAC_EXPLORER}")

    placed = [a for a in av if a.get("aspect") in ASPECT_DEG and a.get("elevation")
              and E_BOTTOM <= a["elevation"] <= E_TOP]
    unplotted = n - len(placed)

    # ---- plot 1: one dot per aspect x 1,000-ft band, area = count, color = mean depth
    ax = _aspect_axes(fig, [0.035, 0.47, 0.29, 0.375], "Avalanches by Aspect and Elevation")
    area, c_unknown = 60, "#c4c4c4"
    groups = {}
    for a in placed:
        band = min(int(a["elevation"] // 1000) * 1000, E_TOP - 1000)
        try:
            depth = float(a.get("depthKey") or 0)
        except ValueError:
            depth = 0
        g = groups.setdefault((a["aspect"], band), {"n": 0, "depths": []})
        g["n"] += 1
        if depth > 0:
            g["depths"].append(depth)
    for (asp, band), g in sorted(groups.items(), key=lambda kv: -kv[1]["n"]):
        r = E_TOP - (band + 500)
        color = DEPTH_CMAP(DEPTH_NORM(np.mean(g["depths"]))) if g["depths"] else c_unknown
        ax.scatter(np.deg2rad(ASPECT_DEG[asp]), r, s=area * g["n"], color=color, alpha=0.9,
                   edgecolor="white", lw=1.0, zorder=3)
        if g["n"] > 1:
            ax.text(np.deg2rad(ASPECT_DEG[asp]), r, str(g["n"]), ha="center", va="center",
                    fontsize=6.5, weight="bold", color="white" if g["depths"] else C_TEXT, zorder=4)
    # keys to the right of plot 1: dot size = count, color = average crown depth
    kx = 0.345
    fig.text(kx, 0.79, "Avalanches per dot", fontsize=7, weight="bold", color=C_TEXT)
    for i, k in enumerate((1, 2, 4)):
        y = 0.755 - i * 0.04
        fig.add_artist(plt.Line2D([kx + 0.012], [y], marker="o", ls="", markersize=np.sqrt(area * k),
                                  color="#9a9a9a", markeredgecolor="white", transform=fig.transFigure))
        fig.text(kx + 0.035, y, str(k), va="center", fontsize=7, color=C_TEXT)
    fig.text(kx, 0.62, "Avg crown depth", fontsize=7, weight="bold", color=C_TEXT)
    cax = fig.add_axes([kx + 0.004, 0.5, 0.012, 0.105])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=DEPTH_NORM, cmap=DEPTH_CMAP), cax=cax)
    cb.set_ticks([0, 12, 24, 36], labels=['0"', '12"', '24"', '36"+'])
    cb.ax.tick_params(labelsize=6, length=2, pad=1)
    cb.outline.set_linewidth(0.4)
    fig.add_artist(plt.Line2D([kx + 0.01], [0.475], marker="o", ls="", markersize=5, color=c_unknown,
                              transform=fig.transFigure))
    fig.text(kx + 0.022, 0.475, "unknown", va="center", fontsize=6.5, color=C_MUTED)

    # ---- plot 2: every avalanche as its own dot, colored/shaped by its problem
    bx = _aspect_axes(fig, [0.5, 0.47, 0.29, 0.375], "Each Avalanche by Problem")
    counts = {}
    slots = {}  # nudge avalanches at the same aspect and similar elevation apart

    def days_ago(a) -> int:
        try:
            d = dt.datetime.strptime(str(a.get("date")), "%Y%m%d").date()
        except ValueError:
            return days
        return max(0, min(days, (end - d).days))

    def age_alpha(age: int) -> float:  # reported today 1.0 ... `days` ago 0.2
        return 1.0 - 0.8 * age / days

    # oldest first, so each newer avalanche is drawn on top of older ones
    for rank, a in enumerate(sorted(placed, key=lambda a: (-days_ago(a), a["elevation"]))):
        prob = _problem_of(a)
        counts[prob] = counts.get(prob, 0) + 1
        key = (a["aspect"], round(a["elevation"] / 250))
        k = slots.get(key, 0)
        slots[key] = k + 1
        offset = (k + 1) // 2 * 7 * (1 if k % 2 else -1)  # 0, +7, -7, +14, ... degrees
        color, marker = PROBLEM_STYLE[prob]
        hollow = prob == "No problem reported"
        bx.scatter(np.deg2rad(ASPECT_DEG[a["aspect"]] + offset), E_TOP - a["elevation"], s=80,
                   marker=marker, facecolor="none" if hollow else color,
                   edgecolor=color if hollow else "white", lw=1.5 if hollow else 0.9,
                   alpha=age_alpha(days_ago(a)), zorder=3 + rank)
    handles = [plt.Line2D([], [], ls="", marker=PROBLEM_STYLE[p][1], markersize=7,
                          markerfacecolor="none" if p == "No problem reported" else PROBLEM_STYLE[p][0],
                          markeredgecolor=PROBLEM_STYLE[p][0] if p == "No problem reported" else "white",
                          label=f"{p} ({counts.get(p, 0)})")
               for p in PROBLEM_STYLE]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.805, 0.82), fontsize=7,
               frameon=False, title="Avalanche problem", title_fontsize=7.5, alignment="left",
               labelspacing=0.75, handletextpad=0.5)
    fig.text(0.81, 0.535, "Other problem = any problem not\nlisted above. An avalanche listing\n"
                          "several problems is shown by\nthe first one. Newest on top.",
             fontsize=6.3, color=C_MUTED, va="top")
    # age key: markers fade from today (opaque) to `days` ago
    fig.text(0.81, 0.47, "Reported (older = fainter):", fontsize=6.5, weight="bold",
             color=C_TEXT, va="center")
    for i, age in enumerate((0, 2, 4, days) if days <= 14 else (0, days // 3, 2 * days // 3, days)):
        x = 0.817 + i * 0.04
        fig.add_artist(plt.Line2D([x], [0.445], marker="o", ls="", markersize=6.5,
                                  color=PROBLEM_STYLE["Persistent Weak Layer"][0],
                                  markeredgecolor="white", alpha=age_alpha(age),
                                  transform=fig.transFigure))
        fig.text(x + 0.008, 0.445, "today" if age == 0 else f"{age}d", fontsize=6,
                 color=C_TEXT, va="center")

    if not av:
        for a_ in (ax, bx):
            a_.text(0, 0, "No avalanches reported", ha="center", va="center", fontsize=9, color=C_MUTED)
    note = "Rings: elevation, 13,000 ft at the centre to 4,000 ft at the edge"
    if unplotted:
        note += f"  ·  {unplotted} without aspect/elevation not shown"
    fig.text(0.42, 0.405, note, ha="center", fontsize=7, color=C_MUTED)

    # ---- four bar charts in a row along the bottom
    charts = [
        ("Avalanches by Type", _avy_tally([a.get("avalancheType") for a in av], AVY_TYPES)),
        ("Avalanches by Problem", _avy_tally([a.get("avalancheProblem") for a in av], AVY_PROBLEMS)),
        ("Avalanches by Weak Layer", _avy_tally([a.get("weakLayer") for a in av], AVY_LAYERS)),
        ("Avalanches by Trigger", _avy_tally([a.get("trigger") for a in av], AVY_TRIGGERS)),
    ]
    for (title, (rows, unreported)), x0 in zip(charts, (0.13, 0.375, 0.62, 0.865)):
        y0, h = 0.07, 0.27
        cx = fig.add_axes([x0, y0, 0.1, h])
        cx.set_axisbelow(True)
        labels = [r[0] for r in rows][::-1]
        vals = [r[1] for r in rows][::-1]
        top = max(vals + [1])
        cx.barh(labels, vals, color=C_COOL, height=0.65)
        for yi, v in enumerate(vals):
            if v:
                cx.text(v + top * 0.04, yi, str(v), va="center", fontsize=6, color=C_TEXT)
        cx.set_xlim(0, top * 1.25)
        cx.tick_params(axis="y", labelsize=6, length=0)
        cx.tick_params(axis="x", labelsize=5.5)
        cx.xaxis.get_major_locator().set_params(integer=True)
        cx.grid(axis="y", visible=False)
        fig.text(x0 - 0.115, y0 + h + 0.025, title, fontsize=8.8, weight="bold", color=C_TEXT)
        if unreported:
            cx.text(1.0, -0.12, f"{unreported} of {n} not reported", transform=cx.transAxes,
                    ha="right", fontsize=6, color=C_MUTED)
    rep.save(fig, section or f"Avalanches past {days} days (UAC)")


# --------------------------------------------------------------------------- forecast: AFD
def afd_section(rep: Report):
    s = soup_of(AFD_URL)
    pre = s.find("pre", class_="glossaryProduct") or s.find("pre")
    text = pre.get_text().strip("\n")
    text_pages(rep, "NWS Area Forecast Discussion — Salt Lake City", text, AFD_URL, mono=True,
               section="NWS forecast discussion (AFD)")


# --------------------------------------------------------------------------- forecast: NWS point
def parse_digital(soup: BeautifulSoup) -> pd.DataFrame:
    """Parse the NWS 'digital' tabular forecast into an hourly DataFrame."""
    rows_out: dict[str, list] = {}
    for table in soup.find_all("table"):
        trs = table.find_all("tr")
        if len(trs) < 15 or "Hour" not in table.get_text():
            continue
        block: dict[str, list] = {}
        for tr in trs:
            cells = [c.get_text(strip=True) for c in tr.find_all("td")]
            if len(cells) < 2:
                if block:
                    for k, v in block.items():
                        rows_out.setdefault(k, []).extend(v)
                    block = {}
                continue
            label = cells[0].replace("°", "").replace("�", "")
            label = re.sub(r"\s+", " ", label)
            block[label] = cells[1:]
        if block:
            for k, v in block.items():
                rows_out.setdefault(k, []).extend(v)
    if not rows_out:
        raise RuntimeError("Couldn't find the tabular forecast on the NWS page")
    n = min(len(v) for v in rows_out.values())
    df = pd.DataFrame({k: v[:n] for k, v in rows_out.items()})

    # build timestamps: date row carries forward
    now = dt.datetime.now(TZ)
    times, cur_date = [], None
    for d, h in zip(df["Date"], df[[c for c in df.columns if c.startswith("Hour")][0]]):
        if d:
            m, day = map(int, d.split("/"))
            yr = now.year + (1 if m < now.month - 6 else 0)
            cur_date = dt.date(yr, m, day)
        if cur_date is None:
            cur_date = now.date()
        times.append(dt.datetime.combine(cur_date, dt.time(int(h)), TZ))
    df.index = pd.DatetimeIndex(times)

    def num(col_start):
        cols = [c for c in df.columns if c.startswith(col_start)]
        return pd.to_numeric(df[cols[0]], errors="coerce") if cols else pd.Series(np.nan, index=df.index)

    out = pd.DataFrame({
        "temp": num("Temperature"), "dewpt": num("Dewpoint"), "chill": num("Wind Chill"),
        "wind": num("Surface Wind"), "gust": num("Gust"), "sky": num("Sky Cover"),
        "pop": num("Precipitation Potential"), "rh": num("Relative Humidity"),
    }, index=df.index)
    out["wdir"] = df.get("Wind Dir")
    for col in ["Rain", "Snow", "Thunder", "Freezing Rain", "Sleet"]:
        if col in df:
            out[col.lower().replace(" ", "_")] = df[col].replace("--", "")
    return out[~out.index.duplicated()].sort_index()


def _is_digital_table(r: requests.Response) -> bool:
    """True for a real tabular-forecast page (not e.g. an NWS "Runtime Error" page)."""
    return "Hour" in r.text and "Temperature" in r.text


def fetch_point_forecast_table(pt: Point) -> tuple[pd.DataFrame, int | None]:
    """All 7 days of the NWS tabular forecast for a point (the page shows 48 h at a
    time), plus the NWS grid-cell elevation printed on the page. Raises unless every
    48-hour page loads, so a partly failed table falls back to the NWS API."""
    frames, grid_elev = [], None
    for ah in (0, 48, 96, 144):
        r = get(pt.url + f"&AheadHour={ah}", validate=_is_digital_table)
        r.encoding = r.apparent_encoding if (r.encoding or "").lower() in ("", "iso-8859-1") else r.encoding
        soup = BeautifulSoup(r.text, "html.parser")
        if grid_elev is None:
            m = re.search(r"Elev\.?\s*([\d,]+)\s*ft", soup.get_text(" "))
            grid_elev = int(m.group(1).replace(",", "")) if m else None
        frames.append(parse_digital(soup))
    fc = pd.concat(frames)
    return fc[~fc.index.duplicated()].sort_index(), grid_elev


# NWS API weather "coverage" -> the abbreviations the tabular forecast uses
_COVERAGE = {"slight_chance": "SChc", "chance": "Chc", "likely": "Lkly", "definite": "Def",
             "occasional": "Ocnl", "isolated": "Iso", "scattered": "Sct", "numerous": "Num",
             "areas": "Areas", "patchy": "Patchy", "widespread": "Wide", "frequent": "Frq",
             "periods": "Pds", "brief": "Brf", "intermittent": "Inter"}
_WEATHER_COL = {"snow": "snow", "snow_showers": "snow", "blowing_snow": "snow",
                "rain": "rain", "rain_showers": "rain", "drizzle": "rain",
                "thunderstorms": "thunder", "freezing_rain": "freezing_rain",
                "freezing_drizzle": "freezing_rain", "sleet": "sleet", "ice_pellets": "sleet"}
_COMPASS16 = "N NNE NE ENE E ESE SE SSE S SSW SW WSW W WNW NW NNW".split()


def _iso_hours(duration: str) -> int:
    """ISO 8601 duration like PT3H, P2D or P4DT12H -> whole hours."""
    m = re.fullmatch(r"P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?)?", duration)
    if not m:
        return 1
    d, h, mins = (int(x) if x else 0 for x in m.groups())
    return max(1, d * 24 + h + (1 if mins else 0))


def _expand(layer: dict, hours: pd.DatetimeIndex) -> pd.Series:
    """Spread an NWS API grid layer ("from T for N hours: value") over hourly times."""
    out = pd.Series(np.nan, index=hours, dtype=object)
    for v in (layer or {}).get("values", []):
        start, dur = v["validTime"].split("/")
        t0 = pd.Timestamp(start).tz_convert(TZ)
        t1 = t0 + pd.Timedelta(hours=_iso_hours(dur))
        for t in hours[(hours >= t0) & (hours < t1)]:
            out.at[t] = v["value"]  # one cell at a time: values can be lists (weather)
    return out


def fetch_point_forecast_api(pt: Point) -> tuple[pd.DataFrame, int | None]:
    """The same NWS forecast for a point from the NWS API (api.weather.gov), shaped like
    the tabular forecast so the charts and daily summary don't change. The API runs on
    separate servers from forecast.weather.gov, so it's a useful fallback. Values match
    the table; the one visible difference is that the API always reports a gust, while
    the table only shows notable ones."""
    meta = get(f"{NWS_API}/points/{pt.lat},{pt.lon}", headers=NWS_API_HEADERS).json()
    grid = get(meta["properties"]["forecastGridData"], headers=NWS_API_HEADERS).json()["properties"]

    now = pd.Timestamp.now(tz=TZ).floor("h")
    hours = pd.date_range(now, now + pd.Timedelta(days=7) - pd.Timedelta(hours=1), freq="h")

    def num(name):
        return pd.to_numeric(_expand(grid.get(name), hours), errors="coerce")

    def f_from_c(s):
        return (s * 9 / 5 + 32).round()

    def mph(s):
        return (s / 1.609344).round()

    wdir = num("windDirection")
    fc = pd.DataFrame({
        "temp": f_from_c(num("temperature")), "dewpt": f_from_c(num("dewpoint")),
        "chill": f_from_c(num("windChill")), "wind": mph(num("windSpeed")),
        "gust": mph(num("windGust")), "sky": num("skyCover").round(),
        "pop": num("probabilityOfPrecipitation").round(), "rh": num("relativeHumidity").round(),
        "wdir": [(_COMPASS16[int((d % 360) / 22.5 + 0.5) % 16] if np.isfinite(d) else "")
                 for d in wdir],
    }, index=hours)
    # precipitation types, e.g. snow "SChc", in the tabular forecast's columns
    for col in set(_WEATHER_COL.values()):
        fc[col] = ""
    for t, conds in _expand(grid.get("weather"), hours).items():
        for c in conds if isinstance(conds, list) else []:
            col = _WEATHER_COL.get(c.get("weather") or "")
            if col and c.get("coverage"):
                fc.at[t, col] = _COVERAGE.get(c["coverage"], c["coverage"].title())
    fc = fc.dropna(subset=["temp"])
    if fc.empty:
        raise RuntimeError(f"NWS API returned no forecast values for {pt.label}")
    elev = (grid.get("elevation") or {}).get("value")
    return fc, (round(elev * 3.28084) if elev is not None else None)


def fetch_point_forecast(pt: Point) -> tuple[pd.DataFrame, int | None, str]:
    """The tabular forecast page first (after retries); the NWS API if it still fails.
    Returns (forecast, NWS grid elevation in ft, source)."""
    try:
        fc, elev = fetch_point_forecast_table(pt)
        return fc, elev, "forecast.weather.gov"
    except Exception as e:
        print(f"  forecast table failed for {pt.label} ({e}); trying the NWS API", file=sys.stderr)
    fc, elev = fetch_point_forecast_api(pt)
    return fc, elev, "NWS API"


def point_forecast_section(rep: Report, pt: Point, section: str):
    fc, grid_elev, source = fetch_point_forecast(pt)
    title = f"NWS Point Forecast — {pt.label} (elev {pt.elev:,} ft)"
    where = f"{pt.lat}, {pt.lon}" + (f"  ·  NWS grid elevation {grid_elev:,} ft" if grid_elev else "")
    if source != "forecast.weather.gov":
        where += f"  ·  source: {source} (forecast table unavailable)"
    forecast_charts_page(rep, fc, title, where, section)
    return "ok" if source == "forecast.weather.gov" else f"ok ({source})"


def forecast_charts_page(rep: Report, fc: pd.DataFrame, title: str, where: str, section: str):
    # left column: the whole 7-day forecast; right column: the next 24 hours in detail
    fig = plt.figure(figsize=PAGE)
    gs = fig.add_gridspec(4, 2, width_ratios=[1.85, 1], height_ratios=[3, 2.4, 1.6, 2],
                          hspace=0.22, wspace=0.13, left=0.075, right=0.975, top=0.89,
                          bottom=0.315)
    axes = [fig.add_subplot(gs[0, 0])]
    axes += [fig.add_subplot(gs[i, 0], sharex=axes[0]) for i in range(1, 4)]
    right = [fig.add_subplot(gs[0, 1])]
    right += [fig.add_subplot(gs[i, 1], sharex=right[0]) for i in range(1, 4)]
    header(fig, title,
           f"{where}  ·  {fc.index[0]:%a %b %d %H:%M} – {fc.index[-1]:%a %b %d %H:%M %Z}")
    fig.text(0.075, 0.905, "7-day forecast", fontsize=9.5, weight="bold", va="bottom")
    fig.text(0.685, 0.905, "Next 24 hours", fontsize=9.5, weight="bold", va="bottom")
    next24_charts(right, fc.iloc[:24])
    ax = axes[0]
    ax.plot(fc.index, fc.temp, color=C_WARM, lw=1.8, label="Temperature")
    ax.plot(fc.index, fc.dewpt, color="#3a7d44", lw=1.1, label="Dew point")
    ax.plot(fc.index, fc.chill, color=C_COOL, lw=1.1, ls="--", label="Wind chill")
    ax.axhline(32, color="#999999", lw=0.8, ls=":")
    ax.set_ylabel("°F"); ax.legend(loc="upper left", ncol=3, fontsize=7.5, frameon=False)

    ax = axes[1]
    ax.plot(fc.index, fc.wind, color=C_MEAN, lw=1.6)
    has_gust = fc.gust.notna().any()
    if has_gust:
        ax.scatter(fc.index, fc.gust, color=C_GUST, marker="^", s=14, zorder=3)
    ymax = np.nanmax([fc.wind.max(), fc.gust.max() if has_gust else 0]) * 1.45 + 3
    ax.set_ylim(0, ymax)
    step = max(1, len(fc) // 28)
    for t, d in zip(fc.index[::step], fc.wdir[::step]):
        if isinstance(d, str) and d:
            ax.text(t, ymax * 0.97, d, fontsize=5.2, ha="center", va="top", color=C_MUTED,
                    rotation=90)
    ax.set_ylabel("mph")
    ax.set_title("Wind (line)" + (", gusts (▲)" if has_gust else ", no gusts forecast")
                 + ", direction (top)", loc="left", fontsize=7.5, color=C_MUTED)

    ax = axes[2]
    ax.bar(fc.index, fc["pop"], width=1 / 24, color=C_COOL, alpha=0.75, label="Precip. potential")
    ax.set_ylim(0, 100); ax.set_ylabel("%")
    ax.legend(loc="upper left", fontsize=7.5, frameon=False)

    ax = axes[3]
    ax.fill_between(fc.index, fc.sky, color="#9aa5b1", alpha=0.45, label="Sky cover")
    ax.plot(fc.index, fc.rh, color="#3a7d44", lw=1.2, label="Rel. humidity")
    ax.set_ylim(0, 100); ax.set_ylabel("%")
    ax.legend(loc="upper left", ncol=2, fontsize=7.5, frameon=False)
    ax.xaxis.set_major_locator(mdates.DayLocator(tz=TZ))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%b %d", tz=TZ))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18], tz=TZ))
    ax.tick_params(axis="x", labelsize=7.5)
    for a in axes:
        a.grid(True, which="major", axis="x", color="#bbbbbb")
    for a in axes[:-1]:
        plt.setp(a.get_xticklabels(), visible=False)
    draw_daily_summary(fig, fc, top=0.255)
    rep.save(fig, section)


def mark_steepest_rise(ax, temp: pd.Series):
    """Highlight the largest rise between two consecutive forecast hours: a shaded hour,
    a thick segment over the line, ringed end points and a "+N°F" label at the top of
    the shaded hour. Ties mark the first one; nothing is marked if it never warms."""
    rise = temp.diff()
    if not (rise > 0).any():
        return
    i = int(np.nanargmax(rise.to_numpy()))
    t0, t1 = temp.index[i - 1], temp.index[i]
    v0, v1 = temp.iloc[i - 1], temp.iloc[i]
    ax.axvspan(t0, t1, color="#ec8600", alpha=0.15, lw=0, zorder=1)
    ax.plot([t0, t1], [v0, v1], color="#d74700", lw=3, solid_capstyle="round", zorder=4)
    ax.scatter([t0, t1], [v0, v1], s=18, facecolor="white", edgecolor="#d74700", lw=1.2, zorder=5)
    # top of the shaded hour, or a little lower if the legend sits up there
    y = 0.72 if i < len(temp) / 2 else 0.965  # legend occupies the top-left half
    ax.text(t0 + (t1 - t0) / 2, y, f"+{v1 - v0:.0f}°F",
            transform=mtransforms.blended_transform_factory(ax.transData, ax.transAxes),
            ha="center", va="top", fontsize=6.5, color="#b83c00", weight="bold", zorder=6,
            path_effects=[patheffects.withStroke(linewidth=2.2, foreground="#fdebd8")])


def next24_charts(axes, fc: pd.DataFrame):
    """Hour-by-hour detail for the first 24 forecast hours: temperature, wind/gusts with
    direction, precipitation potential and sky cover, with values printed on the bars
    and points."""
    t = fc.index
    lbl = dict(fontsize=5.3, ha="center", color=C_TEXT)

    ax = axes[0]  # temperature (+ wind chill)
    ax.plot(t, fc.temp, color=C_WARM, lw=1.6, marker="o", ms=2.4, label="Temp")
    if fc.chill.notna().any():
        ax.plot(t, fc.chill, color=C_COOL, lw=1, ls="--", label="Wind chill")
    ax.axhline(32, color="#999999", lw=0.8, ls=":")
    lo = np.nanmin([fc.temp.min(), fc.chill.min()])
    hi = fc.temp.max()
    ax.set_ylim(lo - 3, hi + (hi - lo) * 0.35 + 4)
    for x, v in zip(t[::2], fc.temp[::2]):
        if np.isfinite(v):
            ax.text(x, v + 1, f"{v:.0f}", va="bottom", **lbl)
    ax.legend(loc="upper left", fontsize=6.5, frameon=False, ncol=2)
    mark_steepest_rise(ax, fc.temp)

    ax = axes[1]  # wind, gusts, direction
    ax.plot(t, fc.wind, color=C_MEAN, lw=1.5, marker="o", ms=2.2)
    has_gust = fc.gust.notna().any()
    if has_gust:
        ax.scatter(t, fc.gust, color=C_GUST, marker="^", s=12, zorder=3)
    ymax = np.nanmax([fc.wind.max(), fc.gust.max() if has_gust else 0, 4]) * 1.6 + 2
    ax.set_ylim(0, ymax)
    for x, v, d in zip(t[::2], fc.wind[::2], fc.wdir[::2]):
        if np.isfinite(v):
            ax.text(x, v + ymax * 0.04, f"{v:.0f}", va="bottom", **lbl)
        if isinstance(d, str) and d:
            ax.text(x, ymax * 0.97, d, fontsize=4.8, ha="center", va="top", color=C_MUTED,
                    rotation=90)
    ax.set_title("mph" + (", gusts ▲" if has_gust else ""), loc="left", fontsize=7,
                 color=C_MUTED)

    ax = axes[2]  # precipitation potential
    ax.bar(t, fc["pop"], width=0.8 / 24, color=C_COOL, alpha=0.8)
    for x, v in zip(t, fc["pop"]):
        if np.isfinite(v) and v > 0:
            ax.text(x, v + 3, f"{v:.0f}", va="bottom", **lbl)
    ax.set_ylim(0, 115)
    ax.set_yticks([0, 50, 100])
    ax.set_title("Precip. potential %", loc="left", fontsize=7, color=C_MUTED)

    ax = axes[3]  # sky cover
    ax.bar(t, fc.sky, width=0.8 / 24, color="#9aa5b1", alpha=0.7)
    for x, v in zip(t[::2], fc.sky[::2]):
        if np.isfinite(v):
            ax.text(x, v + 3, f"{v:.0f}", va="bottom", **lbl)
    ax.set_ylim(0, 115)
    ax.set_yticks([0, 50, 100])
    ax.set_title("Sky cover %", loc="left", fontsize=7, color=C_MUTED)

    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 3), tz=TZ))
    ax.xaxis.set_minor_locator(mdates.HourLocator(tz=TZ))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H", tz=TZ))
    ax.set_xlim(t[0] - pd.Timedelta(minutes=40), t[-1] + pd.Timedelta(minutes=40))
    ax.set_xlabel(f"hour ({t[0]:%Z}), starting {t[0]:%a %H:%M}", fontsize=7)
    for a in axes:
        a.tick_params(axis="both", labelsize=6.5)
        a.grid(True, which="major", axis="x", color="#bbbbbb")
    for a in axes[:-1]:
        plt.setp(a.get_xticklabels(), visible=False)


PRECIP_COLS = [("rain", "Rain"), ("snow", "Snow"), ("thunder", "Thunder"),
               ("freezing_rain", "Frz rain"), ("sleet", "Sleet")]


def _draw_table(ax, header_row, rows, widths, fontsize=7, shade=None, bold=None):
    """Table that exactly fills `ax` (one equal-height row per line)."""
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=header_row, colWidths=widths, cellLoc="center",
                   bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    for (r, c), cl in tbl.get_celld().items():
        cl.set_edgecolor("#d4d4d4")
        cl.set_linewidth(0.4)
        if r == 0:
            cl.set_facecolor("#e8eef3")
            cl.set_text_props(weight="bold")
        else:
            if shade is not None and shade[r - 1]:
                cl.set_facecolor("#f3f5f7")
            if bold is not None and bold[r - 1] and c == 0:
                cl.set_text_props(weight="bold")
    return tbl


def _fmt(v, nd=0):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def daily_summary_rows(fc: pd.DataFrame) -> list[list[str]]:
    """One row per forecast day: high/low, max wind and gust, max PoP, mean sky cover
    and the precip types with their NWS chance wording."""
    daily = fc.groupby(fc.index.date).agg(
        high=("temp", "max"), low=("temp", "min"), wind_max=("wind", "max"),
        gust_max=("gust", "max"), pop_max=("pop", "max"), sky_mean=("sky", "mean"))
    rows = []
    for d, r in daily.iterrows():
        day = fc[fc.index.date == d]
        kinds = []
        for k, lbl in PRECIP_COLS:
            if k in day and day[k].astype(str).str.strip().ne("").any():
                vals = sorted(set(v for v in day[k] if v))
                kinds.append(f"{lbl.lower()} ({'/'.join(vals)})")
        rows.append([pd.Timestamp(d).strftime("%a %b %d"), _fmt(r.high), _fmt(r.low),
                     _fmt(r.wind_max), _fmt(r.gust_max) or "—", _fmt(r.pop_max),
                     _fmt(r.sky_mean), ", ".join(kinds) or "—"])
    return rows


def draw_daily_summary(fig, fc: pd.DataFrame, top: float):
    """Daily summary table across the page, starting just below figure height `top`."""
    rows = daily_summary_rows(fc)
    fig.text(0.06, top, "Daily summary", fontsize=10, weight="bold", va="top")
    h = 0.019 * (len(rows) + 1)
    ax = fig.add_axes([0.06, top - 0.022 - h, 0.89, h])
    _draw_table(ax, ["Day", "High °F", "Low °F", "Max wind", "Max gust", "Max PoP %",
                     "Sky %", "Precip type (chance)"], rows,
                [0.12, 0.08, 0.08, 0.09, 0.09, 0.1, 0.07, 0.37], fontsize=7.2)
    fig.text(0.06, top - 0.022 - h - 0.016,
             "Wind in mph. Precip chances use NWS wording: SChc = slight chance, Chc = chance, "
             "Lkly = likely, Ocnl = occasional, Def = definite.", fontsize=6.8, color=C_MUTED,
             va="top")


# --------------------------------------------------------------------------- forecast: plumes
# weather.utah.edu plume products: model code -> (page title, contents entry)
PLUME_MODELS = {
    "ensgefsds": ("Utah Snow Ensemble (GEFS downscaled) Plumes", "Ensemble plumes (U of U)"),
    "rrfsqsf": ("RRFS Hourly Snow Ensemble Plumes", "RRFS hourly snow plumes (U of U)"),
}


def _plume_images(html: str) -> list[str]:
    """Image paths the plume page lists in its images_vis array."""
    m = re.search(r"images_vis:\s*\[(.*?)\]", html, re.S)
    return re.findall(r'"([^"]+\.(?:png|gif|jpg))"', m.group(1)) if m else []


def plume_section(rep: Report, station: str, name: str, model: str = "ensgefsds"):
    title, section = PLUME_MODELS[model]
    # Without a runcode the page serves the latest run; we still read the run from the HTML.
    paths = _plume_images(get(PLUME_PAGE, params={"t": model, "d": "PL", "r": station}).text)
    if not paths:
        # try today's and the previous two days' runs explicitly
        for back in range(0, 3):
            day = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=back)
            for hh in ("18", "12", "06", "00"):
                rc = f"{day:%Y%m%d}{hh}"
                paths = _plume_images(get(PLUME_PAGE, params={"runcode": rc, "t": model, "d": "PL",
                                                              "r": station}).text)
                if paths:
                    break
            if paths:
                break
    if not paths:
        raise RuntimeError(f"No {model} plume images found on weather.utah.edu for {station}")
    imgs = [fetch_image(requests.compat.urljoin(PLUME_PAGE, p)) for p in paths]
    run = re.search(r"(\d{10})F\d+", paths[0])
    run_txt = f"run {run.group(1)[:8]} {run.group(1)[8:]}Z" if run else ""
    for i, img in enumerate(imgs):
        image_page(rep, f"{title} — {name}", [img],
                   f"University of Utah  ·  {run_txt}  ·  weather.utah.edu (t={model}, d=PL, r={station})",
                   section=section if i == 0 else None,
                   landscape=img.width > img.height)


# --------------------------------------------------------------------------- history: charts
# Box color on the station temperature charts: the day's steepest warming between
# consecutive 2-hour mean temperatures. Blue = slow, orange/red-orange = fast. Fixed
# scale so a color means the same rate on every page; 8 °F+ shows as the top color.
WARMING_CMAP = LinearSegmentedColormap.from_list("warming", [
    "#009cc1", "#2bb0c9", "#5fc3d0", "#8ad0c3", "#b1d3a6", "#cbc276", "#daa237", "#ec8600",
    "#d74700"])
WARMING_NORM = Normalize(vmin=0, vmax=8, clip=True)
WARMING_ALPHA = 0.75
C_WHISKER = "#3a3a3a"
_HALO = [patheffects.Stroke(linewidth=4, foreground="white"), patheffects.Normal()]


def steepest_warming(temp: pd.Series) -> pd.Series:
    """Per local calendar day: the largest rise between consecutive 2-hour mean
    temperatures (°F), 0 if the day only cooled. Binned on local clock time so the
    windows stay 00-02, 02-04, ... across daylight-saving changes."""
    s = temp.dropna()
    if s.empty:
        return pd.Series(dtype=float)
    s.index = s.index.tz_convert(TZ).tz_localize(None)
    step = s.resample("2h", origin="start_day").mean().diff().dropna()
    return step.groupby(step.index.normalize()).max().clip(lower=0)


def candlestick_temp(ax, daily: pd.DataFrame, title: str, width: float = 0.62):
    """Daily temperature. With hourly-derived columns (q1, q3, mean, steep) each day is a
    box-and-whisker: box = 25th-75th percentile, whiskers = low/high, dash = mean, box
    color = steepest 2-hour warming that day. With only high/low (NCEI daily data) each
    day is a plain high-low bar."""
    x = mdates.date2num(pd.to_datetime(daily.index))
    has_box = ({"q1", "q3", "mean"} <= set(daily.columns)
               and daily[["q1", "q3"]].notna().any().any())
    for xi, row in zip(x, daily.itertuples()):
        if np.isnan(row.high) or np.isnan(row.low):
            continue
        if has_box and not np.isnan(row.q1):
            steep = getattr(row, "steep", np.nan)
            color = WARMING_CMAP(WARMING_NORM(0 if np.isnan(steep) else steep))
            ax.vlines(xi, row.low, row.q1, color=C_WHISKER, lw=1.0, zorder=2)
            ax.vlines(xi, row.q3, row.high, color=C_WHISKER, lw=1.0, zorder=2)
            ax.hlines([row.low, row.high], xi - width * 0.22, xi + width * 0.22,
                      color=C_WHISKER, lw=1.0, zorder=2)  # whisker caps
            ax.bar(xi, max(row.q3 - row.q1, 0.3), bottom=row.q1, width=width, color=color,
                   alpha=WARMING_ALPHA, edgecolor=C_WHISKER, lw=0.6, zorder=3)
            ax.hlines(row.mean, xi - width * 0.62, xi + width * 0.62, color=C_MEAN, lw=2.0,
                      zorder=4, path_effects=_HALO)
        else:
            ax.bar(xi, row.high - row.low, bottom=row.low, width=0.62, color=C_WARM,
                   alpha=0.8, zorder=3)
        ax.text(xi, row.high + 0.8, f"{row.high:.0f}", ha="center", va="bottom", fontsize=5.5,
                color=C_MUTED)
        ax.text(xi, row.low - 0.8, f"{row.low:.0f}", ha="center", va="top", fontsize=5.5,
                color=C_MUTED)
    ax.axhline(32, color="#999999", lw=0.8, ls=":")
    ax.set_ylabel("°F")
    ax.set_title(title, loc="left", fontsize=10, weight="bold")
    lo, hi = np.nanmin(daily.low), np.nanmax(daily.high)
    if np.isfinite(lo) and np.isfinite(hi):
        ax.set_ylim(lo - 6, hi + (6 + 0.18 * (hi - lo) if has_box else 6))  # room for legend
    if has_box:
        # color key: small horizontal color bar in the top-left, mean-dash key beside it
        cax = ax.inset_axes([0.012, 0.9, 0.2, 0.035])
        cb = ax.figure.colorbar(plt.cm.ScalarMappable(norm=WARMING_NORM, cmap=WARMING_CMAP),
                                cax=cax, orientation="horizontal", alpha=WARMING_ALPHA)
        cb.set_ticks([0, 2, 4, 6, 8], labels=["0", "2", "4", "6", "8+"])
        cb.ax.tick_params(labelsize=6, length=2, pad=1)
        cb.outline.set_linewidth(0.4)
        ax.text(0.218, 0.918, "steepest 2-h warming (°F)", transform=ax.transAxes,
                fontsize=6.5, va="center", color=C_TEXT)
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([], [], color=C_MEAN, lw=2.0, label="Daily mean")],
                  loc="upper left", bbox_to_anchor=(0.5, 1.0), fontsize=7, frameon=False)


def date_axis(ax):
    lo, hi = ax.get_xlim()  # label every day / 2 days / week depending on the span
    span = hi - lo
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1 if span <= 16 else 2 if span <= 45 else 7))
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%d"))
    ax.tick_params(axis="x", labelsize=7)


C_WIND = "#5b8db8"
C_DIR = "#8e6bb8"


def _candle(ax, x, lo, q1, q3, hi, mean, color, width=0.62):
    """Box from q1 to q3, wick from lo to hi, dark line across the box at the mean."""
    ax.vlines(x, lo, hi, color=color, lw=1.1, zorder=2)
    ax.bar(x, max(q3 - q1, 0.3), bottom=q1, width=width, color=color, alpha=0.55,
           edgecolor=color, zorder=3)
    ax.hlines(mean, x - width * 0.62, x + width * 0.62, color=C_MEAN, lw=2.0, zorder=4)


def _mean_legend(ax, body_color, body_label, wick_label, extra=()):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=body_color, alpha=0.55, edgecolor=body_color, label=body_label),
               Line2D([], [], color=body_color, lw=1.1, label=wick_label),
               Line2D([], [], color=C_MEAN, lw=2.0, label="Daily mean"), *extra]
    ax.legend(handles=handles, loc="upper left", fontsize=7, frameon=False, ncol=len(handles))


def wind_chart(ax, daily: pd.DataFrame, title: str):
    x = mdates.date2num(pd.to_datetime(daily.index))
    for xi, r in zip(x, daily.itertuples()):
        if np.isfinite(r.mean):
            _candle(ax, xi, r.min, r.q1, r.q3, r.max, r.mean, C_WIND)
    extra = []
    if daily["gust"].notna().any():
        from matplotlib.lines import Line2D
        extra = [Line2D([], [], marker="^", ls="", color=C_GUST, label="Peak gust")]
    _mean_legend(ax, C_WIND, "25th–75th percentile", "Min–max (sustained)", extra)
    if daily["gust"].notna().any():
        ax.scatter(x, daily["gust"], marker="^", color=C_GUST, s=26, zorder=5)
        for xi, g in zip(x, daily["gust"]):
            if np.isfinite(g):
                ax.text(xi, g + 1, f"{g:.0f}", ha="center", fontsize=5.5, color=C_GUST)
    ax.set_ylim(0, np.nanmax([daily["gust"].max(), daily["max"].max(), 10]) * 1.25)
    ax.set_ylabel("mph")
    ax.set_title(title, loc="left", fontsize=10, weight="bold")


def _wrapped(lo, hi):
    """Split the span [lo, hi] (degrees, may run past 0 or 360) into pieces inside 0-360."""
    out = []
    for shift in (-360, 0, 360):
        a, b = max(lo + shift, 0), min(hi + shift, 360)
        if b > a:
            out.append((a, b))
    return out


def direction_chart(ax, daily: pd.DataFrame, title: str, width=0.62):
    x = mdates.date2num(pd.to_datetime(daily.index))
    for xi, r in zip(x, daily.itertuples()):
        if not np.isfinite(r.mean):
            continue
        for a, b in _wrapped(r.lo, r.hi):
            ax.vlines(xi, a, b, color=C_DIR, lw=1.1, zorder=2)
        for a, b in _wrapped(r.q1, r.q3):
            ax.bar(xi, max(b - a, 2), bottom=a, width=width, color=C_DIR, alpha=0.55,
                   edgecolor=C_DIR, zorder=3)
        ax.hlines(r.mean, xi - width * 0.62, xi + width * 0.62, color=C_MEAN, lw=2.0, zorder=4)
    ax.set_ylim(0, 430)  # headroom above N for the legend
    ax.set_yticks([0, 90, 180, 270, 360], ["N", "E", "S", "W", "N"])
    ax.set_yticks([45, 135, 225, 315], minor=True)
    ax.grid(True, which="minor", axis="y", color=C_GRID, lw=0.4, ls=":")
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(False)
    ax.set_ylabel("wind from")
    ax.set_title(title, loc="left", fontsize=10, weight="bold")
    _mean_legend(ax, C_DIR, "25th–75th percentile", "10th–90th percentile")


# --------------------------------------------------------------------------- history: NCEI / HOMR
C_COOP_SNOW, C_STN_SNOW = "#2e86ab", "#e07a1f"  # co-op = blue, mesonet station = orange


def snow_qc(obs: pd.DataFrame) -> tuple[pd.Series, pd.Series, int, int]:
    """Station snow depth and 6-h snowfall with sensor spikes removed: depth more than 8 in
    from its 25-hour running median (catches glitches lasting several hours), and 6-h
    snowfall above 24 in. Returns (depth, snow6, depth readings removed, snowfall removed)."""
    depth = obs["depth"].astype(float).copy() if "depth" in obs else pd.Series(dtype=float)
    snow6 = obs["snow6"].astype(float).copy() if "snow6" in obs else pd.Series(dtype=float)
    med = depth.rolling(25, center=True, min_periods=6).median()
    bad_d = (depth - med).abs() > 8
    bad_s = snow6 > 24
    depth[bad_d] = np.nan
    snow6[bad_s] = np.nan
    return depth, snow6, int(bad_d.sum()), int(bad_s.sum())


def ncei_section(rep: Report, days: int, ghcnd: str, snow_station: tuple | None = None,
                 saved_only: bool = False):
    """NCEI co-op page. saved_only: take the station's snow data from data/ instead of
    scraping weather.gov (season report)."""
    # 1) look the co-op station up with the HOMR API (name, elevation, still active?)
    homr = get(HOMR_SEARCH, params={"qid": f"GHCND:{ghcnd}"}).json()
    stations = homr.get("stationCollection", {}).get("stations", [])
    if not stations:
        raise RuntimeError(f"HOMR has no station GHCND:{ghcnd}")
    st = stations[0]
    name = st.get("header", {}).get("preferredName", ghcnd)
    elev = next((e.get("elevationFeet") for e in st.get("location", {}).get("elevations", [])
                 if e.get("elevationFeet")), None)
    if st.get("header", {}).get("por", {}).get("endDate") != "Present":
        print(f"  note: HOMR lists {ghcnd} as no longer active", file=sys.stderr)
    name_elev = f"{name} ({int(float(elev)):,} ft)" if elev else name
    short = name.split(",")[0].title()

    # 2) daily co-op data from the NCEI Access Data Service
    end = dt.datetime.now(TZ).date()
    start = end - dt.timedelta(days=days)
    data = get(NCEI_DATA, params={
        "dataset": "daily-summaries", "stations": ghcnd, "startDate": start.isoformat(),
        "endDate": end.isoformat(), "dataTypes": "TMAX,TMIN,PRCP,SNOW,SNWD",
        "format": "json", "units": "standard"}).json()
    df = pd.DataFrame(data) if data else pd.DataFrame({"DATE": []})
    df.index = pd.to_datetime(df["DATE"])
    for c in ["TMAX", "TMIN", "PRCP", "SNOW", "SNWD"]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df = df.reindex(pd.date_range(start, end, freq="D"))
    daily = pd.DataFrame({"high": df.TMAX, "low": df.TMIN}, index=df.index)

    # 3) the nearby mesonet station's hourly snow depth and 6-hour snowfall (weather.gov)
    s_depth = s_snow6 = pd.Series(dtype=float)
    removed, s_label = (0, 0), ""
    if snow_station:
        s_name, s_elev, s_id = snow_station
        s_label = f"{s_name} ({s_elev:,} ft)"
        try:
            obs = load_saved_obs(s_id) if saved_only else scrape_wrh([s_id], days).get(s_id)
            if obs is not None and not obs.empty:
                obs = obs[obs.index >= pd.Timestamp(start).tz_localize(TZ)]
                s_depth, s_snow6, *removed = snow_qc(obs)
                s_depth.index = s_depth.index.tz_localize(None)
                s_snow6.index = s_snow6.index.tz_localize(None)
        except Exception as e:
            print(f"  snow data for {s_name} unavailable: {e}", file=sys.stderr)
    if not data and s_depth.dropna().empty:
        raise RuntimeError(f"NCEI returned no daily data for {ghcnd} since {start}")

    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=LAND, sharex=True,
                                     gridspec_kw=dict(height_ratios=[2.6, 1.8, 1.8], hspace=0.3,
                                                      left=0.07, right=0.96, top=0.86, bottom=0.08))
    header(fig, f"{name_elev}" + (f" + {s_label}" if s_label else "")
           + f" — last {days} days",
           f"Co-op {ghcnd}: daily, looked up with the HOMR API, data from the NCEI Access Data "
           f"Service (lags days to weeks)" + (f"  ·  {snow_station[2]}: hourly, weather.gov"
                                               if snow_station else ""))
    candlestick_temp(a1, daily, f"Daily temperature (high/low) — {short} co-op")

    # snowfall: co-op new snow per day (wide bars) + station snowfall per 6 h (narrow bars)
    a2.bar(df.index + pd.Timedelta(hours=12), df.SNOW, width=0.8, color=C_COOP_SNOW, alpha=0.55,
           label=f"{short} co-op: new snow per day")
    s6 = s_snow6.dropna()
    if len(s6):
        a2.bar(s6.index - pd.Timedelta(hours=3), s6, width=0.22, color=C_STN_SNOW,
               label=f"{snow_station[0]}: snowfall per 6 h")
    a2.set_ylabel("in")
    a2.set_title("Snowfall", loc="left", fontsize=10, weight="bold")
    a2.legend(loc="upper left", fontsize=7, frameon=False, ncol=2)
    top = np.nanmax([df.SNOW.max(skipna=True) if df.SNOW.notna().any() else 0,
                     s6.max() if len(s6) else 0, 1])
    a2.set_ylim(0, top * 1.35)

    # snow depth: station hourly line + co-op daily dots
    if s_depth.notna().any():
        a3.plot(s_depth.index, s_depth, color=C_STN_SNOW, lw=1.6,
                label=f"{snow_station[0]} (hourly)")
    a3.plot(df.index + pd.Timedelta(hours=12), df.SNWD, color=C_COOP_SNOW, lw=0, marker="o", ms=4,
            label=f"{short} co-op (daily)")
    a3.set_ylabel("in")
    a3.set_ylim(bottom=0)
    a3.set_title("Snow depth", loc="left", fontsize=10, weight="bold")
    a3.legend(loc="upper left", fontsize=7, frameon=False, ncol=2)
    date_axis(a3)
    a3.set_xlim(pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1))
    if not (df.SNOW.notna().any() or df.SNWD.notna().any() or len(s6) or s_depth.notna().any()):
        a2.text(0.5, 0.5, "No snowfall reported for this window", transform=a2.transAxes,
                ha="center", va="center", fontsize=9, color=C_MUTED)

    parts = []
    for label, cols in (("temperature", ["TMAX", "TMIN"]), ("snow", ["SNOW", "SNWD"])):
        last = df.dropna(subset=cols, how="all").index.max()
        parts.append(f"{label} {last:%b %d}" if pd.notna(last) else f"{label} none")
    note = "latest co-op day: " + ", ".join(parts)
    if snow_station and s_depth.notna().any():
        note += f"  ·  {snow_station[0]} depth now {s_depth.dropna().iloc[-1]:.0f} in"
    if sum(removed):
        note += f"  ·  sensor spikes removed: {removed[0]} depth, {removed[1]} snowfall"
    fig.text(0.96, 0.89, note, ha="right", fontsize=7.5, color=C_MUTED)
    rep.save(fig, f"History: NCEI {short} co-op station")


# --------------------------------------------------------------------------- history: Synoptic
def synoptic_timeseries(token: str, stids: list[str], days: int) -> dict[str, pd.DataFrame]:
    end = dt.datetime.now(dt.timezone.utc)
    start = (dt.datetime.now(TZ).replace(hour=0, minute=0, second=0, microsecond=0)
             - dt.timedelta(days=days)).astimezone(dt.timezone.utc)

    def fetch(start):
        return get(f"{SYNOPTIC_BASE}/stations/timeseries", params={
            "token": token, "stid": ",".join(stids), "start": f"{start:%Y%m%d%H%M}",
            "end": f"{end:%Y%m%d%H%M}", "vars": "air_temp,wind_speed,wind_gust,wind_direction",
            "units": "english,speed|mph", "obtimezone": "utc"}).json()

    js = fetch(start)
    if js.get("SUMMARY", {}).get("RESPONSE_CODE") == 403:
        # free accounts only get recent history; fall back to the last 7 days
        js = fetch(end - dt.timedelta(days=7) + dt.timedelta(minutes=10))
    if js.get("SUMMARY", {}).get("RESPONSE_CODE") != 1:
        raise RuntimeError(f"Synoptic timeseries error: {js.get('SUMMARY')}")
    frames = {}
    for st in js.get("STATION", []):
        obs = st.get("OBSERVATIONS", {})
        idx = pd.to_datetime(obs.get("date_time", []), utc=True).tz_convert(TZ)

        def pick(var):
            keys = sorted(k for k in obs if k.startswith(var + "_set_"))
            return pd.to_numeric(pd.Series(obs[keys[0]], index=idx), errors="coerce") if keys \
                else pd.Series(np.nan, index=idx)
        frames[st["STID"]] = pd.DataFrame({"temp": pick("air_temp"), "wind": pick("wind_speed"),
                                           "gust": pick("wind_gust"),
                                           "wdir": pick("wind_direction")}).sort_index()
    return frames


def load_saved_obs(stid: str) -> pd.DataFrame:
    """A station's saved observation history (data/obs_<STID>.csv), local time; empty if
    there's none. Used by the season report, which works from saved data only."""
    path = os.path.join(CACHE_DIR, f"obs_{stid}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(TZ)
    return df.sort_index()


def merge_cache(stid: str, fresh: pd.DataFrame) -> pd.DataFrame:
    """Every run saves the observations it gets to data/obs_<STID>.csv and merges them
    with earlier runs, so a source with less history than the report window (Synoptic's
    free plan, the NWS API) still builds up the full window over time. The `src`
    column records where each reading came from."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"obs_{stid}.csv")
    legacy = os.path.join(CACHE_DIR, f"synoptic_{stid}.csv")  # name used before Nov 2026
    if not os.path.exists(path) and os.path.exists(legacy):
        os.replace(legacy, path)
    frames = [fresh]
    if os.path.exists(path):
        old = pd.read_csv(path, index_col=0)
        old.index = pd.to_datetime(old.index, utc=True).tz_convert(TZ)
        frames.insert(0, old)
    allobs = pd.concat(frames)
    allobs = allobs[~allobs.index.duplicated(keep="last")].sort_index()
    allobs = allobs[allobs.index >= dt.datetime.now(TZ) - dt.timedelta(days=400)]
    out = allobs.copy()
    out.index = out.index.tz_convert("UTC")
    out.to_csv(path)
    return allobs


def daily_stats(df: pd.DataFrame, days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    day = df.index.tz_convert(TZ).normalize().tz_localize(None)
    # Drop partial days (e.g. the first day of the window starting late in the
    # evening): a day needs readings in at least 6 different hours. The NWS API
    # typically has 8-14 of 24 hours per day for these stations. Today is kept even
    # though it is partial.
    hours = pd.Series(df.index.tz_convert(TZ).hour, index=df.index).groupby(day).nunique()
    today_ts = pd.Timestamp(dt.datetime.now(TZ).date())
    keep = hours[(hours >= 6) | (hours.index == today_ts)].index
    df = df[day.isin(keep)]
    day = day[day.isin(keep)]
    g = df.groupby(day)
    t = pd.DataFrame({
        "high": g.temp.max(), "low": g.temp.min(), "mean": g.temp.mean(),
        "q1": g.temp.quantile(0.25), "q3": g.temp.quantile(0.75),
    })
    t["steep"] = steepest_warming(df["temp"]).reindex(t.index)  # box color
    w = pd.DataFrame({
        "mean": g.wind.mean(), "min": g.wind.min(), "max": g.wind.max(),
        "q1": g.wind.quantile(0.25), "q3": g.wind.quantile(0.75),
        "gust": g.gust.max(),  # NaN when the station sent no gust readings that day
    })
    if "wdir" not in df:
        df = df.assign(wdir=np.nan)
    groups = list(df.groupby(day))
    wd = pd.DataFrame([direction_stats(d) for _, d in groups], index=[k for k, _ in groups],
                      columns=["mean", "lo", "q1", "q3", "hi"])
    today = dt.datetime.now(TZ).date()
    rng = pd.date_range(today - dt.timedelta(days=days - 1), today, freq="D")
    return t.reindex(rng), w.reindex(rng), wd.reindex(rng)


def direction_stats(d: pd.DataFrame) -> list[float]:
    """Daily wind-direction summary that handles the wrap at north.

    mean: speed-weighted vector mean direction (0-360). The spread values are the
    10th/25th/75th/90th percentiles of each reading's offset from that mean, added
    back to the mean, so they can fall below 0 or above 360 (the plot wraps them).
    Calm readings (< 1 mph) are skipped because their direction is meaningless.
    """
    s = d.dropna(subset=["wdir"])
    if "wind" in s:
        s = s[s["wind"].fillna(0) >= 1]
    if len(s) < 3:
        return [np.nan] * 5
    rad = np.deg2rad(s["wdir"].to_numpy(float))
    wts = s["wind"].to_numpy(float) if "wind" in s else np.ones(len(s))
    mean = np.rad2deg(np.arctan2((wts * np.sin(rad)).sum(), (wts * np.cos(rad)).sum())) % 360
    rel = (s["wdir"].to_numpy(float) - mean + 180) % 360 - 180
    p10, p25, p75, p90 = np.percentile(rel, [10, 25, 75, 90])
    return [mean, mean + p10, mean + p25, mean + p75, mean + p90]


ROSE_BINS = [(1, 5), (5, 10), (10, 15), (15, 25), (25, np.inf)]
ROSE_COLORS = ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c"]


def wind_rose(ax, obs: pd.DataFrame, hours: int = 24):
    """16-sector wind rose of the last `hours` of readings on polar axes `ax`. Bars are
    stacked by speed band and show % of readings; calm readings (< 1 mph) have no
    direction, so they're counted separately."""
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]), ["N", "E", "S", "W"], fontsize=7)
    ax.tick_params(axis="x", pad=-2)
    ax.tick_params(axis="y", labelsize=5.5, colors=C_MUTED)
    ax.set_title(f"Wind rose, last {hours} h", fontsize=9, weight="bold", pad=4)

    recent = last_hours(obs, hours)
    recent = recent.dropna(subset=["wind", "wdir"]) if "wdir" in recent else recent.iloc[0:0]
    n = len(recent)
    if n == 0:
        ax.text(0, 0, "no data", ha="center", va="center", fontsize=8, color=C_MUTED)
        ax.set_yticks([])
        return
    calm = (recent["wind"] < 1).sum()
    windy = recent[recent["wind"] >= 1]
    sector = ((windy["wdir"] + 11.25) % 360 // 22.5).astype(int)
    theta = np.deg2rad(np.arange(16) * 22.5)
    bottom = np.zeros(16)
    for (lo, hi), color in zip(ROSE_BINS, ROSE_COLORS):
        s = (windy["wind"] >= lo) & (windy["wind"] < hi)
        pct = np.bincount(sector[s], minlength=16) / n * 100
        ax.bar(theta, pct, width=np.deg2rad(22.5) * 0.9, bottom=bottom, color=color,
               edgecolor="white", linewidth=0.4,
               label=f"{lo}–{hi:.0f}" if np.isfinite(hi) else f"{lo}+")
        bottom += pct
    top = max(bottom.max(), 1)
    ax.set_ylim(0, top * 1.08)
    ticks = np.linspace(0, top, 3)[1:]
    ax.set_yticks(ticks, [f"{v:.0f}%" for v in ticks])
    ax.set_rlabel_position(135)
    # speed legend in the lower-left corner of the rose, readings/calm in the lower right
    ax.legend(title="mph", loc="lower left", bbox_to_anchor=(-0.42, -0.2), fontsize=6,
              title_fontsize=6.5, frameon=False, handlelength=0.9, handleheight=0.7,
              labelspacing=0.2, borderaxespad=0)
    ax.text(1.32, -0.2, f"{n} readings\ncalm {calm / n:.0%}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6, color=C_MUTED)


def last_hours(obs: pd.DataFrame, hours: int = 24) -> pd.DataFrame:
    """Readings from the `hours` before the station's latest reading."""
    if obs.empty:
        return obs
    return obs[obs.index > obs.index.max() - pd.Timedelta(hours=hours)]


def recent_temp_chart(ax, obs: pd.DataFrame, hours: int = 24):
    r = last_hours(obs, hours).dropna(subset=["temp"])
    ax.set_title(f"Hourly temperature, last {hours} h", loc="left", fontsize=9, weight="bold")
    if r.empty:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", color=C_MUTED)
        return
    ax.plot(r.index, r.temp, color=C_WARM, lw=1.6, marker="o", ms=2.2)
    ax.axhline(32, color="#999999", lw=0.8, ls=":")
    lo, hi = r.temp.min(), r.temp.max()
    ax.set_ylim(lo - 3, hi + 4)
    ax.annotate(f"{r.temp.iloc[-1]:.0f}°", (r.index[-1], r.temp.iloc[-1]), xytext=(3, 3),
                textcoords="offset points", fontsize=7, color=C_WARM)
    ax.set_ylabel("°F")


def recent_wind_chart(ax, obs: pd.DataFrame, hours: int = 24):
    r = last_hours(obs, hours)
    ax.set_title(f"Hourly wind and gusts, last {hours} h", loc="left", fontsize=9, weight="bold")
    if r.empty or r.wind.isna().all():
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", color=C_MUTED)
        return
    ax.plot(r.index, r.wind, color=C_WIND, lw=1.6, label="Wind")
    if r.gust.notna().any():
        ax.scatter(r.index, r.gust, marker="^", s=12, color=C_GUST, label="Gust", zorder=3)
    ax.set_ylim(0, np.nanmax([r.wind.max(), r.gust.max(), 5]) * 1.3)
    ax.set_ylabel("mph")
    ax.legend(loc="upper left", fontsize=6.5, frameon=False, ncol=2)


def hour_axis(ax):
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18], tz=TZ))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1, tz=TZ))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=TZ))
    ax.tick_params(axis="x", labelsize=6.5)
    ax.tick_params(axis="y", labelsize=7)


def nws_observations(stid: str, days: int) -> pd.DataFrame:
    """Observations from the public NWS API (api.weather.gov), no key needed.

    It keeps roughly the last 7 days and, for these mesonet stations, usually has
    readings for about half the hours of each day.
    """
    start = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)
    url, params, rows = (f"{NWS_API}/stations/{stid}/observations",
                         {"start": start.strftime("%Y-%m-%dT%H:%M:%SZ")}, [])
    for _ in range(30):  # follow pagination
        js = get(url, params=params, headers=NWS_API_HEADERS).json()
        feats = js.get("features", [])
        rows += [f["properties"] for f in feats]
        nxt = js.get("pagination", {}).get("next")
        if not feats or not nxt:
            break
        url, params = nxt, None

    def val(p, key):
        q = p.get(key) or {}
        v, unit = q.get("value"), q.get("unitCode", "")
        if v is None:
            return np.nan
        if unit.endswith("degC"):
            return v * 9 / 5 + 32
        if unit.endswith("km_h-1"):
            return v / 1.609344
        if unit.endswith("m_s-1"):
            return v * 2.236936
        return v

    idx = pd.to_datetime([p["timestamp"] for p in rows], utc=True).tz_convert(TZ)
    df = pd.DataFrame({"temp": [val(p, "temperature") for p in rows],
                       "wind": [val(p, "windSpeed") for p in rows],
                       "gust": [val(p, "windGust") for p in rows],
                       "wdir": [val(p, "windDirection") for p in rows]}, index=idx)
    return df[~df.index.duplicated()].sort_index()


COMPASS = {d: i * 22.5 for i, d in enumerate(
    "N NNE NE ENE E ESE SE SSE S SSW SW WSW W WNW NW NNW".split())}
_WRH_CACHE: dict = {}

# Runs inside the page: returns the observation table as header + rows of text.
_TABLE_JS = """() => {
  const t = document.querySelector('#OBS_DATA');
  if (!t) return null;
  return Array.from(t.querySelectorAll('tr')).map(
    r => Array.from(r.querySelectorAll('th,td')).map(c => c.innerText.trim()));
}"""


def scrape_wrh(stids: list[str], days: int) -> dict[str, pd.DataFrame]:
    """Read the hourly observation table that weather.gov's station timeseries page
    (https://www.weather.gov/wrh/timeseries?site=<ID>) draws in the browser. The table
    is built by JavaScript, so this drives headless Chromium through Playwright.
    One visit per station per run: results are cached per station, so the NCEI page,
    the station pages and both PDF passes share them."""
    todo = [s for s in stids if (s, days) not in _WRH_CACHE]
    if not todo:
        return {s: _WRH_CACHE[(s, days)] for s in stids if _WRH_CACHE[(s, days)] is not None}
    # Chromium lives in ~/.playwright-browsers rather than Playwright's default
    # %LOCALAPPDATA%\ms-playwright, because new AppData folders created from inside
    # packaged Windows apps get virtualized and are invisible to Task Scheduler.
    home_browsers = os.path.join(os.path.expanduser("~"), ".playwright-browsers")
    if os.path.isdir(home_browsers):
        os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", home_browsers)
    from playwright.sync_api import sync_playwright

    # The page accepts at most 720 hours; asking for more silently returns only 72.
    hours = min(days * 24 + 24, 720)
    now = dt.datetime.now(TZ)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(user_agent=HEADERS["User-Agent"])
        for stid in todo:
            _WRH_CACHE[(stid, days)] = None
            for attempt in (1, 2):  # one retry after a short pause
                try:
                    page.goto(f"{WRH_TIMESERIES}?site={stid}&hourly=true&hours={hours}",
                              wait_until="networkidle", timeout=90_000)
                    page.wait_for_selector("#OBS_DATA tr:nth-child(3)", timeout=60_000)
                    rows = page.evaluate(_TABLE_JS)
                    _WRH_CACHE[(stid, days)] = _parse_wrh_table(rows, now)
                    break
                except Exception as e:
                    print(f"  weather.gov scrape failed for {stid} (attempt {attempt}): {e}",
                          file=sys.stderr)
                    if attempt == 1:
                        time.sleep(15)
        browser.close()
    return {s: _WRH_CACHE[(s, days)] for s in stids if _WRH_CACHE.get((s, days)) is not None}


def _parse_wrh_table(rows: list[list[str]], now: dt.datetime) -> pd.DataFrame:
    hdr = [re.sub(r"\s+", " ", h) for h in rows[0]]

    def col(prefix):
        return next((i for i, h in enumerate(hdr) if h.startswith(prefix)), None)

    i_t, i_temp, i_dir, i_spd, i_gst = (col("Date/Time"), col("Temp."), col("Wind Direction"),
                                        col("Wind Speed"), col("Wind Gust"))
    i_depth, i_snow6 = col("Snow Depth"), col("Snowfall 6 Hour")  # only some stations
    if i_t is None or i_temp is None:
        raise RuntimeError(f"unexpected table columns: {hdr}")

    def num(s):
        try:
            return float(s)
        except (TypeError, ValueError):
            return np.nan

    recs = []
    for r in rows[1:]:
        if len(r) <= max(i for i in (i_t, i_temp, i_dir, i_spd) if i is not None):
            continue
        try:  # "Nov 21, 8:00 pm" (no year)
            ts = dt.datetime.strptime(f"{r[i_t]} {now.year}", "%b %d, %I:%M %p %Y")
        except ValueError:
            continue
        if ts > now.replace(tzinfo=None) + dt.timedelta(days=1):  # Dec rows seen in Jan
            ts = ts.replace(year=now.year - 1)
        spd, gst = np.nan, np.nan
        if i_spd is not None:
            m = re.match(r"\s*(\d+(?:\.\d+)?)\s*(?:G\s*(\d+(?:\.\d+)?))?", r[i_spd])
            if m:
                spd = float(m.group(1))
                gst = float(m.group(2)) if m.group(2) else np.nan
        if i_gst is not None and np.isnan(gst):
            gst = num(r[i_gst])
        wdir = COMPASS.get(r[i_dir].strip().upper(), np.nan) if i_dir is not None else np.nan
        if r[i_dir].strip().upper() == "VAR" if i_dir is not None else False:
            wdir = np.nan
        depth = num(r[i_depth]) if i_depth is not None and i_depth < len(r) else np.nan
        snow6 = num(r[i_snow6]) if i_snow6 is not None and i_snow6 < len(r) else np.nan
        recs.append((ts, num(r[i_temp]), spd, gst, wdir, depth, snow6))
    df = pd.DataFrame(recs, columns=["time", "temp", "wind", "gust", "wdir", "depth",
                                     "snow6"]).set_index("time")
    if df[["depth", "snow6"]].isna().all().all():
        df = df.drop(columns=["depth", "snow6"])  # station doesn't measure snow
    df.index = df.index.tz_localize(TZ, ambiguous="NaT", nonexistent="shift_forward")
    df = df[df.index.notna()]
    return df[~df.index.duplicated()].sort_index()


def fetch_station_obs(stations_cfg: list[tuple], token: str | None, days: int,
                      overrides: dict[str, str]):
    """Fresh observations for every station, trying sources in order:
      1. the weather.gov timeseries page (full hourly history for the whole window),
      2. Synoptic, when a token is set and works (last 7 days on the free plan),
      3. the public NWS API (about 7 days, roughly half the hours).
    Returns (stations, {stid: df}, {stid: source label})."""
    stations = [{"name": n, "elev": el, "stid": overrides.get(n, sid)}
                for n, el, sid in stations_cfg]
    ids = [s["stid"] for s in stations]
    series, sources = {}, {}

    try:
        for stid, df in scrape_wrh(ids, days).items():
            if not df.empty:
                series[stid], sources[stid] = df, "weather.gov"
    except Exception as e:  # e.g. Playwright not installed
        print(f"  weather.gov scraping unavailable ({e})", file=sys.stderr)

    missing = [i for i in ids if i not in series]
    if missing and token:
        try:
            got = synoptic_timeseries(token, missing, days)
            for stid, df in got.items():
                if not df.empty:
                    series[stid], sources[stid] = df, "Synoptic"
        except Exception as e:
            print(f"  Synoptic unavailable ({e})", file=sys.stderr)

    for stid in [i for i in ids if i not in series]:
        try:
            series[stid], sources[stid] = nws_observations(stid, days), "NWS API"
        except Exception as e:
            print(f"  NWS API failed for {stid}: {e}", file=sys.stderr)
    return stations, series, sources


# --------------------------------------------------------------------------- GOES cloud cover
GOES_BUCKET = "https://noaa-goes18.s3.amazonaws.com/"  # NOAA open data on AWS, no key
GOES_PRODUCT = "ABI-L2-ACMC"   # GOES-18 Clear Sky Mask, CONUS sector, every 5 minutes
GOES_DAYS = 7                  # cloud cover shown for the last 7 days
GOES_CACHE = os.path.join(CACHE_DIR, "goes_cloud.csv")
C_CLOUD = "#7d7d7d"
_GOES: dict = {}               # this run's result, shared by every report


def _station_latlon(stid: str) -> tuple[float, float]:
    """Station coordinates from the NWS API (api.weather.gov/stations/<id>)."""
    js = get(f"{NWS_API}/stations/{stid}", headers=NWS_API_HEADERS).json()
    lon, lat = js["geometry"]["coordinates"][:2]
    return lat, lon


def _goes_xy(lat: float, lon: float, proj: dict) -> tuple[float, float]:
    """Geodetic lat/lon -> GOES fixed-grid scan angles (radians), per the GOES-R PUG."""
    r_eq, r_pol = proj["semi_major_axis"], proj["semi_minor_axis"]
    H = proj["perspective_point_height"] + r_eq
    lon0 = np.radians(proj["longitude_of_projection_origin"])
    lat, lon = np.radians(lat), np.radians(lon)
    e2 = (r_eq ** 2 - r_pol ** 2) / r_eq ** 2
    phi_c = np.arctan(r_pol ** 2 / r_eq ** 2 * np.tan(lat))
    rc = r_pol / np.sqrt(1 - e2 * np.cos(phi_c) ** 2)
    sx = H - rc * np.cos(phi_c) * np.cos(lon - lon0)
    sy = -rc * np.cos(phi_c) * np.sin(lon - lon0)
    sz = rc * np.sin(phi_c)
    return float(np.arcsin(-sy / np.sqrt(sx ** 2 + sy ** 2 + sz ** 2))), float(np.arctan(sz / sx))


def _goes_hour(hour: dt.datetime, points: dict[str, tuple[float, float]]) -> dict | None:
    """Cloud cover (%) at each point from the first Clear Sky Mask file of `hour` (UTC):
    the share of good-quality pixels flagged cloudy/probably cloudy in the 3x3 block
    (~6 km) around the point. None if the hour has no file yet or downloads keep failing."""
    import h5py

    prefix = f"{GOES_PRODUCT}/{hour:%Y}/{hour.timetuple().tm_yday:03d}/{hour:%H}/"
    for attempt in range(3):
        try:
            listing = requests.get(GOES_BUCKET, params={"list-type": "2", "prefix": prefix,
                                                        "max-keys": "1"}, timeout=60)
            keys = re.findall(r"<Key>([^<]+)</Key>", listing.text)
            if not keys:
                return None
            raw = requests.get(GOES_BUCKET + keys[0], timeout=120)
            raw.raise_for_status()
            with h5py.File(io.BytesIO(raw.content), "r") as f:
                p = f["goes_imager_projection"].attrs
                proj = {k: float(np.asarray(p[k]).ravel()[0]) for k in
                        ("semi_major_axis", "semi_minor_axis", "perspective_point_height",
                         "longitude_of_projection_origin")}
                xs, ys = f["x"], f["y"]
                xo, xsf = float(xs.attrs["add_offset"][0]), float(xs.attrs["scale_factor"][0])
                yo, ysf = float(ys.attrs["add_offset"][0]), float(ys.attrs["scale_factor"][0])
                out = {}
                for sid, (lat, lon) in points.items():
                    x, y = _goes_xy(lat, lon, proj)
                    i, j = round((y - yo) / ysf), round((x - xo) / xsf)
                    b = f["BCM"][i - 1:i + 2, j - 1:j + 2].astype(float)
                    good = (f["DQF"][i - 1:i + 2, j - 1:j + 2] == 0) & (b != 255)
                    out[sid] = 100 * b[good].mean() if good.any() else np.nan
                return out
        except Exception as e:
            last = e
            time.sleep(5 * (attempt + 1))
    print(f"  GOES {hour:%Y-%m-%d %H}Z failed: {last}", file=sys.stderr)
    return None


def goes_cloud_cover(fetch: bool = True) -> pd.DataFrame:
    """Hourly GOES-18 cloud cover (%) for every station in every report, last GOES_DAYS
    days, indexed in local time. Saved hours are kept in data/goes_cloud.csv, so each run
    only downloads the hours it doesn't have yet (one satellite file per hour).
    fetch=False: saved hours only, no downloads (season report)."""
    if "df" in _GOES:
        return _GOES["df"]
    stids = sorted({sid for site in SITES.values() for _, _, sid in site.stations})
    cache = pd.DataFrame()
    if os.path.exists(GOES_CACHE):
        cache = pd.read_csv(GOES_CACHE, index_col=0)
        cache.index = pd.to_datetime(cache.index, utc=True)
    now = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
    hours = [now - dt.timedelta(hours=h) for h in range(1, GOES_DAYS * 24 + 1)]
    new_station = any(s not in cache.columns for s in stids)
    todo = [h for h in hours if new_station or pd.Timestamp(h) not in cache.index]
    if todo and fetch:
        points = {}
        for sid in stids:
            try:
                points[sid] = _station_latlon(sid)
            except Exception as e:
                print(f"  no coordinates for {sid}: {e}", file=sys.stderr)
        t0 = time.time()
        import concurrent.futures as cf
        rows = {}
        with cf.ThreadPoolExecutor(8) as ex:
            for hour, vals in zip(todo, ex.map(lambda h: _goes_hour(h, points), todo)):
                if vals:
                    rows[pd.Timestamp(hour)] = vals
        print(f"  GOES: {len(rows)} of {len(todo)} new hours in {time.time() - t0:.0f}s",
              file=sys.stderr)
        if rows:
            fresh = pd.DataFrame.from_dict(rows, orient="index")
            cache = fresh.combine_first(cache) if not cache.empty else fresh
        cutoff = pd.Timestamp(now - dt.timedelta(days=GOES_DAYS + 3))
        cache = cache[cache.index >= cutoff].sort_index()
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache.to_csv(GOES_CACHE)
    df = cache.copy()
    if not df.empty:
        df.index = df.index.tz_convert(TZ)
        df = df[df.index >= pd.Timestamp(now - dt.timedelta(days=GOES_DAYS)).tz_convert(TZ)]
    _GOES["df"] = df
    return df


def _cloud_axis(ax):
    """Right-hand 0-100% cloud axis drawn underneath `ax`'s own data."""
    ca = ax.twinx()
    ca.set_zorder(ax.get_zorder() - 1)
    ax.patch.set_visible(False)
    ca.set_ylim(0, 115)
    ca.set_yticks([0, 50, 100], ["0%", "50%", "100%"])
    ca.tick_params(axis="y", labelsize=6.5, colors=C_CLOUD, length=2)
    ca.spines["right"].set_visible(True)
    ca.spines["right"].set_color(C_CLOUD)
    ca.grid(False)
    return ca


def cloud_bars(ax, cloud: pd.Series):
    """Daily mean cloud cover as faint gray bars behind the 30-day wind chart (days with
    at least 12 hourly values)."""
    c = cloud.dropna()
    if c.empty:
        return
    day = c.index.normalize().tz_localize(None)
    n = c.groupby(day).size()
    daily = c.groupby(day).mean()[n >= 12]
    ca = _cloud_axis(ax)
    ca.bar(mdates.date2num(daily.index), daily, width=0.8, color=C_CLOUD, alpha=0.2, lw=0)
    for x, v in zip(mdates.date2num(daily.index), daily):
        ca.text(x, v + 2, f"{v:.0f}%", ha="center", fontsize=5.5, color=C_CLOUD)
    ca.set_ylabel("cloud cover", fontsize=7, color=C_CLOUD)


def cloud_line(ax, cloud: pd.Series, obs: pd.DataFrame, hours: int = 24):
    """Hourly cloud cover as a faint gray line behind the 24-hour wind chart."""
    recent = last_hours(obs, hours)
    if recent.empty:
        return
    c = cloud.dropna()
    c = c[(c.index >= recent.index.min()) & (c.index <= recent.index.max() + pd.Timedelta(hours=1))]
    if c.empty:
        return
    ca = _cloud_axis(ax)
    ca.plot(c.index, c, color=C_CLOUD, lw=1.6, alpha=0.3)
    ca.fill_between(c.index, c, color=C_CLOUD, alpha=0.08, lw=0)


def station_section(rep: Report, site: Site, token: str | None, days: int,
                    overrides: dict[str, str], saved_only: bool = False,
                    period: str | None = None, recent_panels: bool = True):
    """Station history pages. saved_only: use only the saved data in data/ (no scraping or
    downloads; season report). period: page-title wording, default "last N days".
    recent_panels: include the last-24-hour column (left out of the season report)."""
    section = f"History: {site.history_label}"
    period = period or f"last {days} days"
    if saved_only:
        stations = [{"name": n, "elev": el, "stid": overrides.get(n, sid)}
                    for n, el, sid in site.stations]
        series = {s["stid"]: load_saved_obs(s["stid"]) for s in stations}
        series = {k: v for k, v in series.items() if not v.empty}
        sources = {k: "saved data" for k in series}
        if not series:
            raise RuntimeError(f"No saved station data in {CACHE_DIR}")
    else:
        stations, series, sources = fetch_station_obs(site.stations, token, days, overrides)
        if not series:
            raise RuntimeError("No station data from weather.gov, Synoptic or the NWS API")
        for stid, df in series.items():
            df["src"] = sources[stid]
        series = {stid: merge_cache(stid, df) for stid, df in series.items()}
    window_start = pd.Timestamp(dt.datetime.now(TZ).date() - dt.timedelta(days=days - 1)).tz_localize(TZ)
    try:
        clouds = goes_cloud_cover(fetch=not saved_only)
    except Exception as e:  # cloud cover is extra; never fail the station pages over it
        print(f"  GOES cloud cover unavailable: {e}", file=sys.stderr)
        clouds = pd.DataFrame()

    first = True
    for s in stations:
        title = f"{s['name']}  ({s['elev']:,} ft)"
        if not s["stid"] or s["stid"] not in series or series[s["stid"]].empty:
            error_page(rep, f"History: {title}",
                       f"No data returned for this station (ID: {s['stid'] or 'not found'}).",
                       section=section if first else None)
            first = False
            continue
        allobs = series[s["stid"]]
        obs = allobs[allobs.index >= window_start]
        srcs = (", ".join(sorted(obs["src"].dropna().unique())) if "src" in obs
                else sources.get(s["stid"], "?"))
        t, w, wd = daily_stats(obs, days)
        cloud = (clouds[s["stid"]] if s["stid"] in clouds and clouds[s["stid"]].notna().any()
                 else None)

        # left column: 30-day daily charts (shared date axis)
        # right column: last-24-hour temperature, wind/gust, and the wind rose
        # (left out of the season report, which shows only the season-long charts)
        fig = plt.figure(figsize=LAND)
        if recent_panels:
            gs = fig.add_gridspec(3, 2, width_ratios=[3.0, 1.0], height_ratios=[3, 2.5, 2.2],
                                  hspace=0.4, wspace=0.24, left=0.06, right=0.955, top=0.86,
                                  bottom=0.07)
        else:
            gs = fig.add_gridspec(3, 1, height_ratios=[3, 2.5, 2.2], hspace=0.4, left=0.06,
                                  right=0.93, top=0.86, bottom=0.07)
        a1 = fig.add_subplot(gs[0, 0])
        a2 = fig.add_subplot(gs[1, 0], sharex=a1)
        a3 = fig.add_subplot(gs[2, 0], sharex=a1)
        if recent_panels:
            b1 = fig.add_subplot(gs[0, 1])
            b2 = fig.add_subplot(gs[1, 1], sharex=b1)
            b3 = fig.add_subplot(gs[2, 1], projection="polar")

        have = t.high.notna().sum()
        header(fig, f"{title} — {period}",
               f"Station {s['stid']}  ·  data: {srcs}  ·  {len(obs):,} observations  ·  "
               f"{have} of {days} days with data  ·  daily = local calendar day ({TZ.key})"
               + ("  ·  gray: GOES-18 cloud cover, last 7 days" if cloud is not None else ""))
        if have < days and not saved_only:
            fig.text(0.975, 0.89, "Older days come from saved earlier runs (data/ folder) when "
                                  "a source has less history", ha="right", fontsize=7.5, color=C_WARM)
        candlestick_temp(a1, t, "Daily temperature — box: 25th–75th pct, whiskers: low/high, "
                                "color: steepest 2-h warming")
        wind_chart(a2, w, "Daily wind speed" + ("  +  daily mean cloud cover (gray bars, right axis)"
                                                if cloud is not None else ""))
        if cloud is not None:
            cloud_bars(a2, cloud)
        direction_chart(a3, wd, "Daily wind direction")
        date_axis(a3)
        plt.setp(a1.get_xticklabels(), visible=False)
        plt.setp(a2.get_xticklabels(), visible=False)
        a3.set_xlim(mdates.date2num(t.index[0]) - 0.7, mdates.date2num(t.index[-1]) + 0.7)

        if recent_panels:
            recent_temp_chart(b1, allobs)
            recent_wind_chart(b2, allobs)
            if cloud is not None:
                b2.set_title("Wind, gusts + cloud, last 24 h", loc="left", fontsize=9, weight="bold")
                cloud_line(b2, cloud, allobs)
            hour_axis(b2)
            b1.tick_params(axis="y", labelsize=7)
            plt.setp(b1.get_xticklabels(), visible=False)
            wind_rose(b3, allobs)

        rep.save(fig, section if first else None)
        first = False
    used = sorted(set(sources.values()))
    return "ok" if saved_only or used == ["weather.gov"] else f"ok ({', '.join(used)})"


# --------------------------------------------------------------------------- title + contents
def title_page(rep: Report, site: Site, days: int, statuses: list[tuple[str, str]],
               subtitle: str = "Comprehensive weather report", window: str | None = None,
               sources: list[str] | None = None):
    fig = plt.figure(figsize=PAGE)
    fig.text(0.06, 0.90, site.title, fontsize=24, weight="bold", color=C_TEXT)
    fig.text(0.06, 0.862, subtitle, fontsize=15, color=C_MUTED)
    fig.text(0.06, 0.832, f"Generated {rep.generated:%A, %B %d %Y at %H:%M %Z}  ·  "
                          + (window or f"history window: last {days} days"), fontsize=9,
             color=C_MUTED)
    y = 0.78
    fig.text(0.06, y, "Contents", fontsize=12, weight="bold"); y -= 0.03
    for name, page in rep.toc:
        fig.text(0.08, y, name, fontsize=10)
        fig.text(0.80, y, f"p. {page + 1}", fontsize=10, ha="right", color=C_MUTED)
        y -= 0.024
    y -= 0.03
    fig.text(0.06, y, "Source status", fontsize=12, weight="bold"); y -= 0.03
    for name, st in statuses:
        fig.text(0.08, y, name, fontsize=9)
        fig.text(0.46, y, st, fontsize=9, color=C_COOL if st == "ok" else C_WARM)
        y -= 0.022
    y -= 0.03
    ids = "|".join(sid for _, _, sid in site.stations)
    srcs = sources or ([UAC_URL, f"{UAC_EXPLORER} (avalanches, {AVY_REGION})", AFD_URL]
            + [p.url for p in site.points]
            + [f"{PLUME_PAGE}?t={m}&d=PL&r={site.plume_station}" for m in PLUME_MODELS]
            + (["https://www.ncei.noaa.gov/access/homr/api"] if site.ncei_ghcnd else [])
            + [f"{WRH_TIMESERIES}?site={ids}&hourly=true&hours=720",
               "https://api.synopticdata.com/v2 (fallback)",
               "https://api.weather.gov (station observations, fallback)"])
    fig.text(0.06, y, "Sources", fontsize=12, weight="bold"); y -= 0.025
    for s in srcs:
        fig.text(0.08, y, s[:115], fontsize=6.8, color=C_MUTED, family="monospace"); y -= 0.018
    return fig


# --------------------------------------------------------------------------- main
def write_report(site: Site, sections: list, out: str, now: dt.datetime, doc_title: str,
                 make_title) -> list[tuple[str, str]]:
    """Run each (name, fn) section into a PDF with a title page listing the contents.
    matplotlib can't insert pages, so pass 1 collects the contents and pass 2 writes the
    final file; downloads are cached, so nothing is fetched twice. A failing section
    becomes an error page; the rest still build. Returns [(section, status)]."""
    def build(path, statuses_in=None, toc_in=None):
        statuses = []
        with PdfPages(path) as pdf:
            rep = Report(pdf, generated=now, label=doc_title)
            if toc_in is not None:
                rep.toc = toc_in
                rep.save(make_title(rep, statuses_in))
                rep.toc = []
            for name, fn in sections:
                print(f"- [{site.key}] {name} ...", flush=True)
                try:
                    statuses.append((name, fn(rep) or "ok"))
                except Exception as e:
                    traceback.print_exc()
                    error_page(rep, name, f"{type(e).__name__}: {e}", section=name)
                    statuses.append((name, f"failed: {type(e).__name__}"))
            d = pdf.infodict()
            d["Title"] = doc_title
            d["CreationDate"] = now
        return rep.toc, statuses

    tmp = out + ".tmp.pdf"
    toc, statuses = build(tmp)
    os.remove(tmp)
    build(out, statuses_in=statuses, toc_in=toc)
    print(f"Wrote {out}\n")
    return statuses


def season_start_date(today: dt.date, mmdd: str = "10-23") -> dt.date:
    """Start of the current season: the most recent MM-DD on or before today."""
    m, d = map(int, mmdd.split("-"))
    start = dt.date(today.year, m, d)
    return start if start <= today else dt.date(today.year - 1, m, d)


def build_season_report(site: Site, start: dt.date, now: dt.datetime, outdir: str,
                        overrides: dict[str, str] | None = None) -> str:
    """Season-to-date history report for one area: the NCEI co-op page, every station
    page and the season's avalanches, from `start` to today. Station data comes only from
    the saved files in data/ (no scraping), so it can run locally any time; the NCEI page
    and the avalanche page fetch their season data from NCEI and the UAC. No forecasts.
    Writes reports/<area>_season_report_<start>_<today>.pdf and returns its path.

    Run it with:  python weather_report.py --season          (all three areas)
                  python weather_report.py --season --site pc
    """
    today = now.date()
    days = (today - start).days + 1  # calendar days in the season so far, today included
    period = f"season {start:%b %d} – {today:%b %d %Y}"
    out = os.path.join(outdir, f"{site.key}_season_report_{start:%Y%m%d}_{today:%Y%m%d}.pdf")
    sections = []
    if site.ncei_ghcnd:
        sections.append(("History: NCEI / HOMR",
                         lambda r: ncei_section(r, days - 1, site.ncei_ghcnd, site.snow_station,
                                                saved_only=True)))
    sections.append((f"History: {site.history_label}",
                     lambda r: station_section(r, site, None, days, overrides or {},
                                               saved_only=True, period=period,
                                               recent_panels=False)))
    sections.append(("Avalanches this season (UAC)",
                     lambda r: avalanche_activity_section(
                         r, days - 1,
                         title=f"Avalanches This Season ({start:%b %d} – {today:%b %d})",
                         section="Avalanches this season (UAC)")))
    ids = "|".join(sid for _, _, sid in site.stations)
    srcs = ([f"data/obs_<ID>.csv (saved weather.gov station history: {ids})",
             "data/goes_cloud.csv (saved GOES-18 cloud cover, last 7 days)"]
            + (["https://www.ncei.noaa.gov/access/homr/api", NCEI_DATA] if site.ncei_ghcnd else [])
            + [f"{UAC_EXPLORER} (avalanches, {AVY_REGION})"])
    write_report(site, sections, out, now, f"{site.title} season report",
                 lambda rep, st: title_page(rep, site, days, st, subtitle="Season history report",
                                            window=period, sources=srcs))
    return out


def build_report(site: Site, args, overrides: dict[str, str], now: dt.datetime,
                 outdir: str) -> tuple[str, list[tuple[str, str]]]:
    out = os.path.join(outdir, f"{site.file_prefix}_{now:%Y%m%d_%H%M}.pdf")
    sections = [
        ("Avalanche forecast (UAC)", lambda r: avalanche_section(r)),
        ("NWS forecast discussion", lambda r: afd_section(r)),
    ]
    for pt in site.points:
        sections.append((f"NWS forecast: {pt.label}",
                         lambda r, pt=pt: point_forecast_section(
                             r, pt, f"NWS forecast: {pt.label} ({pt.elev:,} ft)")))
    sections.append(("Ensemble plumes (U of U)",
                     lambda r: plume_section(r, site.plume_station, site.plume_name)))
    sections.append(("RRFS hourly snow plumes (U of U)",
                     lambda r: plume_section(r, site.plume_station, site.plume_name, "rrfsqsf")))
    if site.ncei_ghcnd:
        sections.append(("History: NCEI / HOMR",
                         lambda r: ncei_section(r, args.days, site.ncei_ghcnd, site.snow_station)))
    sections.append((f"History: {site.history_label}",
                     lambda r: station_section(r, site, args.synoptic_token, args.days, overrides)))
    sections.append(("Avalanches past 7 days (UAC)", lambda r: avalanche_activity_section(r)))

    statuses = write_report(site, sections, out, now, f"{site.title} weather report",
                            lambda rep, st: title_page(rep, site, args.days, st))
    return out, statuses


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", choices=["all", *SITES], default="all",
                    help="which report to build (default: all)")
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports"),
                    help="folder for the PDFs (default: reports/ next to this script)")
    ap.add_argument("--publish-dir", default=PUBLISH_DIR,
                    help="after a full run, copy the PDFs here and delete every other file "
                         "in it (default: the Ski Patrol Google Drive folder; \"\" = off)")
    ap.add_argument("--site-dir", default="",
                    help="also write a small web page here: <key>.pdf (alta.pdf, pc.pdf) at fixed "
                         "names plus index.html linking them; used for GitHub Pages")
    ap.add_argument("--email-to", default=os.environ.get("EMAIL_TO"),
                    help="email the PDFs to these comma-separated addresses (or env EMAIL_TO); "
                         "needs SMTP_USER and SMTP_PASSWORD, see email_reports()")
    ap.add_argument("--email-subject-suffix", default="",
                    help='added to the email subject, e.g. "(updated)" for a catch-up run')
    ap.add_argument("--days", type=int, default=30, help="history window in days (default 30)")
    ap.add_argument("--synoptic-token", default=os.environ.get("SYNOPTIC_TOKEN"),
                    help="Synoptic Data API token (or env SYNOPTIC_TOKEN); fallback source only")
    ap.add_argument("--season", action="store_true",
                    help="instead of the daily reports, build season-to-date history reports "
                         "(NCEI, station history, avalanches) from the saved station data")
    ap.add_argument("--season-start", default="10-23",
                    help="season start as MM-DD (default 10-23); the most recent one is used")
    ap.add_argument("--stid", action="append", default=[],
                    help="override a station ID, e.g. --stid 'ALTA - BASE=ATB' (repeatable)")
    args = ap.parse_args(argv)

    now = dt.datetime.now(TZ)
    os.makedirs(args.outdir, exist_ok=True)
    overrides = dict(s.split("=", 1) for s in args.stid)
    sites = list(SITES.values()) if args.site == "all" else [SITES[args.site]]

    if args.season:
        start = season_start_date(now.date(), args.season_start)
        return [build_season_report(site, start, now, args.outdir, overrides) for site in sites]

    # All reports and both passes share downloads through get()'s cache.
    results = [build_report(site, args, overrides, now, args.outdir) for site in sites]
    outputs = [out for out, _ in results]
    statuses = {site.key: st for site, (_, st) in zip(sites, results)}

    if args.publish_dir and args.site == "all":
        publish(outputs, args.publish_dir)
    elif args.publish_dir:
        print(f"Not publishing: --site {args.site} builds only one report, and publishing "
              f"would delete the other one from {args.publish_dir}")
    complete = all(st.startswith("ok") for sts in statuses.values() for _, st in sts)
    failed = [f"{k}: {name}" for k, sts in statuses.items() for name, st in sts
              if not st.startswith("ok")]
    print("All sections ok" if complete else "Sections that failed: " + "; ".join(failed))
    if args.site_dir:
        build_site(outputs, args.site_dir, now)
        with open(os.path.join(args.site_dir, "status.json"), "w", encoding="utf-8") as f:
            json.dump({"generated": now.isoformat(), "complete": complete, "failed": failed,
                       "sections": {k: dict(sts) for k, sts in statuses.items()}}, f, indent=1)
    if args.email_to:
        email_reports(outputs, args.email_to, now, args.email_subject_suffix)
    return outputs


def build_site(pdfs: list[str], site_dir: str, now: dt.datetime):
    """Copy each report to a fixed name (alta.pdf, pc.pdf) and write index.html linking
    them, so a bookmarked link always gets the newest report."""
    import html as htmlmod

    os.makedirs(site_dir, exist_ok=True)
    rows = []
    for p in pdfs:
        if not os.path.isfile(p):
            continue
        site = next((s for s in SITES.values() if os.path.basename(p).startswith(s.file_prefix)), None)
        name = f"{site.key}.pdf" if site else os.path.basename(p)
        shutil.copy2(p, os.path.join(site_dir, name))
        title = site.title if site else name
        kb = os.path.getsize(p) // 1024
        # ?v= makes browsers fetch the new file instead of a cached copy
        rows.append(f'<a class="card" href="{name}?v={now:%Y%m%d%H%M}">'
                    f'<span class="t">{htmlmod.escape(title)}</span>'
                    f'<span class="s">{name} · {kb:,} KB</span></a>')
    stamp = f"{now:%A, %B %d %Y at %H:%M %Z}".replace(" 0", " ")
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Utah Avy Weather Reports</title>
<style>
  :root {{ --bg:#f6f7f9; --card:#ffffff; --text:#1d2430; --muted:#5d6675; --line:#dde2e8; --accent:#2e86ab; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg:#14181e; --card:#1d232b; --text:#e8ecf1; --muted:#9aa4b1; --line:#2c343e; --accent:#5fb3d6; }}
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--text);
         font:16px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }}
  main {{ max-width:560px; margin:0 auto; padding:40px 16px; }}
  h1 {{ font-size:1.5rem; margin:0 0 4px; }}
  .when {{ color:var(--muted); margin:0 0 24px; font-size:.95rem; }}
  .card {{ display:flex; flex-direction:column; gap:2px; padding:16px 18px; margin:0 0 12px;
          background:var(--card); border:1px solid var(--line); border-left:4px solid var(--accent);
          border-radius:8px; color:inherit; text-decoration:none; }}
  .card:hover {{ border-color:var(--accent); }}
  .t {{ font-weight:600; font-size:1.05rem; }}
  .s {{ color:var(--muted); font-size:.85rem; }}
  footer {{ color:var(--muted); font-size:.8rem; margin-top:28px; }}
</style>
</head>
<body>
<main>
  <h1>Utah Avy Weather Reports</h1>
  <p class="when">Generated {htmlmod.escape(stamp)}</p>
  {"".join(rows)}
  <footer>Updated twice a day (about 7:30 AM and 7:30 PM Mountain time). Sources: Utah Avalanche
  Center, NWS Salt Lake City, University of Utah, NCEI and weather.gov station observations.</footer>
</main>
</body>
</html>
"""
    with open(os.path.join(site_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(page)
    print(f"Wrote web page to {site_dir}")


def email_reports(pdfs: list[str], to: str, now: dt.datetime, subject_suffix: str = ""):
    """Email the PDFs as attachments with the subject "YYYYMMDD Utah Avy Weather Report".

    SMTP settings come from environment variables (never from the repo):
      SMTP_HOST      default smtp.gmail.com
      SMTP_PORT      default 465 (SSL); 587 uses STARTTLS
      SMTP_USER      login, usually the sending address
      SMTP_PASSWORD  for Gmail, an app password (Google Account > Security > App passwords)
      EMAIL_FROM     optional, defaults to SMTP_USER
    `to` may hold several addresses separated by commas. Raises on failure so a
    scheduled run shows up as failed."""
    import smtplib
    from email.message import EmailMessage

    host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("SMTP_PORT", "465"))
    user, password = os.environ.get("SMTP_USER"), os.environ.get("SMTP_PASSWORD")
    if not (user and password):
        raise RuntimeError("Email requested but SMTP_USER / SMTP_PASSWORD aren't set")
    recipients = [a.strip() for a in to.split(",") if a.strip()]

    msg = EmailMessage()
    suffix = subject_suffix.strip()
    msg["Subject"] = f"{now:%Y%m%d} Utah Avy Weather Report" + (f" {suffix}" if suffix else "")
    msg["From"] = os.environ.get("EMAIL_FROM", user)
    msg["To"] = ", ".join(recipients)
    names = [os.path.basename(p) for p in pdfs if os.path.isfile(p)]
    msg.set_content(
        f"Utah avalanche & mountain weather reports generated {now:%A, %B %d %Y at %H:%M %Z}.\n\n"
        + "\n".join(f"  - {n}" for n in names)
        + "\n\nSources: Utah Avalanche Center, NWS Salt Lake City, University of Utah, "
          "NCEI and weather.gov station observations.\n")
    for p in pdfs:
        if os.path.isfile(p):
            with open(p, "rb") as f:
                msg.add_attachment(f.read(), maintype="application", subtype="pdf",
                                   filename=os.path.basename(p))

    if port == 465:
        with smtplib.SMTP_SSL(host, port, timeout=60) as s:
            s.login(user, password)
            s.send_message(msg)
    else:
        with smtplib.SMTP(host, port, timeout=60) as s:
            s.starttls()
            s.login(user, password)
            s.send_message(msg)
    print(f"Emailed {len(names)} reports to {len(recipients)} recipient(s): {msg['Subject']}")


def publish(pdfs: list[str], dest: str):
    """Copy the new reports into `dest`, then delete every other file directly in it.
    Copying first means a failure never leaves the folder empty. Subfolders are left
    alone. On Google Drive, deleted files go to the Drive trash."""
    if not all(os.path.isfile(p) for p in pdfs):
        print(f"Not publishing: a report is missing, so {dest} was left as it was")
        return
    if not os.path.isdir(dest):
        print(f"Not publishing: {dest} doesn't exist (is Google Drive running?)")
        return
    keep = set()
    for p in pdfs:
        shutil.copy2(p, os.path.join(dest, os.path.basename(p)))
        keep.add(os.path.basename(p))
    removed = []
    for name in os.listdir(dest):
        path = os.path.join(dest, name)
        if name not in keep and os.path.isfile(path):
            try:
                os.remove(path)
                removed.append(name)
            except OSError as e:
                print(f"  couldn't delete {path}: {e}")
    print(f"Published {len(keep)} reports to {dest}"
          + (f"; deleted {len(removed)} old file(s): {', '.join(removed)}" if removed else ""))


if __name__ == "__main__":
    main()
