#!/usr/bin/env python3
"""
Comprehensive Alta / Salt Lake mountains weather report -> PDF.

Forecast sections (scraped):
  * Utah Avalanche Center - Salt Lake forecast (text + danger rose + problem graphics)
  * NWS Salt Lake City Area Forecast Discussion (AFD)
  * NWS point forecast for Alta: graphical meteograms + charts/table built from
    the tabular ("digital") forecast
  * University of Utah GEFS downscaled ensemble plume for Alta-Collins

History sections:
  * NCEI: the station is found with the HOMR API (https://www.ncei.noaa.gov/access/homr/api),
    and its daily data comes from the NCEI Access Data Service. HOMR holds station
    *metadata* only. The one NCEI station at Alta is the ALTA co-op station
    (GHCND:USC00420072), which reports daily max/min temperature, precipitation and
    snow, but no wind.
  * Mesonet stations ALTA - BASE, COLLINS, TOP OF COLLINS, MOUNT BALDY and
    LCC - CARDIFF PEAK: daily temperature, wind speed and wind direction candlesticks.
    NCEI/HOMR doesn't carry these stations. Data comes from Synoptic Data (MesoWest)
    when a working token is set (--synoptic-token or SYNOPTIC_TOKEN), which gives
    complete hourly data. Without one it uses the public NWS API (api.weather.gov,
    the same observations the weather.gov timeseries pages show), which needs no key
    but has readings for only about half the hours. Both serve ~7 days, so each run
    saves what it gets in data/ and the charts fill toward the full window over time.

Usage:
    python weather_report.py                      # writes reports/weather_report_<timestamp>.pdf
    python weather_report.py --days 30 -o out.pdf
    set SYNOPTIC_TOKEN=xxxx  (PowerShell: $env:SYNOPTIC_TOKEN="xxxx")

Requirements: requests, beautifulsoup4, pandas, numpy, matplotlib, pillow
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import os
import re
import sys
import textwrap
import traceback
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

# --------------------------------------------------------------------------- config
TZ = ZoneInfo("America/Denver")
LAT, LON = 40.5846, -111.6213
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; AltaWeatherReport/1.0)"}
TIMEOUT = 45

UAC_URL = "https://utahavalanchecenter.org/forecast/salt-lake"
AFD_URL = ("https://forecast.weather.gov/product.php?site=SLC&issuedby=SLC&product=AFD"
           "&format=CI&version=1&glossary=1")
NWS_GRAPHICAL_URL = (f"https://forecast.weather.gov/MapClick.php?lat={LAT}&lon={LON}"
                     "&unit=0&lg=english&FcstType=graphical")
NWS_DIGITAL_URL = (f"https://forecast.weather.gov/MapClick.php?lat={LAT}&lon={LON}"
                   "&lg=english&&FcstType=digital")
PLUME_PAGE = "https://weather.utah.edu/index.php"
PLUME_STATION = "CLN"  # Alta-Collins

HOMR_SEARCH = "https://www.ncei.noaa.gov/access/homr/services/station/search"
NCEI_DATA = "https://www.ncei.noaa.gov/access/services/data/v1"

SYNOPTIC_BASE = "https://api.synopticdata.com/v2"
NWS_API = "https://api.weather.gov"
WRH_TIMESERIES = "https://www.weather.gov/wrh/timeseries"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
# (display name, elevation ft, station ID). The IDs are the same in Synoptic/MesoWest
# and the NWS API. Synoptic lists Mount Baldy as "ALTA - MT BALDY"; if an ID ever
# stops resolving there, the station is matched by name instead.
ALTA_STATIONS = [
    ("ALTA - BASE", 8560, "ATB"),
    ("ALTA - COLLINS", 9662, "CLN"),
    ("ALTA - TOP OF COLLINS", 10443, "ALT"),
    ("ALTA - MOUNT BALDY", 11066, "AMB"),
    ("LCC - CARDIFF PEAK", 10059, "IFF"),
]

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


def get(url: str, **kw) -> requests.Response:
    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT, **kw)
    r.raise_for_status()
    return r


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

    def save(self, fig, section: str | None = None):
        self.page_no += 1
        if section:
            self.toc.append((section, self.page_no))
        fig.text(0.5, 0.012, f"Alta weather report  ·  generated {self.generated:%a %b %d %Y %H:%M %Z}"
                              f"  ·  page {self.page_no}", ha="center", fontsize=7, color=C_MUTED)
        self.pdf.savefig(fig)
        plt.close(fig)


def header(fig, title: str, subtitle: str = ""):
    fig.text(0.06, 0.965, title, fontsize=15, weight="bold", color=C_TEXT, va="top")
    if subtitle:
        fig.text(0.06, 0.937, subtitle, fontsize=8.5, color=C_MUTED, va="top")


def text_pages(rep: Report, title: str, text: str, source: str, mono: bool = False,
               section: str | None = None):
    """Flow long text over as many portrait pages as needed."""
    width = 98 if mono else 105
    lines: list[str] = []
    for para in text.splitlines():
        if not para.strip():
            lines.append("")
            continue
        if mono:
            lines.extend(textwrap.wrap(para, width, replace_whitespace=False,
                                       drop_whitespace=False) or [""])
        else:
            lines.extend(textwrap.wrap(para, width) or [""])
    # collapse runs of blank lines
    cleaned = []
    for ln in lines:
        if ln == "" and cleaned and cleaned[-1] == "":
            continue
        cleaned.append(ln)
    per_page = 72
    while cleaned and cleaned[-1] == "":
        cleaned.pop()
    chunks = [cleaned[i:i + per_page] for i in range(0, len(cleaned), per_page)] or [[]]
    if len(chunks) > 1 and len(chunks[-1]) <= 4:  # avoid a near-empty last page
        chunks[-2].extend(chunks.pop())
    for i, chunk in enumerate(chunks):
        fig = plt.figure(figsize=PAGE)
        header(fig, title + (f" (cont. {i + 1}/{len(chunks)})" if i else ""), source if i == 0 else "")
        fig.text(0.06, 0.915, "\n".join(chunk), fontsize=7.6, va="top", ha="left",
                 family="monospace" if mono else "sans-serif", color=C_TEXT, linespacing=1.32,
                 parse_math=False)
        rep.save(fig, section if i == 0 else None)


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
    # page text is one token per line; rejoin into paragraphs
    headings = {"Special Announcements", "Weather and Snow", "Recent Avalanches",
                "Additional Information", "Description", "Type", "Location",
                "Likelihood", "Size"}
    out, buf = [], []
    for ln in body.split("\n"):
        if ln in headings or ln.startswith("Avalanche Problem #"):
            if buf:
                out.append(" ".join(buf)); buf = []
            if ln in {"Type", "Location", "Likelihood", "Size"}:
                continue
            out.append("")
            out.append(ln.upper())
        elif ln in {"Low", "Moderate", "Considerable", "High", "Extreme",
                    "Learn how to read the forecast", "here"}:
            continue
        else:
            buf.append(ln)
            if re.search(r"[.!?:]$", ln) and len(" ".join(buf)) > 60:
                out.append(" ".join(buf)); buf = []
    if buf:
        out.append(" ".join(buf))
    para_text = "\n".join(out)
    para_text = re.sub(r"\s+([.,;:!?])", r"\1", para_text)

    # images: danger rose + problem roses/graphics, all under /sites/default/files/forecast/
    imgs, caps = [], []
    for img in main.find_all("img"):
        src = img.get("src") or ""
        if "/files/forecast/" in src:
            url = requests.compat.urljoin(UAC_URL, src)
            try:
                imgs.append(fetch_image(url))
                caps.append("Overall danger rose" if not imgs[:-1] else f"Avalanche problem #{len(imgs) - 1} location")
            except Exception:
                pass

    if imgs:
        image_page(rep, "Avalanche Forecast — Salt Lake Mountains", imgs[:3],
                   f"Utah Avalanche Center  ·  {UAC_URL}", section="Avalanche forecast (UAC)",
                   captions=caps[:3])
        text_pages(rep, "Avalanche Forecast — Salt Lake Mountains", para_text, UAC_URL)
    else:
        text_pages(rep, "Avalanche Forecast — Salt Lake Mountains", para_text, UAC_URL,
                   section="Avalanche forecast (UAC)")


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


def nws_point_section(rep: Report):
    # 1) the graphical meteograms, as the page renders them (two 48 h windows)
    gsoup = soup_of(NWS_GRAPHICAL_URL)
    img = next((i for i in gsoup.find_all("img") if "Plotter.php" in (i.get("src") or "")), None)
    meteos = []
    if img:
        base = requests.compat.urljoin(NWS_GRAPHICAL_URL, img["src"])
        for ahour in (0, 48):
            url = re.sub(r"ahour=\d+", f"ahour={ahour}", base)
            try:
                meteos.append(fetch_image(url))
            except Exception:
                pass
    if meteos:
        image_page(rep, "NWS Point Forecast — Alta (graphical)", meteos,
                   f"forecast.weather.gov  ·  {LAT}, {LON}  ·  hours 0–48 and 48–96",
                   section="NWS Alta forecast", landscape=False,
                   captions=["Next 48 hours", "Hours 48–96"])

    # 2) the tabular forecast, all 7 days (the page shows 48 h at a time)
    frames = []
    for ah in (0, 48, 96, 144):
        try:
            frames.append(parse_digital(soup_of(NWS_DIGITAL_URL + f"&AheadHour={ah}")))
        except Exception:
            pass
    if not frames:
        raise RuntimeError("NWS tabular forecast unavailable")
    fc = pd.concat(frames)
    fc = fc[~fc.index.duplicated()].sort_index()

    # --- charts
    fig, axes = plt.subplots(4, 1, figsize=PAGE, sharex=True,
                             gridspec_kw=dict(height_ratios=[3, 2.4, 1.6, 2], hspace=0.18,
                                              left=0.09, right=0.95, top=0.9, bottom=0.07))
    header(fig, "NWS Point Forecast — Alta (hourly charts)",
           f"Built from the tabular forecast  ·  {fc.index[0]:%a %b %d %H:%M} – {fc.index[-1]:%a %b %d %H:%M %Z}")
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
    step = max(1, len(fc) // 40)
    for t, d in zip(fc.index[::step], fc.wdir[::step]):
        if isinstance(d, str) and d:
            ax.text(t, ymax * 0.97, d, fontsize=5.5, ha="center", va="top", color=C_MUTED,
                    rotation=90)
    ax.set_ylabel("mph")
    ax.set_title("Sustained wind (line)" + (", gusts (▲)" if has_gust else ", no gusts forecast")
                 + ", direction (labels along top)", loc="left", fontsize=8, color=C_MUTED)

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
    for a in axes:
        a.grid(True, which="major", axis="x", color="#bbbbbb")
    rep.save(fig, None if meteos else "NWS Alta forecast")

    # --- daily summary table
    daily = fc.groupby(fc.index.date).agg(
        high=("temp", "max"), low=("temp", "min"), wind_max=("wind", "max"),
        gust_max=("gust", "max"), pop_max=("pop", "max"), sky_mean=("sky", "mean"))
    precip_types = []
    for d in daily.index:
        day = fc[fc.index.date == d]
        kinds = []
        for col in ["snow", "rain", "thunder", "freezing_rain", "sleet"]:
            if col in day and day[col].astype(str).str.strip().ne("").any():
                vals = sorted(set(v for v in day[col] if v))
                kinds.append(f"{col.replace('_', ' ')} ({'/'.join(vals)})")
        precip_types.append(", ".join(kinds) or "—")
    daily["precip"] = precip_types

    fig = plt.figure(figsize=PAGE)
    header(fig, "NWS Point Forecast — Alta (daily summary)", NWS_DIGITAL_URL[:110])
    ax = fig.add_axes([0.05, 0.45, 0.9, 0.45]); ax.axis("off")
    cell = [[pd.Timestamp(d).strftime("%a %b %d"), f"{r.high:.0f}", f"{r.low:.0f}",
             f"{r.wind_max:.0f}", "—" if np.isnan(r.gust_max) else f"{r.gust_max:.0f}",
             f"{r.pop_max:.0f}", f"{r.sky_mean:.0f}", textwrap.fill(r.precip, 34)]
            for d, r in daily.iterrows()]
    tbl = ax.table(cellText=cell, loc="upper center", cellLoc="center",
                   colLabels=["Day", "High °F", "Low °F", "Max wind", "Max gust", "Max PoP %",
                              "Sky %", "Precip type (chance)"],
                   colWidths=[0.12, 0.08, 0.08, 0.09, 0.09, 0.1, 0.07, 0.37])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 2.2)
    for (r, c), cl in tbl.get_celld().items():
        cl.set_edgecolor("#cccccc")
        if r == 0:
            cl.set_facecolor("#e8eef3"); cl.set_text_props(weight="bold")
    fig.text(0.06, 0.40, "Wind in mph. Precip type chances use NWS wording "
                         "(SChc = slight chance, Chc = chance, Lkly = likely, Ocnl = occasional, Def = definite).",
             fontsize=7.5, color=C_MUTED)
    rep.save(fig)


# --------------------------------------------------------------------------- forecast: plumes
def plume_section(rep: Report):
    # Without a runcode the page serves the latest run; we still read the run from the HTML.
    r = get(PLUME_PAGE, params={"t": "ensgefsds", "d": "PL", "r": PLUME_STATION})
    html = r.text
    paths = re.findall(r'"([^"]*ENSGEFSDSPL[^"]*\.png)"', html)
    if not paths:
        # try today's and yesterday's runs explicitly
        for back in range(0, 3):
            day = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=back)
            for hh in ("18", "12", "06", "00"):
                rc = f"{day:%Y%m%d}{hh}"
                html = get(PLUME_PAGE, params={"runcode": rc, "t": "ensgefsds", "d": "PL",
                                               "r": PLUME_STATION}).text
                paths = re.findall(r'"([^"]*ENSGEFSDSPL[^"]*\.png)"', html)
                if paths:
                    break
            if paths:
                break
    if not paths:
        raise RuntimeError("No plume images found on weather.utah.edu")
    imgs = [fetch_image(requests.compat.urljoin(PLUME_PAGE, p)) for p in paths]
    run = re.search(r"(\d{10})F\d+", paths[0])
    run_txt = f"run {run.group(1)[:8]} {run.group(1)[8:]}Z" if run else ""
    for i, img in enumerate(imgs):
        image_page(rep, "GEFS Downscaled Ensemble Plumes — Alta-Collins", [img],
                   f"University of Utah  ·  {run_txt}  ·  weather.utah.edu (t=ensgefsds, d=PL, r=CLN)",
                   section="Ensemble plumes (U of U)" if i == 0 else None,
                   landscape=img.width > img.height)


# --------------------------------------------------------------------------- history: charts
def candlestick_temp(ax, daily: pd.DataFrame, title: str):
    """daily needs columns high, low and optionally open, close."""
    x = mdates.date2num(pd.to_datetime(daily.index))
    has_oc = {"open", "close"} <= set(daily.columns) and daily[["open", "close"]].notna().any().any()
    for xi, row in zip(x, daily.itertuples()):
        if np.isnan(row.high) or np.isnan(row.low):
            continue
        if has_oc and not (np.isnan(row.open) or np.isnan(row.close)):
            up = row.close >= row.open
            color = C_WARM if up else C_COOL
            lo, hi = sorted([row.open, row.close])
            ax.vlines(xi, row.low, row.high, color=color, lw=1.1, zorder=2)
            ax.bar(xi, max(hi - lo, 0.4), bottom=lo, width=0.62, color=color,
                   edgecolor=color, zorder=3)
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
        ax.set_ylim(lo - 6, hi + 6)
    if has_oc:
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=C_WARM, label="Warmed (midnight→midnight)"),
                           Patch(color=C_COOL, label="Cooled (midnight→midnight)")],
                  loc="upper left", fontsize=7, frameon=False, ncol=2)


def date_axis(ax):
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
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
def ncei_section(rep: Report, days: int):
    # 1) find the station with the HOMR API
    homr = get(HOMR_SEARCH, params={"name": "ALTA", "state": "UT"}).json()
    stations = homr["stationCollection"]["stations"]
    ghcnd, name, elev = None, "ALTA", None
    for st in stations:
        ids = {i["idType"]: i["id"] for i in st.get("identifiers", [])}
        if "GHCND" in ids and st.get("header", {}).get("por", {}).get("endDate") == "Present":
            ghcnd = ids["GHCND"]
            name = st.get("header", {}).get("preferredName", name)
            for loc in st.get("location", {}).get("elevations", []) if isinstance(st.get("location"), dict) else []:
                elev = loc.get("elevations", [{}])[0].get("elevation") if isinstance(loc, dict) else None
            break
    if not ghcnd:
        raise RuntimeError("HOMR returned no active GHCN-Daily station named ALTA, UT")

    # 2) daily data from the NCEI Access Data Service
    end = dt.datetime.now(TZ).date()
    start = end - dt.timedelta(days=days)
    data = get(NCEI_DATA, params={
        "dataset": "daily-summaries", "stations": ghcnd, "startDate": start.isoformat(),
        "endDate": end.isoformat(), "dataTypes": "TMAX,TMIN,PRCP,SNOW,SNWD",
        "format": "json", "units": "standard"}).json()
    if not data:
        raise RuntimeError(f"NCEI returned no daily data for {ghcnd} since {start}")
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df["DATE"])
    for c in ["TMAX", "TMIN", "PRCP", "SNOW", "SNWD"]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df = df.reindex(pd.date_range(start, end, freq="D"))
    daily = pd.DataFrame({"high": df.TMAX, "low": df.TMIN}, index=df.index)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=LAND, sharex=True,
                                 gridspec_kw=dict(height_ratios=[3, 2], hspace=0.15,
                                                  left=0.07, right=0.96, top=0.86, bottom=0.08))
    header(fig, f"{name} — NCEI daily observations, last {days} days",
           f"Station {ghcnd} (found via the HOMR API)  ·  data from the NCEI Access Data Service  ·  "
           f"NCEI data usually lag a few days")
    candlestick_temp(a1, daily, "Daily temperature (high/low)")
    a2.bar(df.index, df.SNOW, color=C_COOL, width=0.62, label="New snow (in)")
    a2.set_ylabel("in"); a2.set_title("Snowfall and snow depth", loc="left", fontsize=10, weight="bold")
    if df.SNWD.notna().any():
        a3 = a2.twinx()
        a3.plot(df.index, df.SNWD, color=C_MEAN, lw=1.5, label="Snow depth (in)")
        a3.set_ylabel("depth (in)"); a3.grid(False); a3.spines["right"].set_visible(True)
        h1, l1 = a2.get_legend_handles_labels(); h2, l2 = a3.get_legend_handles_labels()
        a2.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7, frameon=False)
    date_axis(a2)
    last = df.dropna(subset=["TMAX", "TMIN"], how="all").index.max()
    if pd.notna(last):
        fig.text(0.96, 0.89, f"latest NCEI day: {last:%b %d}", ha="right", fontsize=8, color=C_MUTED)
    rep.save(fig, "History: NCEI ALTA co-op station")


# --------------------------------------------------------------------------- history: Synoptic
def _norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def synoptic_resolve(token: str, overrides: dict[str, str]) -> list[dict]:
    meta = get(f"{SYNOPTIC_BASE}/stations/metadata",
               params={"token": token, "radius": f"{LAT},{LON},8", "complete": 1}).json()
    if meta.get("SUMMARY", {}).get("RESPONSE_CODE") != 1:
        raise RuntimeError(f"Synoptic metadata error: {meta.get('SUMMARY')}")
    by_name = {_norm(s["NAME"]): s for s in meta.get("STATION", [])}
    by_id = {s["STID"]: s for s in meta.get("STATION", [])}
    out = []
    for name, elev, stid in ALTA_STATIONS:
        st = (by_id.get(overrides.get(name, "")) or by_id.get(stid)
              or by_name.get(_norm(name)) or by_name.get(_norm(name.replace("MOUNT", "MT"))))
        out.append({"name": name, "elev": elev, "stid": st["STID"] if st else None,
                    "meta_elev": st.get("ELEVATION") if st else None})
    return out


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


def merge_cache(stid: str, fresh: pd.DataFrame) -> pd.DataFrame:
    """The free Synoptic plan only serves the last 7 days, so every run saves what it
    gets to data/synoptic_<STID>.csv and merges it with earlier runs. Running the
    report at least once a week builds up the full history window over time."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"synoptic_{stid}.csv")
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
        "high": g.temp.max(), "low": g.temp.min(),
        "open": g.temp.apply(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan),
        "close": g.temp.apply(lambda s: s.dropna().iloc[-1] if s.notna().any() else np.nan),
    })
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


def wind_rose(fig, beside_ax, obs: pd.DataFrame, hours: int = 24):
    """16-sector wind rose of the last `hours` of readings, placed to the right of
    `beside_ax`. Bars are stacked by speed band and show % of readings; calm
    readings (< 1 mph) have no direction, so they're counted separately."""
    pos = beside_ax.get_position()
    size_h = pos.height
    size_w = size_h * fig.get_figheight() / fig.get_figwidth()  # keep it round
    x0 = 0.885 - size_w / 2
    ax = fig.add_axes([x0, pos.y0, size_w, size_h], projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]), ["N", "E", "S", "W"], fontsize=7)
    ax.tick_params(axis="y", labelsize=5.5, colors=C_MUTED)
    ax.set_title(f"Wind rose, last {hours} h", fontsize=9, weight="bold", pad=10)

    cutoff = obs.index.max() - pd.Timedelta(hours=hours) if len(obs) else None
    recent = obs[obs.index > cutoff] if cutoff is not None else obs
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
        sel = (windy["wind"] >= lo) & (windy["wind"] < hi)
        pct = np.bincount(sector[sel], minlength=16) / n * 100
        ax.bar(theta, pct, width=np.deg2rad(22.5) * 0.9, bottom=bottom, color=color,
               edgecolor="white", linewidth=0.4,
               label=f"{lo}–{hi:.0f}" if np.isfinite(hi) else f"{lo}+")
        bottom += pct
    top = max(bottom.max(), 1)
    ax.set_ylim(0, top * 1.08)
    ax.set_yticks(np.linspace(0, top, 3)[1:], [f"{v:.0f}%" for v in np.linspace(0, top, 3)[1:]])
    ax.set_rlabel_position(135)
    # speed legend in the free space above the rose (right of the wind-speed panel)
    fig.legend(*ax.get_legend_handles_labels(), title="mph", loc="lower center",
               bbox_to_anchor=(0.885, pos.y1 + 0.075), fontsize=6.5, title_fontsize=7,
               frameon=False, ncol=1, handlelength=1, handleheight=0.8, labelspacing=0.25)
    fig.text(0.885, pos.y0 - 0.035, f"{n} readings · calm {calm / n:.0%}", ha="center",
             fontsize=6.5, color=C_MUTED)


def nws_observations(stid: str, days: int) -> pd.DataFrame:
    """Observations from the public NWS API (api.weather.gov), no key needed.

    It keeps roughly the last 7 days and, for these mesonet stations, usually has
    readings for about half the hours of each day.
    """
    hdr = {"User-Agent": "(alta-weather-report, personal use)", "Accept": "application/geo+json"}
    start = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)
    url, params, rows = (f"{NWS_API}/stations/{stid}/observations",
                         {"start": start.strftime("%Y-%m-%dT%H:%M:%SZ")}, [])
    for _ in range(30):  # follow pagination
        r = requests.get(url, params=params, headers=hdr, timeout=TIMEOUT)
        r.raise_for_status()
        js = r.json()
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
    One visit per station per run."""
    key = (tuple(stids), days)
    if key in _WRH_CACHE:  # main() builds the PDF twice; scrape only once
        return _WRH_CACHE[key]
    from playwright.sync_api import sync_playwright

    # The page accepts at most 720 hours; asking for more silently returns only 72.
    hours = min(days * 24 + 24, 720)
    out: dict[str, pd.DataFrame] = {}
    now = dt.datetime.now(TZ)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(user_agent=HEADERS["User-Agent"])
        for stid in stids:
            try:
                page.goto(f"{WRH_TIMESERIES}?site={stid}&hourly=true&hours={hours}",
                          wait_until="networkidle", timeout=90_000)
                page.wait_for_selector("#OBS_DATA tr:nth-child(3)", timeout=60_000)
                rows = page.evaluate(_TABLE_JS)
                out[stid] = _parse_wrh_table(rows, now)
            except Exception as e:
                print(f"  weather.gov scrape failed for {stid}: {e}", file=sys.stderr)
        browser.close()
    _WRH_CACHE[key] = out
    return out


def _parse_wrh_table(rows: list[list[str]], now: dt.datetime) -> pd.DataFrame:
    hdr = [re.sub(r"\s+", " ", h) for h in rows[0]]

    def col(prefix):
        return next((i for i, h in enumerate(hdr) if h.startswith(prefix)), None)

    i_t, i_temp, i_dir, i_spd, i_gst = (col("Date/Time"), col("Temp."), col("Wind Direction"),
                                        col("Wind Speed"), col("Wind Gust"))
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
        recs.append((ts, num(r[i_temp]), spd, gst, wdir))
    df = pd.DataFrame(recs, columns=["time", "temp", "wind", "gust", "wdir"]).set_index("time")
    df.index = df.index.tz_localize(TZ, ambiguous="NaT", nonexistent="shift_forward")
    df = df[df.index.notna()]
    return df[~df.index.duplicated()].sort_index()


def fetch_station_obs(token: str | None, days: int, overrides: dict[str, str]):
    """Fresh observations for every station, trying sources in order:
      1. the weather.gov timeseries page (full hourly history for the whole window),
      2. Synoptic, when a token is set and works (last 7 days on the free plan),
      3. the public NWS API (about 7 days, roughly half the hours).
    Returns (stations, {stid: df}, {stid: source label})."""
    stations = [{"name": n, "elev": el, "stid": overrides.get(n, sid)}
                for n, el, sid in ALTA_STATIONS]
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


def synoptic_section(rep: Report, token: str | None, days: int, overrides: dict[str, str]):
    stations, series, sources = fetch_station_obs(token, days, overrides)
    if not series:
        raise RuntimeError("No station data from weather.gov, Synoptic or the NWS API")
    for stid, df in series.items():
        df["src"] = sources[stid]
    series = {stid: merge_cache(stid, df) for stid, df in series.items()}
    window_start = pd.Timestamp(dt.datetime.now(TZ).date() - dt.timedelta(days=days - 1)).tz_localize(TZ)

    first = True
    for s in stations:
        title = f"{s['name']}  ({s['elev']:,} ft)"
        if not s["stid"] or s["stid"] not in series or series[s["stid"]].empty:
            error_page(rep, f"History: {title}",
                       f"No data returned for this station (ID: {s['stid'] or 'not found'}).",
                       section="History: Alta stations" if first else None)
            first = False
            continue
        obs = series[s["stid"]]
        obs = obs[obs.index >= window_start]
        srcs = (", ".join(sorted(obs["src"].dropna().unique())) if "src" in obs
                else sources.get(s["stid"], "?"))
        t, w, wd = daily_stats(obs, days)
        fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=LAND, sharex=True,
                                         gridspec_kw=dict(height_ratios=[3, 2.5, 2.2], hspace=0.28,
                                                          left=0.07, right=0.725, top=0.86,
                                                          bottom=0.07))
        have = t.high.notna().sum()
        header(fig, f"{title} — last {days} days",
               f"Station {s['stid']}  ·  data: {srcs}  ·  {len(obs):,} observations  ·  "
               f"{have} of {days} days with data  ·  daily = local calendar day ({TZ.key})")
        if have < days:
            fig.text(0.93, 0.89, "Sources serve ~7 days; older days come from saved earlier runs "
                                 "(data/ folder)", ha="right", fontsize=7.5, color=C_WARM)
        candlestick_temp(a1, t, "Daily temperature — wick: high/low, body: midnight→midnight change")
        wind_chart(a2, w, "Daily wind speed")
        direction_chart(a3, wd, "Daily wind direction")
        date_axis(a3)
        x0 = mdates.date2num(t.index[0]) - 0.7
        a3.set_xlim(x0, mdates.date2num(t.index[-1]) + 0.7)
        wind_rose(fig, a3, series[s["stid"]])
        rep.save(fig, "History: Alta stations" if first else None)
        first = False
    used = sorted(set(sources.values()))
    return "ok" if used == ["weather.gov"] else f"ok ({', '.join(used)})"


# --------------------------------------------------------------------------- title + contents
def title_page(rep: Report, days: int, statuses: list[tuple[str, str]]):
    fig = plt.figure(figsize=PAGE)
    fig.text(0.06, 0.90, "Alta & Salt Lake Mountains", fontsize=24, weight="bold", color=C_TEXT)
    fig.text(0.06, 0.862, "Comprehensive weather report", fontsize=15, color=C_MUTED)
    fig.text(0.06, 0.832, f"Generated {rep.generated:%A, %B %d %Y at %H:%M %Z}  ·  "
                          f"history window: last {days} days", fontsize=9, color=C_MUTED)
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
        fig.text(0.42, y, st, fontsize=9, color=C_COOL if st == "ok" else C_WARM)
        y -= 0.022
    y -= 0.03
    srcs = [UAC_URL, AFD_URL, NWS_GRAPHICAL_URL, NWS_DIGITAL_URL,
            f"{PLUME_PAGE}?t=ensgefsds&d=PL&r={PLUME_STATION}",
            "https://www.ncei.noaa.gov/access/homr/api", "https://api.synopticdata.com/v2",
            "https://api.weather.gov (station observations)"]
    fig.text(0.06, y, "Sources", fontsize=12, weight="bold"); y -= 0.025
    for s in srcs:
        fig.text(0.08, y, s[:115], fontsize=6.8, color=C_MUTED, family="monospace"); y -= 0.018
    return fig


# --------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--output", help="output PDF path")
    ap.add_argument("--days", type=int, default=30, help="history window in days (default 30)")
    ap.add_argument("--synoptic-token", default=os.environ.get("SYNOPTIC_TOKEN"),
                    help="Synoptic Data API token (or env SYNOPTIC_TOKEN)")
    ap.add_argument("--stid", action="append", default=[],
                    help="override a station ID, e.g. --stid 'ALTA - BASE=ATB' (repeatable)")
    args = ap.parse_args(argv)

    now = dt.datetime.now(TZ)
    out = args.output or os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports",
                                      f"weather_report_{now:%Y%m%d_%H%M}.pdf")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    overrides = dict(s.split("=", 1) for s in args.stid)

    # Build body pages into a temp PDF first so the title page can list page numbers;
    # then write the title page + body. (matplotlib can't insert pages, so render twice.)
    sections = [
        ("Avalanche forecast (UAC)", lambda r: avalanche_section(r)),
        ("NWS forecast discussion", lambda r: afd_section(r)),
        ("NWS Alta point forecast", lambda r: nws_point_section(r)),
        ("Ensemble plumes (U of U)", lambda r: plume_section(r)),
        ("History: NCEI / HOMR", lambda r: ncei_section(r, args.days)),
        ("History: Alta stations", lambda r: synoptic_section(r, args.synoptic_token, args.days, overrides)),
    ]

    def build(path, statuses_in=None, toc_in=None):
        statuses = []
        with PdfPages(path) as pdf:
            rep = Report(pdf, generated=now)
            if toc_in is not None:
                rep.toc = toc_in
                rep.save(title_page(rep, args.days, statuses_in))
                rep.toc = []
            for name, fn in sections:
                print(f"- {name} ...", flush=True)
                try:
                    statuses.append((name, fn(rep) or "ok"))
                except Exception as e:
                    traceback.print_exc()
                    error_page(rep, name, f"{type(e).__name__}: {e}", section=name)
                    statuses.append((name, f"failed: {type(e).__name__}"))
            d = pdf.infodict()
            d["Title"] = "Alta & Salt Lake Mountains weather report"
            d["CreationDate"] = now
        return rep.toc, statuses

    # Pass 1 collects the contents; pass 2 writes the final file with a title page.
    # Pass 2 re-fetches everything, so cache HTTP responses between passes.
    _cache: dict = {}
    _orig_get = requests.get

    def cached_get(url, *a, **kw):
        key = (url, tuple(sorted((kw.get("params") or {}).items())))
        if key not in _cache:
            _cache[key] = _orig_get(url, *a, **kw)
        return _cache[key]

    requests.get = cached_get
    try:
        tmp = out + ".tmp.pdf"
        toc, statuses = build(tmp)
        os.remove(tmp)
        build(out, statuses_in=statuses, toc_in=toc)
    finally:
        requests.get = _orig_get
    print(f"\nWrote {out}")
    return out


if __name__ == "__main__":
    main()
