# Utah Avy Weather Report

A Python script that builds mountain weather and avalanche PDF reports for three areas of the
Wasatch, twice a day, and publishes them by email and at fixed web links.

| Report | File | Forecast points | History stations |
|---|---|---|---|
| **Alta & Little Cottonwood** | `alta_weather_report_YYYYMMDD_HHMM.pdf` | Alta Base, Alta-Collins Top, Mount Superior | ALTA - BASE, COLLINS, TOP OF COLLINS, MOUNT BALDY, LCC - CARDIFF PEAK |
| **Park City & Deer Valley** | `pc_weather_report_YYYYMMDD_HHMM.pdf` | PC Base, PC Jupiter | DV - BURNS, DV - EMPIRE PEAK |
| **Big Cottonwood Canyon** | `bcc_weather_report_YYYYMMDD_HHMM.pdf` | Spruces, Kessler, Flagstaff | SPRUCES, REYNOLDS PEAK, SOLITUDE HONEYCOMB PEAK |

## Latest reports

The newest reports are always at these links (updated after every scheduled run):

- Index page: <https://jmstrnthl.github.io/AvyWeatherReport/>
- Alta: <https://jmstrnthl.github.io/AvyWeatherReport/alta.pdf>
- Park City: <https://jmstrnthl.github.io/AvyWeatherReport/pc.pdf>
- Big Cottonwood: <https://jmstrnthl.github.io/AvyWeatherReport/bcc.pdf>
- Run status: <https://jmstrnthl.github.io/AvyWeatherReport/status.json>

## What's in a report

1. **Title page** – contents, the status of every section, and the sources used.
2. **Avalanche forecast** – the Utah Avalanche Center (UAC) Salt Lake forecast: danger rose,
   forecast text, and each avalanche problem with its location rose.
3. **NWS Area Forecast Discussion** – Salt Lake City.
4. **NWS point forecasts** (one page per point) – 7-day hourly charts (temperature, wind,
   precipitation chance, sky cover), a next-24-hours column with the steepest forecast warming
   marked, and a daily summary table.
5. **Ensemble plumes** – University of Utah GEFS-downscaled "Utah Snow Ensemble" plume and the
   RRFS hourly snow ensemble plume for the area's plume site.
6. **NCEI co-op page** – daily high/low, snowfall and snow depth from the area's NCEI co-op
   station (found with the HOMR API), plus a nearby station's 6-hour snowfall and hourly snow
   depth.
7. **Station history** (one page per station) – 30-day daily temperature (box-and-whisker,
   colored by the day's steepest 2-hour warming), wind speed, wind direction, GOES-18 satellite
   cloud cover for the last 7 days, and the last 24 hours of temperature, wind and a wind rose.
8. **Avalanches past 7 days** – UAC avalanches in the Salt Lake region by aspect and elevation,
   each avalanche colored by problem and faded by age, plus counts by type, problem, weak layer
   and trigger.

## How it runs

### Automatically (GitHub Actions)

`.github/workflows/daily-report.yml` runs on GitHub's servers:

| Time (MST) | UTC | What |
|---|---|---|
| 7:30 AM | 14:30 | Main run: build all three reports, email them, update the links |
| 8:00 AM | 15:00 | Catch-up: rebuilds only if the 7:30 run had a failed section or didn't publish |
| 7:30 PM | 02:30 | Main run |
| 8:00 PM | 03:00 | Catch-up |

GitHub schedules are in UTC, so under daylight saving (MDT) every run is an hour later in
local time. Scheduled runs can also start a few minutes late. Use **Actions → Daily weather
report → Run workflow** for a manual run (manual runs always build).

Each run:
- emails the PDFs with the subject `YYYYMMDD Utah Avy Weather Report` (catch-up runs add
  `(updated)`);
- publishes the PDFs, `index.html` and `status.json` to GitHub Pages;
- commits new station readings and satellite cloud cover to `data/` ("Update station
  history"), so pull before working locally: `git pull`.

**Repository secrets** (Settings → Secrets and variables → Actions):

| Secret | Needed for |
|---|---|
| `EMAIL_TO` | where to send the reports (comma-separate several) |
| `SMTP_USER` | the sending account, e.g. a Gmail address |
| `SMTP_PASSWORD` | that account's **app password** (Gmail: Google Account → Security → App passwords) |
| `SYNOPTIC_TOKEN` | optional; fallback source for station data |

Optional repository variables `SMTP_HOST` and `SMTP_PORT` for a non-Gmail sender
(defaults `smtp.gmail.com`, `465`).

### Locally

```powershell
pip install -r requirements.txt
$env:PLAYWRIGHT_BROWSERS_PATH = "$HOME\.playwright-browsers"
python -m playwright install chromium-headless-shell
```

Headless Chromium is used to read the weather.gov station pages, which are drawn by
JavaScript. The script uses `~/.playwright-browsers` automatically when it exists.

## Command-line options

```powershell
python weather_report.py [options]
```

| Option | What it does |
|---|---|
| *(none)* | Build all three daily reports into `reports/` |
| `--site alta` / `pc` / `bcc` | Build just one area (default `all`) |
| `--outdir FOLDER` | Where to write the PDFs (default `reports/` next to the script) |
| `--days N` | Station history window in days (default 30; weather.gov serves at most 30) |
| `--site-dir FOLDER` | Also write the web page: `alta.pdf`, `pc.pdf`, `bcc.pdf`, `index.html` and `status.json` (used by GitHub Pages) |
| `--email-to ADDRESSES` | Email the PDFs to these comma-separated addresses (or set `EMAIL_TO`); needs `SMTP_USER` and `SMTP_PASSWORD` |
| `--email-subject-suffix TEXT` | Added to the email subject, e.g. `(updated)` |
| `--publish-dir FOLDER` | After a full run, copy the PDFs here and **delete every other file in it**. Defaults to the Ski Patrol Google Drive folder; `--publish-dir ""` turns it off |
| `--synoptic-token TOKEN` | Synoptic Data token (or set `SYNOPTIC_TOKEN`); only a fallback source |
| `--stid "NAME=ID"` | Override a station's ID, e.g. `--stid "ALTA - BASE=ATB"` (repeatable) |
| `--season` | Build **season history reports** instead of the daily reports (see below) |
| `--season-start MM-DD` | Season start for `--season` (default `10-23`; the most recent one is used) |

Examples:

```powershell
# all three daily reports, no Google Drive copy
python weather_report.py --publish-dir ""

# just Park City, into another folder
python weather_report.py --site pc --outdir C:\temp\reports --publish-dir ""

# what the GitHub workflow runs
python weather_report.py --publish-dir "" --site-dir _site
```

### Season history report

```powershell
git pull
python weather_report.py --season
python weather_report.py --season --site bcc
python weather_report.py --season --season-start 11-01
```

Writes `reports/<area>_season_report_<start>_<today>.pdf` for each area, covering the season
start to today, with **history only** (no forecasts):

- the NCEI co-op page for the whole season;
- every station page, built from the saved station data in `data/` (no scraping), with the
  season-long charts only;
- **Avalanches This Season** – the season's UAC avalanches (aspect/elevation plots and counts);
- **Avalanche Activity by Week** – heatmaps of avalanches per week by problem, aspect and
  elevation band.

The station pages can only go back as far as the saved data, so `git pull` first to get the
readings GitHub has collected.

## Data and reliability

- **Retries:** every download is retried (3 more attempts after 10, 30 and 60 seconds),
  including pages that load but contain an error such as the NWS "Runtime Error" page.
- **Forecast fallback:** if a forecast.weather.gov table still fails, that point's forecast
  comes from the NWS API (api.weather.gov) instead; the page and `status.json` say so.
- **Station data:** weather.gov station pages first, then Synoptic (if a token is set), then
  the NWS API. Readings are saved to `data/obs_<ID>.csv` (up to 400 days) and merged in, so
  history builds up even when a source has less.
- **Satellite cloud cover:** one GOES-18 Clear Sky Mask image per hour from NOAA's public
  AWS bucket; the hourly values for every station are saved in `data/goes_cloud.csv`, so each
  run only downloads new hours.
- **Sensor spikes** in station snow depth and 6-hour snowfall are filtered out (the page says
  how many).
- A section that still fails becomes an error page; the rest of the report builds normally.

## Configuration

Areas, forecast points, stations, plume sites, NCEI stations and snow stations are set in the
`SITES` block near the top of `weather_report.py`. Each area is one `Site(...)` entry, so
adding a forecast point or station is a one-line change.

## Sources

- Utah Avalanche Center forecast and observations explorer: <https://utahavalanchecenter.org>
- NWS Salt Lake City forecast discussion and point forecasts: <https://forecast.weather.gov>
- NWS API: <https://api.weather.gov>
- weather.gov station timeseries: <https://www.weather.gov/wrh/timeseries>
- University of Utah ensemble plumes: <https://weather.utah.edu>
- NCEI HOMR and Access Data Service: <https://www.ncei.noaa.gov/access/homr/api>
- GOES-18 Clear Sky Mask (NOAA Open Data on AWS): `noaa-goes18` bucket
- Synoptic Data (optional): <https://synopticdata.com>

## Older version

The version built to run on a home PC (Windows Task Scheduler + Google Drive for Desktop) is
preserved as tag `local-pc-v1` and branch `local-pc`.
