# -*- coding: utf-8 -*-
"""One-time builder for the power table Session 10 works on. NOT course material.

Downloads hourly day-ahead prices for West Denmark (price area DK1) and the
wind and solar forecasts published the day before, from Energinet's Energi Data
Service, and writes one CSV. It also writes a small JSON for the lecture's map:
the outlines of Denmark and its neighbours, and two days of prices and
cross-border flows. Students never run this script: the outputs are committed.

Source     : Energi Data Service, Energinet (https://www.energidataservice.dk/)
             datasets Elspotprices, Forecasts_Hour, ProductionConsumptionSettlement.
             Open data, no key. Country outlines from Natural Earth (public domain).
Fetched    : 2026-09-24
Coverage   : 2022-01-01 to 2024-12-31, hourly, Danish local time

The table, one row per hour:

    time     the hour, Danish local time (the clock hour the price applies to)
    date     the day, as text (2024-10-04), for grouping the hours of one day
    hour     the clock hour, 0 to 23
    weekday  the day of the week, 0 for Monday to 6 for Sunday
    weekend  1 on a Saturday or a Sunday, else 0
    price    the day-ahead price for DK1, in kroner per MWh
    wind     the day-ahead forecast of onshore plus offshore wind output, MW
    solar    the day-ahead forecast of solar output, MW

Timing, from the dataset's own metadata: the day-ahead forecast "for the next day
is published at 18:00 Danish time zone", six hours AFTER the day-ahead auction
closes at noon. A bidder at noon has its own weather-based forecast, a few hours
older, so this column stands in for it and the deck's scores are, if anything,
slightly optimistic. The deck says so. The repeated hour when the clocks go back is averaged,
and the missing hour when they go forward is simply absent. Seven days on which
Energinet published no day-ahead forecast (2022-11-20, 2023-04-11, 2024-04-13
to 2024-04-15, 2024-06-01, 2024-11-17) are left out.

    python tools/build_session10_table.py

Outputs:
    session_10/data/power.csv
    data/power.csv                 the copy the notebooks' Colab fallback reads
    session_10/data/map.json       read by the deck when it is rendered
"""
import json
import math
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.energidataservice.dk/dataset/"
START, END = "2022-01-01T00:00", "2025-01-01T00:00"
CACHE = Path.home() / ".cache" / "mlfin_session10"
CACHE.mkdir(parents=True, exist_ok=True)
NATURAL_EARTH = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
                 "master/geojson/ne_50m_admin_0_countries.geojson")


def fetch_json(url, name):
    path = CACHE / name
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    print("  fetching", name)
    for attempt in range(8):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "mlfin-course-build/1.0"})
            with urllib.request.urlopen(req, timeout=300) as r:
                data = json.load(r)
            path.write_text(json.dumps(data), encoding="utf-8")
            return data
        except Exception as e:                       # the service rate-limits with HTTP 429
            wait = 20 * (attempt + 1)
            print(f"    {type(e).__name__}; retrying in {wait} s")
            time.sleep(wait)
    raise RuntimeError("could not fetch " + url)


def dataset(name, areas):
    flt = urllib.request.quote(json.dumps({"PriceArea": areas}, separators=(",", ":")))
    url = f"{API}{name}?start={START}&end={END}&filter={flt}&limit=0"
    return pd.DataFrame(fetch_json(url, f"{name}_{'_'.join(areas)}.json")["records"])


# ------------------------------------------------------------- the table
price = dataset("Elspotprices", ["DK1", "DK2"])
price["time"] = pd.to_datetime(price["HourDK"])
forecast = dataset("Forecasts_Hour", ["DK1"])
forecast["time"] = pd.to_datetime(forecast["HourDK"])

dk1 = (price[price["PriceArea"] == "DK1"].groupby("time")["SpotPriceDKK"].mean()
       .rename("price"))
fc = forecast.pivot_table(index="time", columns="ForecastType",
                          values="ForecastDayAhead", aggfunc="mean")
table = pd.concat([dk1, fc["Offshore Wind"] + fc["Onshore Wind"], fc["Solar"]],
                  axis=1, keys=["price", "wind", "solar"]).dropna()
table = table.sort_index()
table["price"] = table["price"].round(2)
table["wind"] = table["wind"].round(1)
table["solar"] = table["solar"].round(1)
if not 25_000 < len(table) < 27_000:
    raise RuntimeError(f"expected about 26,000 hours, got {len(table)}")
table.insert(0, "date", table.index.strftime("%Y-%m-%d"))
table.insert(1, "hour", table.index.hour)
table.insert(2, "weekday", table.index.weekday)
table.insert(3, "weekend", (table.index.weekday >= 5).astype(int))

for out in [ROOT / "session_10" / "data" / "power.csv", ROOT / "data" / "power.csv"]:
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index_label="time", date_format="%Y-%m-%d %H:%M")
    print("wrote", out.relative_to(ROOT), table.shape, f"{out.stat().st_size / 1e6:.2f} MB")

# ------------------------------------------------------------- the map
flows = dataset("ProductionConsumptionSettlement", ["DK1", "DK2"])
flows["time"] = pd.to_datetime(flows["HourDK"])
exch = [c for c in flows.columns if c.startswith("Exchange") and c.endswith("_MWh")]

# Sign convention, checked rather than assumed: West Denmark exports when it is
# windy, so the sum of its exchanges should fall as the wind rises if a positive
# value means power flowing IN.
d1 = flows[flows["PriceArea"] == "DK1"].set_index("time")
net = d1[[c for c in exch if c in d1.columns]].sum(axis=1).groupby(level=0).mean()   # the repeated DST hour
corr = float(pd.concat([net, table["wind"]], axis=1).dropna().corr().iloc[0, 1])
positive_is_import = corr < 0
print(f"correlation of DK1 net exchange with the wind forecast: {corr:+.3f} "
      f"-> positive means {'import' if positive_is_import else 'export'}")

# the windiest and the stillest weekday of the 2024 heating season
h24 = table.loc["2024"]
daily = h24.groupby(h24.index.date).agg(wind=("wind", "mean"), price=("price", "mean"))
daily.index = pd.to_datetime(daily.index)
season = daily[(daily.index.month.isin([1, 2, 3, 10, 11, 12])) & (daily.index.weekday < 5)]
windy_day, still_day = season["wind"].idxmax(), season["wind"].idxmin()
print(f"windy day {windy_day.date()} (mean forecast {season.loc[windy_day, 'wind']:,.0f} MW, "
      f"average price {season.loc[windy_day, 'price']:,.0f}); still day {still_day.date()} "
      f"({season.loc[still_day, 'wind']:,.0f} MW, {season.loc[still_day, 'price']:,.0f})")

BORDERS = {   # column, the area it belongs to, a schematic cable from Denmark outwards, where its MW label goes
    "NO": ("ExchangeNO_MWh", "DK1", (9.3, 57.05), (8.05, 58.1), (9.15, 57.72)),
    "SE1": ("ExchangeSE_MWh", "DK1", (10.45, 57.25), (11.85, 57.72), (11.3, 57.3)),
    "DE1": ("ExchangeGE_MWh", "DK1", (9.3, 55.0), (9.65, 54.3), (8.85, 54.62)),
    "NL": ("ExchangeNL_MWh", "DK1", (8.35, 55.45), (6.85, 53.45), (7.05, 54.5)),
    "GB": ("ExchangeGB_MWh", "DK1", (8.15, 55.9), (4.75, 54.62), (6.2, 55.55)),   # Viking Link; Britain is off the map
    "SE2": ("ExchangeSE_MWh", "DK2", (12.62, 55.78), (13.3, 55.62), (13.1, 55.95)),
    "DE2": ("ExchangeGE_MWh", "DK2", (11.95, 54.72), (12.12, 54.18), (12.62, 54.45)),
    "BELT": ("ExchangeGreatBelt_MWh", "DK1", (10.55, 55.35), (11.2, 55.45), (10.9, 55.18)),
}


def day_block(day):
    hours = pd.date_range(day, periods=24, freq="h")
    out = {"date": day.strftime("%Y-%m-%d"), "weekday": day.strftime("%A"),
           "dk1": [], "dk2": [], "wind": [], "flows": {k: [] for k in BORDERS}}
    p = price.groupby(["PriceArea", "time"])["SpotPriceDKK"].mean()      # unique keys, sorted
    for h in hours:
        out["dk1"].append(round(float(p.get(("DK1", h), np.nan)), 1))
        out["dk2"].append(round(float(p.get(("DK2", h), np.nan)), 1))
        out["wind"].append(round(float(table["wind"].get(h, np.nan)), 1))
        for key, (col, area, _, _, _) in BORDERS.items():
            rows = flows[(flows["PriceArea"] == area) & (flows["time"] == h)]
            v = float(rows[col].iloc[0]) if len(rows) and col in rows else 0.0
            v = -v if positive_is_import else v          # stored as EXPORT from Denmark: positive = outwards
            out["flows"][key].append(round(v, 1))
    return out


# outlines: Denmark split at the Great Belt into the two price areas
LON0, LON1, LAT0, LAT1 = 4.5, 15.5, 53.0, 58.6
KX = 600 / ((LON1 - LON0) * math.cos(math.radians(56.0)))
W_SVG, H_SVG = 600, round((LAT1 - LAT0) * KX)


def project(lon, lat):
    x = (lon - LON0) * math.cos(math.radians(56.0)) * KX
    y = (LAT1 - lat) * KX
    return round(min(max(x, -2), W_SVG + 2), 1), round(min(max(y, -2), H_SVG + 2), 1)


def ring_path(ring):
    pts, last = [], None
    for lon, lat in ring:
        p = project(lon, lat)
        if p != last:
            pts.append(p)
            last = p
    if len(pts) < 3:
        return ""
    return "M" + " L".join(f"{x},{y}" for x, y in pts) + " Z"


ne = fetch_json(NATURAL_EARTH, "ne_50m_admin_0_countries.geojson")
shapes = {"DK1": [], "DK2": []}
for f in ne["features"]:
    name = f["properties"]["NAME"]
    if name not in {"Denmark", "Norway", "Sweden", "Germany", "Netherlands", "United Kingdom"}:
        continue
    geom = f["geometry"]
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    for poly in polys:
        ring = poly[0]
        lons = [c[0] for c in ring]
        lats = [c[1] for c in ring]
        if max(lons) < LON0 or min(lons) > LON1 or max(lats) < LAT0 or min(lats) > LAT1:
            continue
        path = ring_path(ring)
        if not path:
            continue
        if name == "Denmark":
            shapes["DK1" if np.mean(lons) < 11.0 else "DK2"].append(path)
        else:
            shapes.setdefault(name, []).append(path)

cables = {k: {"from": project(*a), "to": project(*b), "tag": project(*tag), "area": area}
          for k, (_, area, a, b, tag) in BORDERS.items()}
labels = {"Norway": project(7.3, 58.42), "Sweden": project(13.9, 57.35), "Germany": project(10.4, 53.55),
          "Netherlands": project(6.3, 53.12), "to Britain": project(5.25, 54.38),
          "West Denmark": project(9.05, 56.3), "East Denmark": project(11.8, 55.52)}

# three wind sites in West Denmark: Horns Rev offshore, onshore in Thy, Anholt offshore
turbines = [project(7.9, 55.52), project(8.55, 56.98), project(11.2, 56.6)]

MAP = {"width": W_SVG, "height": H_SVG, "shapes": {k: " ".join(v) for k, v in shapes.items()},
       "cables": cables, "labels": labels, "turbines": turbines,
       "wind_max": int(table["wind"].max()),
       "days": [day_block(windy_day), day_block(still_day)],
       "note": "flows are EXPORTS from Denmark in MWh, positive outwards"}
out = ROOT / "session_10" / "data" / "map.json"
out.write_text(json.dumps(MAP, separators=(",", ":")), encoding="utf-8")
print("wrote", out.relative_to(ROOT), f"{out.stat().st_size / 1e3:.0f} kB,",
      "shapes:", {k: v.count("M") for k, v in MAP["shapes"].items()})
for blk in MAP["days"]:
    tot = {k: round(sum(v) / 24) for k, v in blk["flows"].items()}
    print(f"  {blk['date']} {blk['weekday']}: DK1 avg {np.nanmean(blk['dk1']):,.0f}, "
          f"wind avg {np.nanmean(blk['wind']):,.0f} MW, average export per border {tot}")
