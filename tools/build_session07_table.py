# -*- coding: utf-8 -*-
"""One-time builder for the Aarhus house-sales table Session 7 works on. NOT
course material, and not part of release.py: it downloads, so run it by hand.

Source     : "Danish Residential Housing Prices 1992-2024", compiled by Martin
             Frederiksen from boliga.dk (sale prices are public records from
             the Danish land registry, tinglysning), Kaggle, updated 2024-11-29.
             https://www.kaggle.com/datasets/martinfrederiksen/danish-residential-housing-prices-1992-2024
             Raw files and scraper: https://github.com/MartinSamFred/Danish-residential-housingPrices-1992-2024
Terms      : the compiler states the data is "intended to be used primarily
             for educational purposes only" and that he owns none of it.
             This script keeps a small teaching extract: no street addresses,
             coordinates rounded to about 100 metres.
Geocoding  : Danmarks Adressers Web API (DAWA, api.dataforsyningen.dk), the
             public address register. Free, no key.
Basemap    : Esri "World Light Gray" canvas tiles (c) Esri, HERE,
             Garmin, OpenStreetMap contributors and the GIS user community,
             fetched once for the map figures in the deck.
Fetched    : 2026-09-12

What the script keeps, out of 1.5 million Danish sales:

    price      sale price in Danish kroner                          (target)
    area       living area, square metres
    rooms      number of rooms
    built      year the house was built
    type       villa or townhouse
    town       the postal town, one of 22 in and around Aarhus
    zip        the postcode
    centre_km  distance from Aarhus Cathedral, kilometres (from the coordinates)
    sold       the year of the sale, 2021 to 2024
    quarter    the quarter of the sale, 2021Q1 to 2024Q3
    date       the date of the sale
    lat, lon   the location, rounded to three decimals (about 100 metres)

Rows: ordinary sales ("regular_sale": no family transfers, auctions or other
special sales) of villas and townhouses in the Aarhus postcodes, January 2021
to September 2024 (the source stops in October 2024, so the last, partial
quarter is left out), sorted by date so the 2024 sales sit at the bottom,
ready to be the test rows. The result is about 7,600 rows.

Outputs:
    session_07/data/aarhus_houses.csv     the table (also copied to data/)
    session_07/data/aarhus_basemap.png    light basemap for the map figures
    session_07/data/aarhus_inner.png      the same for the inner postcodes, crisper
    session_07/data/*.json                their bounds
    <scratch>/aarhus_addresses_private.csv  addresses + exact coordinates,
                                            written OUTSIDE the repository
                                            for choosing the example house

    python tools/build_session07_table.py
"""
import io
import json
import math
import re
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "session_07" / "data"
OUTS = [OUT_DIR / "aarhus_houses.csv", ROOT / "data" / "aarhus_houses.csv"]
PRIVATE = Path(tempfile.gettempdir()) / "aarhus_addresses_private.csv"

KAGGLE_ZIP = ("https://www.kaggle.com/api/v1/datasets/download/"
              "martinfrederiksen/danish-residential-housing-prices-1992-2024")
DAWA = "https://api.dataforsyningen.dk/adgangsadresser"
# the label-free grey base only: the deck writes the town names itself, from
# the data, in its own font
TILES = ["https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"]
UA = {"User-Agent": "mlfin-2026 course build (jobo@econ.au.dk)"}

AARHUS_ZIPS = [8000, 8200, 8210, 8220, 8230, 8240, 8250, 8260, 8270, 8310, 8320,
               8330, 8340, 8355, 8361, 8380, 8381, 8462, 8471, 8520, 8530, 8541]
YEARS = (2021, 2024)
LAST_DAY = "2024-09-30"                  # the source stops in October 2024
CATHEDRAL = (56.1567, 10.2108)           # Aarhus Domkirke, lat, lon
BASEMAPS = {                              # name: (south, west, north, east, zoom)
    "aarhus_basemap": (56.03, 9.98, 56.32, 10.36, 12),   # the whole table
    "aarhus_inner": (56.11, 10.05, 56.24, 10.31, 13),    # the inner postcodes, crisper
}


def fetch(url, timeout=120):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def repair(text):
    """Undo the UTF-8-read-as-Latin-1 damage some rows carry: the two-character
    sequences Ã© for é, Ã for Å and so on, inside otherwise correct text."""
    if not isinstance(text, str) or "Ã" not in text:
        return text

    def fix(m):
        try:
            return m.group(0).encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return m.group(0)
    return re.sub(r"Ã.", fix, text)


def variants(address):
    """The address as given, without a floor suffix, and with a capital Ø
    restored where the scrape turned it into A (Astermarksvej for
    Østermarksvej), one word at a time."""
    base = re.sub(r",\s*(st|kl|[0-9]+)\.?\s*(th|tv|mf)?\.?\s*$", "", address).strip()
    out = [address] if base == address else [address, base]
    words = base.split(" ")
    for i, w in enumerate(words):
        if w.startswith("A") and len(w) > 2:
            out.append(" ".join(words[:i] + ["Ø" + w[1:]] + words[i + 1:]))
    return out


def load_sales():
    print("downloading the Kaggle archive (38 MB) ...")
    raw = fetch(KAGGLE_ZIP)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        with z.open("DKHousingPrices.parquet") as f:
            df = pd.read_parquet(io.BytesIO(f.read()))
    print("  ", len(df), "sales in the archive")
    return df


def select(df):
    d = df[df["zip_code"].isin(AARHUS_ZIPS)]
    d = d[d["house_type"].isin(["Villa", "Townhouse"])]
    d = d[d["sales_type"] == "regular_sale"]
    d = d[(d["date"].dt.year >= YEARS[0]) & (d["date"].dt.year <= YEARS[1])]
    d = d[d["date"] <= LAST_DAY]
    d = d.copy()
    d["address"] = d["address"].map(repair)
    d["city"] = d["city"].map(repair).str.replace("\x85", "", regex=False)
    d["city"] = d["city"].replace({"Ãbyhøj": "Åbyhøj"})
    print("  ", len(d), "ordinary house sales in the Aarhus postcodes,", YEARS)
    return d


def geocode_one(args):
    address, zipcode = args
    for candidate in variants(address):
        q = urllib.parse.urlencode({"q": f"{candidate}, {zipcode}", "struktur": "mini", "per_side": 1})
        for attempt in range(3):
            try:
                hits = json.loads(fetch(f"{DAWA}?{q}", timeout=30))
                break
            except Exception:
                time.sleep(1.5 * (attempt + 1))
                hits = []
        if hits and str(hits[0]["postnr"]) == str(zipcode):
            return float(hits[0]["y"]), float(hits[0]["x"])
    return (np.nan, np.nan)


def geocode(d):
    print("geocoding", len(d), "addresses with DAWA ...")
    pairs = list(zip(d["address"], d["zip_code"]))
    with ThreadPoolExecutor(max_workers=4) as pool:
        coords = list(pool.map(geocode_one, pairs))
    lat = np.array([c[0] for c in coords])
    lon = np.array([c[1] for c in coords])
    print("  ", int(np.isnan(lat).sum()), "addresses not found")
    return lat, lon


def haversine_km(lat, lon, lat0, lon0):
    p1, p2 = np.radians(lat), np.radians(lat0)
    dphi = np.radians(lat0 - lat)
    dlmb = np.radians(lon0 - lon)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def build(d, lat, lon):
    table = pd.DataFrame({
        "price": d["purchase_price"].astype(int).values,
        "area": d["sqm"].round().astype(int).values,
        "rooms": d["no_rooms"].astype(int).values,
        "built": d["year_build"].astype(int).values,
        "type": d["house_type"].astype(str).str.lower().values,
        "town": d["city"].values,
        "zip": d["zip_code"].astype(int).values,
        "centre_km": haversine_km(lat, lon, *CATHEDRAL).round(1),
        "sold": d["date"].dt.year.values,
        "quarter": (d["date"].dt.year.astype(str) + "Q" + d["date"].dt.quarter.astype(str)).values,
        "date": d["date"].dt.strftime("%Y-%m-%d").values,
        "lat": np.round(lat, 3),
        "lon": np.round(lon, 3),
    })
    private = pd.DataFrame({"address": d["address"].values, "zip": d["zip_code"].values,
                            "town": d["city"].values, "lat": lat, "lon": lon,
                            "price": table["price"].values, "date": table["date"].values})
    keep = ~np.isnan(lat)
    table, private = table[keep], private[keep]
    order = np.argsort(table["date"].values, kind="stable")
    table = table.iloc[order].reset_index(drop=True)
    private = private.iloc[order].reset_index(drop=True)
    if table.isna().any().any():
        raise RuntimeError("the table still has missing values")
    return table, private


def tile_xy(lat, lon, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n
    return x, y


def basemap(name, s, w, n, e, ZOOM):
    """Stitch the tiles covering the box and record the exact bounds of the image."""
    from PIL import Image
    x0, y1 = tile_xy(s, w, ZOOM)
    x1, y0 = tile_xy(n, e, ZOOM)
    xs = range(int(x0), int(x1) + 1)
    ys = range(int(y0), int(y1) + 1)
    print("fetching", len(xs) * len(ys), "basemap tiles ...")
    img = Image.new("RGBA", (256 * len(xs), 256 * len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            tile = None
            for url in TILES:              # the grey base, then the labels on top
                layer = Image.open(io.BytesIO(fetch(url.format(z=ZOOM, x=x, y=y)))).convert("RGBA")
                tile = layer if tile is None else Image.alpha_composite(tile, layer)
            img.paste(tile, (256 * i, 256 * j))
            time.sleep(0.05)
    img = img.convert("RGB")
    # bounds of the stitched image, from tile corners back to degrees
    nn = 2 ** ZOOM
    west = xs[0] / nn * 360.0 - 180.0
    east = (xs[-1] + 1) / nn * 360.0 - 180.0
    north = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * ys[0] / nn))))
    south = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (ys[-1] + 1) / nn))))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img.save(OUT_DIR / f"{name}.png")
    (OUT_DIR / f"{name}.json").write_text(json.dumps(
        {"west": west, "east": east, "south": south, "north": north, "zoom": ZOOM,
         "attribution": "(c) Esri, HERE, Garmin, OpenStreetMap contributors"}, indent=2))
    print("   wrote", (OUT_DIR / f"{name}.png").relative_to(ROOT), img.size)


def main():
    if "--basemap-only" in sys.argv:
        for name, box in BASEMAPS.items():
            basemap(name, *box)
        return
    df = load_sales()
    d = select(df)
    lat, lon = geocode(d)
    table, private = build(d, lat, lon)
    for out in OUTS:
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out, index=False)
        print("wrote", out.relative_to(ROOT), table.shape)
    private.to_csv(PRIVATE, index=False)
    print("wrote", PRIVATE, "(addresses and exact coordinates; not in the repository)")
    if "--no-basemap" not in sys.argv:
        for name, box in BASEMAPS.items():
            basemap(name, *box)


if __name__ == "__main__":
    main()
