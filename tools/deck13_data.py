# -*- coding: utf-8 -*-
"""Everything Session 13's deck computes before it is rendered, cached in
session_13/data/deck13.json by `python tools/deck13_data.py session_13/data`: the houses of
Session 7 and the hours of Session 10, each with the trees of Sessions 11 and 12
next to the models those sessions ended with. Imported by the deck's first cell.

The live cells in the deck fit the same models with the same settings, so a
number quoted from here and a number printed by a cell come from one model.
Every setting is chosen without the test rows:

- the houses: the tree's smallest leaf on Session 7's folds (KFold, 5, shuffled,
  seed 0); the depth and the number of boosted trees on the 2023 sales, fitted
  on 2021 and 2022, then refitted on 2021 to 2023 with that many trees; the
  forest is the lecture's forest of Session 11 (max_features='sqrt'), untuned;
- the hours: the depth and the number of boosted trees on 2023, fitted on 2022,
  then refitted on 2022 and 2023; the forest's smallest leaf on Session 10's
  folds (TimeSeriesSplit, 5).

The houses are scored as Session 7 scored them (the RMSE in kroner of the
valuations, a model fitted in logs, and the share of 2024 sales underwater at
80 percent of the valuation); the hours as Session 10 scored them (the AUC,
the Brier score, and the P&L of trading every hour at a threshold of 0.6 for
50 kroner a trade).
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from xgboost import XGBRegressor, XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import deck10_data as D10                                   # noqa: E402  (Session 10's table and models)

warnings.filterwarnings("ignore")

CATHEDRAL = (56.1567, 10.2108)              # Aarhus Domkirke, as in tools/build_session07_table.py
MAP_SOUTH, MAP_NORTH, MAP_LON = 56.124, 56.206, 10.174   # the inner postcodes, as on Session 7's map
MAP_W, MAP_H, MAP_NX, MAP_NY = 900, 440, 90, 44        # pixels, and cells of the price map
NOT_COLUMNS = ["price", "sold", "quarter", "date", "zip", "lat", "lon"]
PLACE = ["area", "rooms", "built", "centre_km", "lat", "lon", "type_villa"]   # coordinates instead of towns
DEPTHS = [2, 3, 4, 6]
RATE = 0.05
COST, THRESHOLD = 50, 0.6
EXTRA = ["price_before", "wind", "solar"]   # three more columns of Session 10's table


def rmse(actual, predicted):
    return float(np.sqrt(((np.asarray(actual) - np.asarray(predicted)) ** 2).mean()))


def underwater(actual, predicted):
    return float((0.8 * np.asarray(predicted) > np.asarray(actual)).mean())


def within(actual, predicted, share):
    a, p = np.asarray(actual), np.asarray(predicted)
    return float((np.abs(p - a) / a <= share).mean())


def haversine_km(lat, lon, lat0, lon0):
    lat, lon = np.radians(lat), np.radians(lon)
    lat0, lon0 = np.radians(lat0), np.radians(lon0)
    a = np.sin((lat - lat0) / 2) ** 2 + np.cos(lat) * np.cos(lat0) * np.sin((lon - lon0) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def mercator_y(lat):
    return np.log(np.tan(np.radians(lat) / 2 + np.pi / 4))


def _map_image(folder, bounds, scale=2):
    """Crop Session 7's basemap of the inner postcodes to the map's bounds and save
    it as map13.png at twice the map's size. The tiles are in Web Mercator, so
    pixel columns are linear in longitude and pixel rows in Mercator latitude."""
    import json
    from PIL import Image
    folder = Path(folder)
    src = json.loads((folder / "aarhus_inner.json").read_text())
    img = Image.open(folder / "aarhus_inner.png")
    w, h = img.size
    top, bot = mercator_y(src["north"]), mercator_y(src["south"])
    box = ((bounds["west"] - src["west"]) / (src["east"] - src["west"]) * w,
           (top - mercator_y(bounds["north"])) / (top - bot) * h,
           (bounds["east"] - src["west"]) / (src["east"] - src["west"]) * w,
           (top - mercator_y(bounds["south"])) / (top - bot) * h)
    assert box[0] >= 0 and box[1] >= 0 and box[2] <= w and box[3] <= h, box
    img.crop(tuple(round(b) for b in box)).convert("RGB").resize((MAP_W * scale, MAP_H * scale), Image.LANCZOS).save(folder / "map13.png")


def _stopped(make, fit_x, fit_y, stop_x, stop_y):
    """Every depth fitted on the earlier rows and stopped on the later block; the
    depth with the lowest score on that block, and its number of trees."""
    best = None
    for depth in DEPTHS:
        m = make(depth, 3000, stop=True)
        m.fit(fit_x, fit_y, eval_set=[(stop_x, stop_y)], verbose=False)
        if best is None or m.best_score < best[0]:
            best = (float(m.best_score), depth, int(m.best_iteration) + 1)
    return best[1], best[2]


def _xgb_reg(depth, n, stop=False, **kw):
    extra = {"early_stopping_rounds": 100} if stop else {}
    return XGBRegressor(n_estimators=n, learning_rate=RATE, max_depth=depth, random_state=0, **extra, **kw)


def _xgb_clf(depth, n, stop=False, **kw):
    extra = {"early_stopping_rounds": 100} if stop else {}
    return XGBClassifier(n_estimators=n, learning_rate=RATE, max_depth=depth, random_state=0, **extra, **kw)


# ============================================================ the houses
def houses(folder="data"):
    H = pd.read_csv(Path(folder) / "aarhus_houses.csv")
    W = pd.get_dummies(H, columns=["town", "type"], dtype=int)
    cols = [c for c in W.columns if c not in NOT_COLUMNS]
    tr, te = W[W["sold"] <= 2023], W[W["sold"] == 2024]
    fr, sr = tr[tr["sold"] <= 2022], tr[tr["sold"] == 2023]
    y, ly, yt = tr["price"], np.log(tr["price"]), te["price"]
    folds = KFold(n_splits=5, shuffle=True, random_state=0)
    pipe = Pipeline([("scale", StandardScaler()), ("ridge", Ridge())])
    grid = {"ridge__alpha": [0.01, 0.1, 1, 10, 100, 1000]}

    M, cols_of = {}, {}
    M["ridge"] = GridSearchCV(pipe, grid, cv=folds, scoring="neg_root_mean_squared_error").fit(tr[cols], ly)
    cols_of["ridge"] = cols
    leaf = GridSearchCV(DecisionTreeRegressor(random_state=0), {"min_samples_leaf": [5, 10, 20, 50, 100]}, cv=folds,
                        scoring="neg_root_mean_squared_error").fit(tr[cols], ly).best_params_["min_samples_leaf"]
    M["tree"] = DecisionTreeRegressor(min_samples_leaf=leaf, random_state=0).fit(tr[cols], ly)
    cols_of["tree"] = cols
    M["forest"] = RandomForestRegressor(n_estimators=100, max_features="sqrt", random_state=0, n_jobs=-1).fit(tr[cols], ly)
    cols_of["forest"] = cols
    d_towns, n_towns = _stopped(_xgb_reg, fr[cols], np.log(fr["price"]), sr[cols], np.log(sr["price"]))
    M["boosted"] = _xgb_reg(d_towns, n_towns).fit(tr[cols], ly)
    cols_of["boosted"] = cols
    d_place, n_place = _stopped(_xgb_reg, fr[PLACE], np.log(fr["price"]), sr[PLACE], np.log(sr["price"]))
    M["boosted_place"] = _xgb_reg(d_place, n_place).fit(tr[PLACE], ly)
    cols_of["boosted_place"] = PLACE
    M["ridge_place"] = GridSearchCV(pipe, grid, cv=folds, scoring="neg_root_mean_squared_error").fit(tr[PLACE], ly)
    cols_of["ridge_place"] = PLACE
    rise = {"area": 1}                       # the value may only rise with the area
    mono = _xgb_reg(d_place, 3000, stop=True, monotone_constraints=rise)
    mono.fit(fr[PLACE], np.log(fr["price"]), eval_set=[(sr[PLACE], np.log(sr["price"]))], verbose=False)
    n_mono = int(mono.best_iteration) + 1
    M["boosted_rising"] = _xgb_reg(d_place, n_mono, monotone_constraints=rise).fit(tr[PLACE], ly)
    cols_of["boosted_rising"] = PLACE

    def value(name, rows):
        return np.exp(M[name].predict(rows[cols_of[name]]))

    D = {"n_train": int(len(tr)), "n_test": int(len(te)), "n_columns": len(cols), "tree_leaf": int(leaf),
         "tree_depth": int(M["tree"].get_depth()), "tree_leaves": int(M["tree"].get_n_leaves()),
         "ridge_alpha": M["ridge"].best_params_["ridge__alpha"],
         "boosted": {"depth": d_towns, "trees": n_towns}, "boosted_place": {"depth": d_place, "trees": n_place},
         "boosted_rising": {"depth": d_place, "trees": n_mono}}
    average = np.full(len(te), y.mean())
    D["scores"] = {"average": {"rmse": rmse(yt, average), "underwater": underwater(yt, average),
                               "w10": within(yt, average, 0.1), "w20": within(yt, average, 0.2)}}
    for name in M:
        v = value(name, te)
        D["scores"][name] = {"rmse": rmse(yt, v), "underwater": underwater(yt, v),
                             "w10": within(yt, v, 0.1), "w20": within(yt, v, 0.2)}

    house = te[te["town_Aarhus V"] == 1].head(1)
    D["house"] = {"index": int(house.index[0]), "price": int(house["price"].iloc[0]),
                  "area": int(house["area"].iloc[0]), "lat": float(house["lat"].iloc[0]), "lon": float(house["lon"].iloc[0]),
                  "values": {name: round(float(value(name, house)[0])) for name in M}}

    # the house as its area grows: one curve per model
    areas = list(range(60, 305, 5))
    grown = pd.concat([house] * len(areas))
    grown["area"] = areas
    D["curve"] = {"areas": areas, "values": {name: [round(float(v)) for v in value(name, grown)]
                                             for name in ["ridge", "forest", "boosted_place", "boosted_rising"]},
                  "data_max": int(tr["area"].quantile(0.99))}
    falls = {}
    sample = te.sample(200, random_state=0)
    span = list(range(100, 255, 5))
    for name in ["ridge", "forest", "boosted_place", "boosted_rising"]:
        bad = 0
        for i in range(len(sample)):
            rows = pd.concat([sample.iloc[[i]]] * len(span))
            rows["area"] = span
            bad += int((np.diff(value(name, rows)) < -1).any())
        falls[name] = bad
    D["falls"] = falls

    # thirty more square metres on six houses of 2024
    six = te.sample(6, random_state=3)
    bigger = six.copy()
    bigger["area"] = bigger["area"] + 30
    D["thirty"] = {name: [round(float(v), 3) for v in value(name, bigger) / value(name, six) - 1]
                   for name in ["ridge", "boosted_place", "forest"]}
    D["thirty_rows"] = six[["area", "built"]].astype(int).values.tolist()

    # the house placed anywhere in the inner postcodes, on a map with true proportions
    south, north = MAP_SOUTH, MAP_NORTH
    ytop, ybot = mercator_y(north), mercator_y(south)
    span = np.degrees((MAP_W / MAP_H) * (ytop - ybot))
    west, east = MAP_LON - span / 2, MAP_LON + span / 2
    lons = np.linspace(west, east, MAP_NX + 1)
    lats = np.linspace(south, north, MAP_NY + 1)
    clon, clat = (lons[:-1] + lons[1:]) / 2, (lats[:-1] + lats[1:]) / 2
    glon, glat = np.meshgrid(clon, clat)
    glon, glat = glon.ravel(), glat.ravel()
    near = np.zeros(len(glon), dtype=int)
    for k in range(0, len(glon), 400):
        d = haversine_km(glat[k:k + 400, None], glon[k:k + 400, None], H["lat"].values[None, :], H["lon"].values[None, :])
        near[k:k + 400] = (d <= 0.4).sum(axis=1)
    keep = near >= 3
    spot = pd.concat([house] * int(keep.sum()))
    spot["lat"], spot["lon"] = glat[keep], glon[keep]
    spot["centre_km"] = np.round(haversine_km(glat[keep], glon[keep], *CATHEDRAL), 1)

    def px(lon):
        return (lon - west) / (east - west) * MAP_W

    def py(lat):
        return (ytop - mercator_y(lat)) / (ytop - ybot) * MAP_H

    ix = np.tile(np.arange(MAP_NX), MAP_NY)[keep]
    iy = np.repeat(np.arange(MAP_NY), MAP_NX)[keep]
    cells = []
    for j in range(int(keep.sum())):
        x0, x1 = px(lons[ix[j]]), px(lons[ix[j] + 1])
        y0, y1 = py(lats[iy[j] + 1]), py(lats[iy[j]])
        cells.append([round(x0, 1), round(y0, 1), round(x1 - x0 + 0.6, 1), round(y1 - y0 + 0.6, 1)])
    D["map"] = {"width": MAP_W, "height": MAP_H, "cells": cells,
                "values": {name: [round(float(v) / 1e6, 2) for v in value(name, spot)] for name in ["ridge_place", "boosted_place"]},
                "house": [round(px(D["house"]["lon"]), 1), round(py(D["house"]["lat"]), 1)],
                "bounds": {"south": south, "west": float(west), "north": north, "east": float(east)}}
    _map_image(folder, D["map"]["bounds"])

    # one core: the seconds to fit and to value the 2024 sales
    D["timing"] = {}
    for name, make, cs in [("ridge", lambda: Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=D["ridge_alpha"]))]), cols),
                           ("tree", lambda: DecisionTreeRegressor(min_samples_leaf=leaf, random_state=0), cols),
                           ("forest", lambda: RandomForestRegressor(n_estimators=100, max_features="sqrt", random_state=0, n_jobs=1), cols),
                           ("boosted", lambda: _xgb_reg(d_towns, n_towns, n_jobs=1), cols),
                           ("boosted_place", lambda: _xgb_reg(d_place, n_place, n_jobs=1), PLACE)]:
        m = make()
        t0 = time.perf_counter()
        m.fit(tr[cs], ly)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        m.predict(te[cs])
        D["timing"][name] = {"fit": fit_s, "predict": time.perf_counter() - t0}
    D["_models"], D["_cols"] = M, cols_of
    D["_frames"] = {"H": H, "W": W, "train": tr, "test": te, "house": house}
    return D


# ============================================================ the hours
def hours(folder="data"):
    P = D10.prepared(folder)
    tr, te = P.loc[:"2023"], P.loc["2024"]
    fr, sr = tr.loc[:"2022"], tr.loc["2023"]
    y, yt = tr["up"], te["up"]
    folds = TimeSeriesSplit(n_splits=5)
    near = D10.NEAR
    rich = near + EXTRA
    move = (te["price"] - te["price_before"]).to_numpy()

    def pnl(p, threshold=THRESHOLD, cost=COST):
        long = p >= threshold
        short = p <= 1 - threshold
        return int(round(move[long].sum() - move[short].sum() - cost * (long.sum() + short.sum())))

    M, cols_of = {}, {}
    M["logistic"], cols_of["logistic"] = D10.pipe().fit(tr[D10.FEATURES], y), D10.FEATURES
    M["knn"], cols_of["knn"] = D10.near_pipe().fit(tr[near], y), near
    M["logistic_more"], cols_of["logistic_more"] = D10.pipe().fit(tr[D10.FEATURES + EXTRA], y), D10.FEATURES + EXTRA
    M["knn_more"], cols_of["knn_more"] = D10.near_pipe().fit(tr[rich], y), rich
    d_near, n_near = _stopped(_xgb_clf, fr[near], fr["up"], sr[near], sr["up"])
    M["boosted"], cols_of["boosted"] = _xgb_clf(d_near, n_near).fit(tr[near], y), near
    d_rich, n_rich = _stopped(_xgb_clf, fr[rich], fr["up"], sr[rich], sr["up"])
    M["boosted_more"], cols_of["boosted_more"] = _xgb_clf(d_rich, n_rich).fit(tr[rich], y), rich
    leaves = {}
    for cols, key in [(near, "forest"), (rich, "forest_more")]:
        best = None
        for leaf in [5, 20, 50, 100]:
            cv = cross_val_score(RandomForestClassifier(n_estimators=100, min_samples_leaf=leaf, max_features="sqrt",
                                                        random_state=0, n_jobs=-1), tr[cols], y, cv=folds, scoring="roc_auc").mean()
            if best is None or cv > best[0]:
                best = (cv, leaf)
        leaves[key] = best[1]
        M[key] = RandomForestClassifier(n_estimators=100, min_samples_leaf=best[1], max_features="sqrt", random_state=0,
                                        n_jobs=-1).fit(tr[cols], y)
        cols_of[key] = cols

    def prob(name, rows):
        return M[name].predict_proba(rows[cols_of[name]])[:, 1]

    D = {"n_train": int(len(tr)), "n_test": int(len(te)), "boosted": {"depth": d_near, "trees": n_near},
         "boosted_more": {"depth": d_rich, "trees": n_rich}, "leaves": leaves, "scores": {}}
    for name in M:
        p = prob(name, te)
        D["scores"][name] = {"auc": float(roc_auc_score(yt, p)), "brier": float(brier_score_loss(yt, p)), "pnl": pnl(p)}
    for name, model in [("logistic", D10.pipe()), ("knn", D10.near_pipe()), ("logistic_more", D10.pipe()), ("knn_more", D10.near_pipe()),
                        ("boosted", _xgb_clf(d_near, n_near)), ("boosted_more", _xgb_clf(d_rich, n_rich)),
                        ("forest", RandomForestClassifier(n_estimators=100, min_samples_leaf=leaves["forest"], max_features="sqrt",
                                                          random_state=0, n_jobs=-1)),
                        ("forest_more", RandomForestClassifier(n_estimators=100, min_samples_leaf=leaves["forest_more"],
                                                               max_features="sqrt", random_state=0, n_jobs=-1))]:
        D["scores"][name]["folds"] = float(cross_val_score(model, tr[cols_of[name]], y, cv=folds, scoring="roc_auc").mean())
    D["rule_pnl"] = pnl((te["wind_change"] < 0).astype(float).to_numpy(), 0.5)

    # what happened, and what two models expected, by the wind change at three price levels (2024)
    edges = list(tr["price_before"].quantile([1 / 3, 2 / 3]).round(0).astype(int))
    level = pd.cut(te["price_before"], [-np.inf] + edges + [np.inf], labels=["low", "middle", "high"])
    bands = pd.cut(te["wind_change"], [-np.inf, -1000, -300, 300, 1000, np.inf])
    frame = pd.DataFrame({"level": level, "band": bands, "up": yt, "logistic_more": prob("logistic_more", te),
                          "boosted_more": prob("boosted_more", te)})
    table = frame.groupby(["level", "band"], observed=False).agg(up=("up", "mean"), n=("up", "size"),
                                                                logistic_more=("logistic_more", "mean"),
                                                                boosted_more=("boosted_more", "mean"))
    D["levels"] = {"edges": [int(e) for e in edges], "bands": ["below −1,000", "−1,000 to −300", "−300 to +300",
                                                               "+300 to +1,000", "above +1,000"],
                   "rows": {lv: {k: [None if pd.isna(v) else round(float(v), 3) for v in table.loc[lv][k]]
                                 for k in ["up", "logistic_more", "boosted_more"]} | {"n": [int(v) for v in table.loc[lv]["n"]]}
                            for lv in ["low", "middle", "high"]}}
    trframe = pd.DataFrame({"level": pd.cut(tr["price_before"], [-np.inf] + edges + [np.inf], labels=["low", "middle", "high"]),
                            "band": pd.cut(tr["wind_change"], [-np.inf, -1000, -300, 300, 1000, np.inf]), "up": y})
    D["levels_train"] = trframe.groupby(["level", "band"], observed=False)["up"].mean().unstack().round(3).values.tolist()

    # the eight hours of Session 10's forecast, with the new models' answers
    rounds = []
    for stamp in D10.ROUNDS:
        ts = pd.Timestamp(stamp)
        row = te.loc[[ts]]
        rounds.append({"short": D10._date_words(ts, short=True), "hour": int(ts.hour), "truth": int(row["up"].iloc[0]),
                       "price_before": round(float(row["price_before"].iloc[0])), "price": round(float(row["price"].iloc[0])),
                       "logit": round(float(prob("logistic", row)[0]), 3), "knn": round(float(prob("knn", row)[0]), 3),
                       "forest": round(float(prob("forest_more", row)[0]), 3),
                       "boosted": round(float(prob("boosted_more", row)[0]), 3)})
    D["game"] = {"rounds": rounds}
    truth = np.array([r["truth"] for r in rounds])
    D["game_scores"] = {}
    for key in ["logit", "knn", "forest", "boosted"]:
        q = np.array([r[key] for r in rounds])
        D["game_scores"][key] = {"brier": float(np.mean((q - truth) ** 2)),
                                 "logloss": float(log_loss(truth, np.clip(q, 0.01, 0.99), labels=[0, 1]))}

    # one core: the seconds to fit and to score the 8,663 hours of 2024
    D["timing"] = {}
    for name, make, cs in [("logistic", D10.pipe, D10.FEATURES), ("knn", D10.near_pipe, near),
                           ("forest_more", lambda: RandomForestClassifier(n_estimators=100, min_samples_leaf=leaves["forest_more"],
                                                                         max_features="sqrt", random_state=0, n_jobs=1), rich),
                           ("boosted_more", lambda: _xgb_clf(d_rich, n_rich, n_jobs=1), rich)]:
        m = make()
        t0 = time.perf_counter()
        m.fit(tr[cs], y)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        m.predict_proba(te[cs])
        D["timing"][name] = {"fit": fit_s, "predict": time.perf_counter() - t0}
    D["_models"], D["_cols"] = M, cols_of
    D["_frames"] = {"train": tr, "test": te}
    return D


def build(folder="data"):
    return {"houses": houses(folder), "hours": hours(folder)}


def public(D):
    """Everything the deck draws, without the fitted models and tables."""
    return {part: {k: v for k, v in D[part].items() if not k.startswith("_")} for part in D}


def save(folder="data"):
    """Fit everything and write it to <folder>/deck13.json, with map13.png beside it.
    Run it with the course's Python (scikit-learn 1.5, XGBoost 2.1, the versions the
    live cells were checked against); the deck's render only reads the file."""
    import json
    D = public(build(folder))
    D["versions"] = {"python": sys.version.split()[0], "sklearn": __import__("sklearn").__version__,
                     "xgboost": __import__("xgboost").__version__}
    (Path(folder) / "deck13.json").write_text(json.dumps(D, separators=(",", ":")), encoding="utf-8")
    return D


if __name__ == "__main__":
    out = save(sys.argv[1] if len(sys.argv) > 1 else "data")
    print("wrote deck13.json:", out["versions"])
