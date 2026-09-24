# -*- coding: utf-8 -*-
"""Everything Session 10's deck computes before it is rendered: the models the
widgets draw, the P&L table, the calibration buckets, the neighbour pool, the
forecast hours and the explorers' data. Imported by the deck's first cell.

The live cells in the deck build the same columns in the same order and fit
the same models with the same defaults (StandardScaler, LogisticRegression(),
KNeighborsClassifier(n_neighbors=201)), so a number quoted from here and a
number printed by a cell come from one model.

The label: an hour is `up` when its price is higher than the price at the same
hour the day before. "The day before" is the previous row of the same hour,
which is the previous calendar day except after the seven days with no data.

The feature lab's scores take a few minutes to compute, so they are cached in
data/lab.json; rebuild it with

    python tools/deck10_data.py session_10/data --lab
"""
import json
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (roc_auc_score, confusion_matrix, brier_score_loss, log_loss,
                             accuracy_score)

warnings.filterwarnings("ignore")

DAYS = [f"weekday_{d}" for d in range(7)]
FEATURES = ["wind_change"] + DAYS + ["change_before"]       # the logistic regression
NEAR = ["wind_change", "change_before", "weekday", "hour"]   # what k-NN measures distance on
LEVELS = ["wind", "solar"] + DAYS                             # the crisis models
K = 201
COST = 50            # kroner per MWh traded, the P&L's default
BIG = 150            # a big move, kroner either way
BAND = 50            # "about the same", kroner either way
WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# The eight hours students forecast at the start, chosen by hand from 2024:
# four that went up and four that did not, one of each weekday (Thursday twice),
# eight different hours and months, a large rise and a large fall in the wind
# forecast, a jump that fell back and a fall that recovered, one hour where
# both models are confidently wrong, and one where they disagree. In date order.
ROUNDS = ["2024-01-02 13:00", "2024-04-24 19:00", "2024-05-23 12:00", "2024-07-12 20:00",
          "2024-08-31 10:00", "2024-09-15 09:00", "2024-10-10 18:00", "2024-12-02 17:00"]

# The feature lab: nine columns to switch on and off, raw ones and built ones.
LAB = [("tomorrow's wind", ["wind"], "raw", "power['wind']"),
       ("today's price", ["price_before"], "raw", "power.groupby('hour')['price'].shift(1)"),
       ("weekday", ["weekday"], "raw", "power['weekday']"),
       ("hour", ["hour"], "raw", "power['hour']"),
       ("wind change", ["wind_change"], "built", "power['wind'] - power.groupby('hour')['wind'].shift(1)"),
       ("today's move", ["change_before"], "built", "power['price_before'] - power.groupby('hour')['price'].shift(2)"),
       ("sun change", ["solar_change"], "built", "power['solar'] - power.groupby('hour')['solar'].shift(1)"),
       ("weekday as 7 columns", DAYS, "built", "pd.get_dummies(power['weekday'], prefix='weekday', dtype=int)"),
       ("size of the wind change", ["size_wind_change"], "built", "power['wind_change'].abs()")]


def pipe(**kw):
    return Pipeline([("scale", StandardScaler()), ("logit", LogisticRegression(**kw))])


def near_pipe(k=K):
    return Pipeline([("scale", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=k))])


def read(folder="data"):
    return pd.read_csv(Path(folder) / "power.csv", parse_dates=["time"]).set_index("time")


def load(folder="data"):
    """The table as the deck's cells build it, in the same order, before the split."""
    power = read(folder)
    power["price_before"] = power.groupby("hour")["price"].shift(1)
    power["up"] = (power["price"] > power["price_before"]).astype(int)
    power["wind_change"] = power["wind"] - power.groupby("hour")["wind"].shift(1)
    dummies = pd.get_dummies(power["weekday"], prefix="weekday", dtype=int)
    for c in dummies:
        power[c] = dummies[c]
    power["change_before"] = power["price_before"] - power.groupby("hour")["price"].shift(2)
    return power


def prepared(folder="data"):
    """After the deck's split cell: the first two days, which have no day before, dropped."""
    return load(folder).dropna()


def _date_words(ts, short=False):
    ts = pd.Timestamp(ts)
    if short:
        return f"{ts:%a} {ts.day} {ts:%b}"
    return f"{ts:%A} {ts.day} {ts:%B %Y}"


def _grid_of(values, frame):
    """One row per calendar day, 24 values per row, None where an hour is missing."""
    days = pd.date_range(frame.index.min().normalize(), frame.index.max().normalize(), freq="D")
    s = pd.Series(values, index=frame.index)
    s = s[~s.index.duplicated()]
    out = []
    for d in days:
        row = []
        for h in range(24):
            v = s.get(d + pd.Timedelta(hours=h), np.nan)
            row.append(None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
        out.append(row)
    return [d.strftime("%Y-%m-%d") for d in days], out


def _ints(s):
    return s.round(0).astype(int).to_numpy()


def _curve(model, frame):
    return [round(float(v), 4) for v in model.predict_proba(frame)[:, 1]]


def _dots(frame, label, edges):
    """Share of `label` and the count of hours, by bin of the wind change."""
    g = frame.groupby(pd.cut(frame["wind_change"], edges), observed=False)[label].agg(["mean", "size"])
    return {"share": [None if n == 0 else round(float(m), 4) for m, n in zip(g["mean"], g["size"])],
            "n": [int(n) for n in g["size"]]}


def build(folder="data"):
    raw = read(folder)
    t = prepared(folder)
    train, test = t.loc[:"2023"], t.loc["2024"]
    y_tr, y_te = train["up"], test["up"]
    upv = y_te.to_numpy() == 1
    D = {"n_rows": len(t), "n_train": len(train), "n_test": len(test),
         "share_up": {int(y): round(float(v), 4) for y, v in t["up"].groupby(t.index.year).mean().items()},
         "majority": round(float(1 - y_te.mean()), 4),
         "rule_acc": round(float(accuracy_score(y_te, (test["wind_change"] < 0).astype(int))), 4)}

    # ---- the ladder, as the deck's loop fits it
    ladder = [("the wind change", ["wind_change"]), ("+ weekday as a number", ["wind_change", "weekday"]),
              ("+ weekday as 7 columns", ["wind_change"] + DAYS), ("+ today's move", FEATURES)]
    fitted = {}
    D["ladder"] = []
    for name, cols in ladder:
        m = pipe().fit(train[cols], y_tr)
        q = m.predict_proba(test[cols])[:, 1]
        fitted[name] = m
        D["ladder"].append({"name": name, "auc": round(float(roc_auc_score(y_te, q)), 4),
                            "acc": round(float(accuracy_score(y_te, q >= 0.5)), 4)})
    model = fitted["+ today's move"]
    p = model.predict_proba(test[FEATURES])[:, 1]
    D["auc"] = float(roc_auc_score(y_te, p))
    D["acc"] = float(accuracy_score(y_te, p >= 0.5))
    D["cm"] = confusion_matrix(y_te, (p >= 0.5).astype(int)).tolist()
    D["coef"] = dict(zip(FEATURES, [round(float(v), 3) for v in model.named_steps["logit"].coef_[0]]))
    folds = TimeSeriesSplit(n_splits=5)
    D["c_folds"] = {str(C): round(float(cross_val_score(pipe(C=C), train[FEATURES], y_tr, cv=folds,
                                                       scoring="roc_auc").mean()), 4)
                    for C in [0.001, 0.01, 0.1, 1, 10]}
    cpath = []
    for C in np.logspace(-4, 1, 21):
        m = pipe(C=C)
        cv = cross_val_score(m, train[FEATURES], y_tr, cv=folds, scoring="roc_auc").mean()
        m.fit(train[FEATURES], y_tr)
        q = m.predict_proba(test[FEATURES])[:, 1]
        cpath.append({"C": float(C), "coef": [round(float(v), 4) for v in m.named_steps["logit"].coef_[0]],
                      "folds": round(float(cv), 4), "auc": round(float(roc_auc_score(y_te, q)), 4),
                      "brier": round(float(brier_score_loss(y_te, q)), 4),
                      "hist": [int(v) for v in np.histogram(q, bins=20, range=(0, 1))[0]]})
    D["cpath"] = {"rows": cpath, "names": ["wind change"] + WEEKDAY_NAMES + ["today's move"]}

    # ---- the S-curve: share up by wind change and weekday, and three models' curves
    edges = np.arange(-3000, 3001, 250)
    grid = np.arange(-3500, 3501, 50)
    sc = {"centres": [int(v) for v in (edges[:-1] + edges[1:]) / 2], "grid": [int(v) for v in grid],
          "dots": {str(d): _dots(train[train["weekday"] == d], "up", edges) for d in range(7)},
          "curves": {"alone": [], "number": [], "columns": []},
          "share_by_day": [round(float(v), 4) for v in train.groupby("weekday")["up"].mean()],
          "auc": {"alone": D["ladder"][0]["auc"], "number": D["ladder"][1]["auc"], "columns": D["ladder"][2]["auc"]},
          "avg_p": {}}
    sc["dots"]["all"] = _dots(train, "up", edges)
    for key, name, cols in [("alone", "the wind change", ["wind_change"]), ("number", "+ weekday as a number", ["wind_change", "weekday"]),
                            ("columns", "+ weekday as 7 columns", ["wind_change"] + DAYS)]:
        fit_p = pd.Series(fitted[name].predict_proba(train[cols])[:, 1], index=train.index)
        sc["avg_p"][key] = [round(float(v), 4) for v in fit_p.groupby(train["weekday"]).mean()]
    for d in range(7):
        onehot = {c: (1 if c == f"weekday_{d}" else 0) for c in DAYS}
        f_alone = pd.DataFrame({"wind_change": grid})
        f_num = pd.DataFrame({"wind_change": grid, "weekday": d})
        f_col = pd.DataFrame({"wind_change": grid, **{c: v for c, v in onehot.items()}})[["wind_change"] + DAYS]
        sc["curves"]["alone"].append(_curve(fitted["the wind change"], f_alone))
        sc["curves"]["number"].append(_curve(fitted["+ weekday as a number"], f_num))
        sc["curves"]["columns"].append(_curve(fitted["+ weekday as 7 columns"], f_col))
    D["scurve"] = sc
    early = load(folder)            # the wind-change cell runs before the split drops the first days
    D["wind_bands"] = {str(k): round(float(v), 4) for k, v in
                       early.groupby(pd.cut(early["wind_change"], [-np.inf, -1000, -300, 300, 1000, np.inf]),
                                     observed=False)["up"].mean().items()}
    D["move_bands"] = {str(k): round(float(v), 4) for k, v in
                       early.groupby(pd.cut(early["change_before"], [-np.inf, -200, -50, 50, 200, np.inf]),
                                     observed=False)["up"].mean().items()}

    # ---- the direction trade: long 1 MW when p >= t, short when p <= 1 - t
    move = (test["price"] - test["price_before"]).to_numpy()
    rows = []
    for th in np.round(np.arange(0.50, 0.951, 0.01), 2):
        lg, sh = p >= th, p <= 1 - th
        rows.append({"thr": float(th),
                     "lu": [int((lg & upv).sum()), round(float(move[lg & upv].sum()), 2)],
                     "ld": [int((lg & ~upv).sum()), round(float(move[lg & ~upv].sum()), 2)],
                     "su": [int((sh & upv).sum()), round(float(-move[sh & upv].sum()), 2)],
                     "sd": [int((sh & ~upv).sum()), round(float(-move[sh & ~upv].sum()), 2)],
                     "nu": int((~lg & ~sh & upv).sum()), "nd": int((~lg & ~sh & ~upv).sum())})
    costs = list(range(0, 201, 5))
    rule_long = (test["wind_change"] < 0).to_numpy()
    rule_gross = float(np.where(rule_long, move, -move).sum())
    D["trade"] = {"rows": rows, "costs": costs, "cost": COST, "n": int(len(move)),
                  "perfect": [round(float(np.clip(np.abs(move) - c, 0, None).sum())) for c in costs],
                  "rule": [round(rule_gross - c * len(move)) for c in costs]}

    def pnl(q, mv, th, cost):
        lg, sh = q >= th, q <= 1 - th
        return round(float(mv[lg].sum() - mv[sh].sum() - cost * (lg.sum() + sh.sum())))

    D["pnl_half"] = pnl(p, move, 0.5, COST)
    D["pnl_by_thr"] = {f"{th:.2f}": pnl(p, move, th, COST) for th in np.round(np.arange(0.5, 0.951, 0.01), 2)}
    best = max(D["pnl_by_thr"], key=D["pnl_by_thr"].get)
    D["pnl_best_thr"], D["pnl_best"] = best, D["pnl_by_thr"][best]
    D["pnl_perfect"] = D["trade"]["perfect"][costs.index(COST)]
    D["pnl_rule"] = D["trade"]["rule"][costs.index(COST)]
    D["move_abs"] = round(float(np.abs(move).sum()))
    D["move_sum"] = round(float(move.sum()))
    p_tr = model.predict_proba(train[FEATURES])[:, 1]
    move_tr = (train["price"] - train["price_before"]).to_numpy()
    D["pnl_train"] = {str(th): pnl(p_tr, move_tr, th, COST) for th in [0.5, 0.55, 0.6, 0.65, 0.7]}
    chosen = max(D["pnl_train"], key=D["pnl_train"].get)
    D["pnl_chosen_thr"], D["pnl_chosen"] = chosen, pnl(p, move, float(chosen), COST)
    D["best_thr_by_cost"] = {}
    for c in [0, 25, 50, 100, 150]:
        vals = {f"{th:.2f}": pnl(p, move, th, c) for th in np.round(np.arange(0.5, 0.951, 0.01), 2)}
        b = max(vals, key=vals.get)
        D["best_thr_by_cost"][str(c)] = [b, vals[b], vals["0.50"]]

    # ---- the crisis: a price above 1,000 kroner, trained through 2022 or not
    spike = (t["price"] > 1000).astype(int)
    D["spike_share"] = {int(y): round(float(v), 4) for y, v in spike.groupby(t.index.year).mean().items()}
    crisis = {}
    for key, start, end in [("crisis", "2022-01-01", "2023-12-31"), ("calm", "2023-01-01", "2023-12-31"),
                            ("y2022", "2022-01-01", "2022-12-31")]:
        idx = t.loc[start:end].index
        m = pipe().fit(t.loc[idx, LEVELS], spike.loc[idx])
        ps = m.predict_proba(test[LEVELS])[:, 1]
        ys = spike.loc[test.index].to_numpy()
        edges_c = [0, 0.1, 0.2, 0.4, 0.6, 1.0]
        g = pd.DataFrame({"p": ps, "y": ys}).groupby(pd.cut(ps, edges_c), observed=False).agg(
            n=("y", "size"), said=("p", "mean"), happened=("y", "mean"))
        crisis[key] = {
            "base": round(float(spike.loc[idx].mean()), 4),
            "mean_p": round(float(ps.mean()), 4), "actual": round(float(ys.mean()), 4),
            "auc": round(float(roc_auc_score(ys, ps)), 4), "brier": round(float(brier_score_loss(ys, ps)), 4),
            "called": int((ps >= 0.5).sum()),
            "buckets": [{"lo": edges_c[i], "hi": edges_c[i + 1], "n": int(g["n"].iloc[i]),
                         "said": None if g["n"].iloc[i] == 0 else round(float(g["said"].iloc[i]), 3),
                         "happened": None if g["n"].iloc[i] == 0 else round(float(g["happened"].iloc[i]), 3)}
                        for i in range(len(edges_c) - 1)],
        }
    D["crisis"] = crisis

    # ---- 2024 day by hour: prices, the day before, the model's probabilities and mistakes
    days24, price_grid = _grid_of(_ints(test["price"]), test)
    _, before_grid = _grid_of(_ints(test["price_before"]), test)
    _, prob_grid = _grid_of(np.round(p, 3), test)
    outcome = np.where(p >= 0.5, np.where(upv, "TP", "FP"), np.where(upv, "FN", "TN"))
    _, err_grid = _grid_of(outcome, test)
    _, move_grid = _grid_of(_ints(test["price"] - test["price_before"]), test)
    _, wind_grid = _grid_of(_ints(test["wind"]), test)
    _, windb_grid = _grid_of(_ints(test["wind"] - test["wind_change"]), test)
    have = sorted(set(t["date"]))
    before_day = []
    for d in days24:
        earlier = [x for x in have if x < d]
        before_day.append(earlier[-1] if earlier else None)
    days_up = test.assign(right=((p >= 0.5) == upv).astype(int)).groupby("date").agg(
        ups=("up", "sum"), n=("up", "size"), right=("right", "sum"))
    whole = np.maximum(days_up["ups"], days_up["n"] - days_up["ups"]) >= 20
    D["whole_days"] = {"days": int(len(days_up)), "whole": int(whole.sum()),
                       "acc_whole": round(float(days_up["right"][whole].sum() / days_up["n"][whole].sum()), 4),
                       "acc_split": round(float(days_up["right"][~whole].sum() / days_up["n"][~whole].sum()), 4)}
    D["mean_p_2024"] = round(float(p.mean()), 4)
    _, up_grid = _grid_of(upv.astype(int), test)
    _, pred_grid = _grid_of((p >= 0.5).astype(int), test)
    D["heat"] = {"days": days24, "before_day": before_day, "price": price_grid, "before": before_grid, "prob": prob_grid,
                 "error": err_grid, "move": move_grid, "up": up_grid, "pred": pred_grid,
                 "wind": wind_grid, "wind_before": windb_grid}

    # ---- 2022 and 2023, one day at a time: the years every model learns from
    early = raw.loc[:"2023"]
    ex_have = sorted(set(early["date"]))
    all_days, all_grid = _grid_of(_ints(early["price"]), early)
    keep = [i for i, d in enumerate(all_days) if d in set(ex_have)]          # the 2 days with no data
    ex_days = [all_days[i] for i in keep]
    means = early.groupby("date")["price"].mean()
    D["explorer"] = {"days": ex_days, "curves": [all_grid[i] for i in keep],
                     "means": [round(float(means[d])) for d in ex_days]}
    D["explorer"]["marks"] = {"the most expensive day": means.idxmax(), "the cheapest day": means.idxmin()}

    # ---- hours above or below a price, by year
    levels_kr = list(range(-500, 3001, 50))
    yrs = [2022, 2023, 2024]
    D["hours"] = {"levels": levels_kr, "years": yrs,
                  "total": [int((raw.index.year == y).sum()) for y in yrs],
                  "below": [[int((raw.loc[str(y), "price"] < v).sum()) for v in levels_kr] for y in yrs],
                  "above": [[int((raw.loc[str(y), "price"] > v).sum()) for v in levels_kr] for y in yrs]}

    # ---- the median price by band of the wind or the sun forecast, all years and each year
    bands = {"wind": ([0, 500, 1500, 3000, 6000], ["below 500 MW", "500 to 1,500 MW", "1,500 to 3,000 MW", "above 3,000 MW"]),
             "solar": ([-1, 10, 300, 1000, 5000], ["no sun, below 10 MW", "10 to 300 MW", "300 to 1,000 MW", "above 1,000 MW"])}
    D["bands"] = {}
    for col, (edges_b, names_b) in bands.items():
        out = {"names": names_b}
        for key, frame in [("all", raw)] + [(str(y), raw.loc[str(y)]) for y in yrs]:
            gb = frame.groupby(pd.cut(frame[col], edges_b), observed=False)["price"].agg(["median", "size"])
            out[key] = {"median": [None if n == 0 else round(float(v)) for v, n in zip(gb["median"], gb["size"])],
                        "n": [int(n) for n in gb["size"]]}
        D["bands"][col] = out
    d24 = raw.loc["2024"].groupby("date")["price"].mean()
    D["heat_marks"] = {"the most expensive day": d24.idxmax(), "the cheapest day": d24.idxmin()}

    # ---- the shape of a day, month by month: average price and forecasts by hour
    month = raw.index.to_period("M")
    by = raw.groupby([month, raw["hour"]])
    labels = sorted(set(month))
    D["months"] = {"labels": [f"{m.strftime('%B')} {m.year}" for m in labels],
                   "price": [[round(float(by["price"].mean()[(m, h)])) for h in range(24)] for m in labels],
                   "solar": [[round(float(by["solar"].mean()[(m, h)])) for h in range(24)] for m in labels],
                   "wind": [[round(float(by["wind"].mean()[(m, h)])) for h in range(24)] for m in labels]}
    for key, keep_rows in [("wd", raw["weekend"] == 0), ("we", raw["weekend"] == 1)]:
        sub = raw[keep_rows]
        by2 = sub.groupby([sub.index.to_period("M"), sub["hour"]])
        for col in ("price", "solar"):
            mean2 = by2[col].mean()
            D["months"][col + "_" + key] = [[round(float(mean2.get((m, h), np.nan))) if not pd.isna(mean2.get((m, h), np.nan)) else None
                                             for h in range(24)] for m in labels]
    D["weekend_08"] = {"weekday": round(float(raw[(raw["weekend"] == 0) & (raw["hour"] == 8)]["price"].mean())),
                       "weekend": round(float(raw[(raw["weekend"] == 1) & (raw["hour"] == 8)]["price"].mean()))}

    # ---- the neighbour pool: the training hours, on the four columns k-NN uses
    sd = train[NEAR].std(ddof=0)
    mu = train[NEAR].mean()
    D["knn"] = {"wind_change": [int(v) for v in _ints(train["wind_change"])],
                "change_before": [int(v) for v in _ints(train["change_before"])],
                "weekday": [int(v) for v in train["weekday"]], "hour": [int(v) for v in train["hour"]],
                "up": [int(v) for v in y_tr], "mean": [round(float(v), 4) for v in mu],
                "sd": [round(float(v), 4) for v in sd]}
    knn = near_pipe().fit(train[NEAR], y_tr)
    q_knn = knn.predict_proba(test[NEAR])[:, 1]
    D["knn_auc"] = float(roc_auc_score(y_te, q_knn))
    D["knn_acc"] = float(accuracy_score(y_te, q_knn >= 0.5))
    bare = KNeighborsClassifier(n_neighbors=K).fit(train[NEAR], y_tr)
    D["knn_bare_auc"] = float(roc_auc_score(y_te, bare.predict_proba(test[NEAR])[:, 1]))
    D["knn_k_folds"] = {str(k): round(float(cross_val_score(near_pipe(k), train[NEAR], y_tr, cv=folds,
                                                           scoring="roc_auc").mean()), 4)
                        for k in [51, 101, 201, 301, 501]}
    D["logit_folds"] = D["c_folds"]["1"]

    # ---- three classes: lower, about the same, higher, by 50 kroner
    diff = t["price"] - t["price_before"]
    mv3 = pd.cut(diff, [-np.inf, -BAND, BAND, np.inf], labels=["lower", "about the same", "higher"]).astype(str)
    m3tr, m3te = mv3.loc[:"2023"], mv3.loc["2024"]
    three = pipe().fit(train[FEATURES], m3tr)
    pred3 = three.predict(test[FEATURES])
    knn3 = near_pipe().fit(train[NEAR], m3tr)
    rule3 = np.where(test["wind_change"] > 300, "lower", np.where(test["wind_change"] < -300, "higher", "about the same"))
    order3 = ["lower", "about the same", "higher"]
    D["three"] = {"acc": round(float(accuracy_score(m3te, pred3)), 4),
                  "majority": round(float(m3te.value_counts(normalize=True).max()), 4),
                  "shares": {k: round(float(v), 4) for k, v in m3te.value_counts(normalize=True).items()},
                  "rule": round(float(accuracy_score(m3te, rule3)), 4),
                  "knn": round(float(accuracy_score(m3te, knn3.predict(test[NEAR]))), 4),
                  "cm": confusion_matrix(m3te, pred3, labels=order3).tolist()}

    # ---- a big move either way: more than 150 kroner
    big = (diff.abs() > BIG).astype(int)
    size = t["wind_change"].abs().rename("size_wind_change")
    b_tr, b_te = big.loc[:"2023"], big.loc["2024"]
    tr_s = pd.concat([train, size.loc[:"2023"]], axis=1)
    te_s = pd.concat([test, size.loc["2024"]], axis=1)
    D["big"] = {"share": {int(y): round(float(v), 4) for y, v in big.groupby(t.index.year).mean().items()}}
    for name, cols in [("plain", FEATURES), ("size", FEATURES + ["size_wind_change"])]:
        m = pipe().fit(tr_s[cols], b_tr)
        D["big"][name] = round(float(roc_auc_score(b_te, m.predict_proba(te_s[cols])[:, 1])), 4)
    D["big"]["knn"] = round(float(roc_auc_score(b_te, near_pipe().fit(train[NEAR], b_tr).predict_proba(test[NEAR])[:, 1])), 4)
    vtr = tr_s.assign(big=b_tr)
    vg = pd.DataFrame({"wind_change": grid, "size_wind_change": np.abs(grid)})
    v_lin = pipe().fit(tr_s[["wind_change"]], b_tr)
    v_abs = pipe().fit(tr_s[["size_wind_change"]], b_tr)
    v_knn = near_pipe().fit(tr_s[["wind_change"]], b_tr)
    D["vshape"] = {"centres": sc["centres"], "grid": sc["grid"], "dots": _dots(vtr, "big", edges),
                   "curves": {"linear": _curve(v_lin, vg[["wind_change"]]), "size": _curve(v_abs, vg[["size_wind_change"]]),
                              "knn": _curve(v_knn, vg[["wind_change"]])},
                   "auc": {"linear": round(float(roc_auc_score(b_te, v_lin.predict_proba(te_s[["wind_change"]])[:, 1])), 4),
                           "size": round(float(roc_auc_score(b_te, v_abs.predict_proba(te_s[["size_wind_change"]])[:, 1])), 4),
                           "knn": round(float(roc_auc_score(b_te, v_knn.predict_proba(te_s[["wind_change"]])[:, 1])), 4)}}

    # ---- the forecasts: eight hours, what the models see, and their answers
    cand = test.assign(p_log=p, p_knn=q_knn)
    rounds = []
    for stamp in ROUNDS:
        ts = pd.Timestamp(stamp)
        r = cand.loc[ts]
        rounds.append({"when": _date_words(ts), "short": _date_words(ts, short=True),
                       "weekday": f"{ts:%A}", "hour": int(ts.hour),
                       "price_2before": round(float(r["price_before"] - r["change_before"])),
                       "price_before": round(float(r["price_before"])), "price": round(float(r["price"])),
                       "wind_before": round(float(r["wind"] - r["wind_change"])), "wind": round(float(r["wind"])),
                       "truth": int(r["up"]), "logit": round(float(r["p_log"]), 3), "knn": round(float(r["p_knn"]), 3)})
    D["game"] = {"rounds": rounds}
    truth = np.array([r["truth"] for r in rounds])
    for name in ["logit", "knn"]:
        q = np.array([r[name] for r in rounds])
        D[f"game_{name}_brier"] = round(float(np.mean((q - truth) ** 2)), 4)
        D[f"game_{name}_logloss"] = round(float(log_loss(truth, np.clip(q, 0.01, 0.99), labels=[0, 1])), 4)

    # ---- the S&P 500: the same kind of model, yesterday's return and the weekday
    prices = pd.read_csv(Path(folder).resolve().parent.parent / "data" / "prices.csv", parse_dates=["date"])
    spy = prices[prices["ticker"] == "SPY"].set_index("date")["close"]
    ret = spy.pct_change()
    s = pd.DataFrame({"ret": ret, "ret_before": ret.shift(1)}).dropna()
    s["up"] = (s["ret"] > 0).astype(int)
    sd_ = pd.get_dummies(s.index.weekday, prefix="weekday", dtype=int)
    sd_.index = s.index
    s = pd.concat([s, sd_], axis=1)
    scols = ["ret_before"] + list(sd_.columns)
    a_, b_ = s.loc["2022":"2023"], s.loc["2024"]
    sm = pipe().fit(a_[scols], a_["up"])
    sp = sm.predict_proba(b_[scols])[:, 1]
    D["spy"] = {"auc": round(float(roc_auc_score(b_["up"], sp)), 4), "acc": round(float(accuracy_score(b_["up"], sp >= 0.5)), 4),
                "up_share": round(float(b_["up"].mean()), 4), "n": int(len(b_))}

    # ---- the feature lab, cached
    lab_path = Path(folder) / "lab.json"
    D["lab"] = json.loads(lab_path.read_text(encoding="utf-8")) if lab_path.exists() else None
    return D


def _lab_one(tr, te, cols, label):
    folds = TimeSeriesSplit(n_splits=5)
    out = []
    for m in (pipe(), near_pipe()):
        cv = cross_val_score(m, tr[cols], tr[label], cv=folds, scoring="roc_auc").mean()
        m.fit(tr[cols], tr[label])
        out += [round(float(cv), 4), round(float(roc_auc_score(te[label], m.predict_proba(te[cols])[:, 1])), 4)]
    return out


def build_lab(folder="data"):
    """AUC on the folds and on 2024, for every set of the lab's columns, both labels, both models."""
    from joblib import Parallel, delayed
    raw = read(folder)
    t = prepared(folder)
    t["solar_change"] = (raw["solar"] - raw.groupby("hour")["solar"].shift(1)).loc[t.index]
    t["size_wind_change"] = t["wind_change"].abs()
    t["big"] = ((t["price"] - t["price_before"]).abs() > BIG).astype(int)
    tr, te = t.loc[:"2023"], t.loc["2024"]
    n = len(LAB)
    jobs = []
    for label in ["up", "big"]:
        for mask in range(1, 2 ** n):
            cols = []
            for i in range(n):
                if mask >> i & 1:
                    cols += LAB[i][1]
            jobs.append((label, mask, cols))
    res = Parallel(n_jobs=-1, verbose=0)(delayed(_lab_one)(tr, te, c, l) for l, m, c in jobs)
    out = {"chips": [{"name": a, "cols": b, "kind": c, "code": d} for a, b, c, d in LAB],
           "up": [None] * 2 ** n, "big": [None] * 2 ** n}
    for (label, mask, cols), r in zip(jobs, res):
        out[label][mask] = r
    return out


if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "session_10/data"
    if "--lab" in sys.argv:
        import time
        t0 = time.time()
        lab = build_lab(folder)
        Path(folder, "lab.json").write_text(json.dumps(lab, separators=(",", ":")), encoding="utf-8")
        print(f"wrote {Path(folder, 'lab.json')} in {time.time() - t0:.0f} s")
        sys.exit(0)
    D = build(folder)
    print("rows", D["n_rows"], "train", D["n_train"], "test", D["n_test"], "share up", D["share_up"],
          "majority", D["majority"], "rule", D["rule_acc"])
    print("wind bands", D["wind_bands"])
    print("move bands", D["move_bands"])
    for r in D["ladder"]:
        print("  ladder", r)
    print("final AUC", round(D["auc"], 4), "acc", round(D["acc"], 4), "cm", D["cm"])
    print("coef", D["coef"])
    print("C on folds", D["c_folds"])
    print("scurve average p by weekday", D["scurve"]["avg_p"])
    print("share by day", D["scurve"]["share_by_day"])
    print("trade: at 0.5", D["pnl_half"], "| best", D["pnl_best_thr"], D["pnl_best"], "| perfect", D["pnl_perfect"],
          "| rule", D["pnl_rule"], "| |move| sum", D["move_abs"], "move sum", D["move_sum"])
    print("best threshold by cost [thr, best, at 0.5]", D["best_thr_by_cost"])
    print("training thresholds", D["pnl_train"], "-> chosen", D["pnl_chosen_thr"], D["pnl_chosen"])
    print("whole days", D["whole_days"], "mean p 2024", D["mean_p_2024"], "at 0.6", D["pnl_by_thr"]["0.60"])
    print("spike shares", D["spike_share"])
    print("C path", [(round(r["C"], 5), r["folds"], r["auc"], r["brier"]) for r in D["cpath"]["rows"]][::4])
    print("hours above 1,000", [D["hours"]["above"][i][D["hours"]["levels"].index(1000)] for i in range(3)], "below 0",
          [D["hours"]["below"][i][D["hours"]["levels"].index(0)] for i in range(3)], "total", D["hours"]["total"])
    print("bands", {k: (v["all"]["median"], v["2024"]["median"]) for k, v in D["bands"].items()}, "08:00", D["weekend_08"], "marks", D["heat_marks"])
    for k, v in D["crisis"].items():
        print(k, {kk: vv for kk, vv in v.items() if kk != "buckets"}, [(b["n"], b["said"], b["happened"]) for b in v["buckets"]])
    print("k-NN AUC", round(D["knn_auc"], 4), "acc", round(D["knn_acc"], 4), "without scaler", round(D["knn_bare_auc"], 4),
          "k on folds", D["knn_k_folds"], "logit folds", D["logit_folds"])
    print("three", D["three"])
    print("big", D["big"], "vshape AUC", D["vshape"]["auc"])
    print("explorer marks", D["explorer"]["marks"], len(D["explorer"]["days"]), D["explorer"]["days"][0])
    print("spy", D["spy"])
    print("game: logit brier", D["game_logit_brier"], "logloss", D["game_logit_logloss"],
          "| knn brier", D["game_knn_brier"], "logloss", D["game_knn_logloss"])
    for r in D["game"]["rounds"]:
        print(f"  {r['when']:<28} {r['hour']:>2}:00  price {r['price_2before']:>6} {r['price_before']:>6} -> {r['price']:>6}"
              f"  wind {r['wind_before']:>5} -> {r['wind']:>5}  up {r['truth']}  logit {r['logit']:.3f}  knn {r['knn']:.3f}")
    print("lab cached:", D["lab"] is not None)
    print("sizes (kB):", {k: round(len(json.dumps(v)) / 1000) for k, v in D.items() if isinstance(v, (dict, list))})
