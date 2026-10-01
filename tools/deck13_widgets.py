# -*- coding: utf-8 -*-
"""The interactive pieces of Session 13's deck, each returned as one block of
HTML (a style block, the markup, and a script), in the pattern and with the
helpers of Session 10's tools/deck10_widgets.py.

Two of them are PLAY slides, with controls: the house placed anywhere in the
inner postcodes (two buttons and a pointer readout) and the house as its area
grows (a slider and a toggle). Two of them only read what students entered in
earlier decks, kept in this browser by Sessions 7 and 10 (localStorage keys
mlfin7-guess, mlfin10-updown and mlfin10-updown-revised): the house valued by
every model, and the eight hours of Session 10 scored again with the trees.
Nothing plays by itself.

    import deck13_widgets as W
    print(W.emit(W.place_map(DATA["houses"]["map"])))     # inside a  #| output: asis  cell
"""
from deck10_widgets import HELPERS, BASE_CSS, emit, _fill   # noqa: F401  (emit is used by the deck)

NAMES = {"ridge": "linear regression", "tree": "a tree", "forest": "a forest", "boosted": "boosted trees",
         "boosted_place": "boosted trees on coordinates", "ridge_place": "linear regression on coordinates",
         "boosted_rising": "boosted trees, told to rise"}


# ============================================================ the house placed anywhere
PLACE = r"""
<style>__BASECSS__
.w-pm-top { display: flex; align-items: center; gap: 1em; margin: 0 0 0.35em; }
.w-pm-wrap { position: relative; width: __W__px; }
.w-pm-key { position: absolute; right: 10px; bottom: 10px; background: rgba(252,252,251,0.92); border: 1px solid #c3c2b7; border-radius: 6px; padding: 5px 9px; font: 12.5px "Segoe UI", sans-serif; color: #52514e; }
.w-pm-key .bar { width: 170px; height: 10px; border-radius: 3px; margin: 3px 0 2px; }
.w-pm-key .ends { display: flex; justify-content: space-between; }
.w-pm-credit { position: absolute; left: 8px; bottom: 6px; pointer-events: none; background: rgba(252,252,251,0.85); border-radius: 4px; padding: 1px 6px; font: 11px "Segoe UI", sans-serif; color: #6e6d68; }
</style>
<div id="pm">
<div class="w-pm-top"><span class="w-marks"><button class="w-btn on" data-m="ridge_place">linear regression</button><button class="w-btn" data-m="boosted_place">boosted trees</button></span>
<span class="w-read" id="pm-read"></span></div>
<div class="w-pm-wrap">
<svg id="pm-svg" width="__W__" height="__H__" viewBox="0 0 __W__ __H__" role="img" aria-label="A map of the inner Aarhus postcodes, shaded by what a model says the house from Aarhus V would be worth at each spot"></svg>
<div class="w-pm-key"><div>the house valued here, million kroner</div><div class="bar" id="pm-bar"></div><div class="ends"><span id="pm-lo"></span><span id="pm-hi"></span></div></div>
<div class="w-pm-credit">map: &copy; Esri, HERE, Garmin, OpenStreetMap contributors</div>
</div>
</div>
<script>
(function () {
__HELPERS__
  var D = __DATA__;
  var svg = document.getElementById("pm-svg"); if (!svg) return;
  var LO = D.lo, HI = D.hi, ST = [[247, 250, 253], [134, 182, 239], [28, 92, 171], [13, 54, 107]];
  function colour(v) { return ramp(ST, (v - LO) / (HI - LO)); }
  el("image", { href: "data/map13.png", x: 0, y: 0, width: D.width, height: D.height, preserveAspectRatio: "none" }, svg);
  var cells = el("g", {}, svg), mark = el("g", {}, svg), model = "ridge_place", rects = [];
  D.cells.forEach(function (c, k) { rects.push(el("rect", { x: c[0], y: c[1], width: c[2], height: c[3], "fill-opacity": 0.72, "data-k": k }, cells)); });
  el("circle", { cx: D.house[0], cy: D.house[1], r: 9, fill: "#d99a1c", stroke: "#fcfcfb", "stroke-width": 2.5 }, mark);
  txt(mark, D.house[0] + 14, D.house[1] + 5, "the house", { fill: "#8a6508", size: 15, weight: 700, halo: true });
  var read = document.getElementById("pm-read");
  function base() { read.innerHTML = "At the house's own spot: <b>" + D.own[model].toFixed(2) + " million</b>. Point at the map to read another spot."; }
  function paint() { rects.forEach(function (r, k) { r.setAttribute("fill", colour(D.values[model][k])); }); base(); }
  cells.addEventListener("mousemove", function (e) { var k = e.target.getAttribute("data-k"); if (k === null) return; read.innerHTML = "The house placed here: <b>" + D.values[model][+k].toFixed(2) + " million kroner</b>."; });
  cells.addEventListener("mouseleave", base);
  document.querySelectorAll("#pm .w-btn").forEach(function (b) { b.addEventListener("click", function () { document.querySelectorAll("#pm .w-btn").forEach(function (x) { x.classList.remove("on"); }); b.classList.add("on"); model = b.getAttribute("data-m"); paint(); }); });
  var bar = document.getElementById("pm-bar"); bar.style.background = "linear-gradient(to right," + [0, 0.33, 0.66, 1].map(function (t) { return ramp(ST, t); }).join(",") + ")";
  document.getElementById("pm-lo").textContent = LO.toFixed(0); document.getElementById("pm-hi").textContent = HI.toFixed(0) + " or more";
  paint();
})();
</script>
"""


def place_map(data, own):
    """`data` is deck13_data's houses()["map"]; `own` the house's value under each map model."""
    return _fill(PLACE, data={"width": data["width"], "height": data["height"], "cells": data["cells"],
                              "values": data["values"], "house": data["house"], "lo": 2, "hi": 8,
                              "own": {k: own[k] / 1e6 for k in data["values"]}},
                 w=str(data["width"]), h=str(data["height"]))


# ============================================================ the house, valued by every model
VALUED = r"""
<style>__BASECSS__
.w-val { margin: 0.2em 0 0; }
.w-val-row { display: flex; align-items: center; gap: 14px; margin: 7px 0; }
.w-val-lab { flex: 0 0 13.5em; text-align: right; font: 600 16px "Segoe UI", sans-serif; color: #52514e; }
.w-val-track { flex: 1 1 0; height: 30px; background: #f4f4f1; border-radius: 6px; position: relative; overflow: hidden; }
.w-val-bar { height: 100%; border-radius: 6px; background: #86b6ef; }
.w-val-bar.guess { background: #d99a1c; }
.w-val-bar.price { background: #0d366b; }
.w-val-bar.best { background: #1c5cab; }
.w-val-line { position: absolute; top: 0; bottom: 0; width: 0; border-left: 2px dashed #0d366b; }
.w-val-num { flex: 0 0 9.5em; font: 600 16px "Cascadia Mono", Consolas, monospace; color: #0d366b; }
.w-val-num.unset { color: #c3c2b7; font-weight: 400; }
</style>
<div class="w-val" id="val"></div>
<script>
(function () {
__HELPERS__
  var D = __DATA__, box = document.getElementById("val"); if (!box) return;
  function guess() { try { var v = parseFloat(localStorage.getItem("mlfin7-guess")); return isFinite(v) && v > 0 ? v : null; } catch (e) { return null; } }
  function render() {
    var g = guess(), rows = [["your guess, Session 7", g, "guess"]].concat(D.rows.map(function (r) { return [r[0], r[1], r[2]]; })).concat([["sold for", D.price, "price"]]);
    var top = Math.max.apply(null, rows.map(function (r) { return r[1] || 0; })) * 1.04, html = "";
    rows.forEach(function (r) {
      var w = r[1] ? (100 * r[1] / top) : 0, num = r[1] ? fmt(r[1]) + " kr" : "no guess entered";
      html += "<div class='w-val-row'><span class='w-val-lab'>" + r[0] + "</span><div class='w-val-track'><div class='w-val-bar " + r[2] + "' style='width:" + w + "%'></div><div class='w-val-line' style='left:" + (100 * D.price / top) + "%'></div></div><span class='w-val-num" + (r[1] ? "" : " unset") + "'>" + num + "</span></div>";
    });
    box.innerHTML = html;
  }
  render();
  onShow(box, render);
})();
</script>
"""


def valued(house):
    """`house` is deck13_data's houses()["house"]."""
    v = house["values"]
    order = ["ridge", "tree", "forest", "boosted", "boosted_place"]
    closest = min(order, key=lambda k: abs(v[k] - house["price"]))
    labels = {"ridge": "linear regression, Session 7", "tree": "a tree", "forest": "a forest", "boosted": "boosted trees",
              "boosted_place": "boosted trees on coordinates"}
    rows = [[labels[k], v[k], "best" if k == closest else ""] for k in order]
    return _fill(VALUED, data={"rows": rows, "price": house["price"]})


# ============================================================ a bigger house
BIGGER = r"""
<style>__BASECSS__
.w-bg-top { display: flex; align-items: center; gap: 1.2em; margin-bottom: 0.2em; }
.w-bg-top label { font: 600 0.52em "Segoe UI", sans-serif; color: #0d366b; display: flex; align-items: center; gap: 0.5em; white-space: nowrap; }
.w-bg-top input[type=range] { width: 17em; }
</style>
<div id="bg">
<div class="w-bg-top"><label>area <input class="w-range" type="range" id="bg-a" min="60" max="300" step="5" value="150" aria-label="the area of the house"> <span class="w-val" id="bg-av"></span></label>
<span class="w-marks"><button class="w-btn" id="bg-rise">boosted trees, told to rise</button></span></div>
<svg id="bg-svg" width="1040" height="400" viewBox="0 0 1040 400" role="img" aria-label="The value of the house from Aarhus V as its area grows from 60 to 300 square metres, under linear regression, a forest and boosted trees"></svg>
<div class="w-read" id="bg-read"></div>
</div>
<script>
(function () {
__HELPERS__
  var D = __DATA__, svg = document.getElementById("bg-svg"); if (!svg) return;
  var L = 70, R = 840, T = 12, B = 350, A0 = 60, A1 = 300, V1 = D.vmax, showRise = false;
  function X(a) { return L + (a - A0) / (A1 - A0) * (R - L); } function Y(v) { return B - v / V1 * (B - T); }
  var LINES = [["ridge", "linear regression", "#0d366b"], ["forest", "a forest", "#898781"], ["boosted_place", "boosted trees", "#b3402f"], ["boosted_rising", "boosted trees, told to rise", "#b8860b"]];
  function at(name, a) { var i = Math.round((a - D.areas[0]) / 5); return D.values[name][Math.max(0, Math.min(D.areas.length - 1, i))] / 1e6; }
  function draw() {
    var a = +document.getElementById("bg-a").value;
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    el("rect", { x: X(D.data_max), y: T, width: R - X(D.data_max), height: B - T, fill: "#f1f0ec" }, svg);
    txt(svg, X(D.data_max) + 8, T + 18, "larger than 99 percent of the sales", { fill: "#898781", size: 13 });
    for (var v = 0; v <= V1 + 1e-9; v += 2) { el("line", { x1: L, y1: Y(v), x2: R, y2: Y(v), stroke: v === 0 ? "#c3c2b7" : "#ecebe6" }, svg); txt(svg, L - 8, Y(v) + 4, v.toFixed(0), { anchor: "end" }); }
    [60, 100, 150, 200, 250, 300].forEach(function (x) { txt(svg, X(x), B + 19, x + " m²", { anchor: "middle" }); });
    txt(svg, 18, (T + B) / 2, "the house's value, million kroner", { anchor: "middle", rotate: -90, fill: "#52514e", size: 13 });
    LINES.forEach(function (l) {
      if (l[0] === "boosted_rising" && !showRise) return;
      var d = ""; D.areas.forEach(function (x, i) { d += (i ? " L" : "M") + X(x).toFixed(1) + "," + Y(D.values[l[0]][i] / 1e6).toFixed(1); });
      el("path", { d: d, fill: "none", stroke: l[2], "stroke-width": l[0] === "forest" ? 2.6 : 3.2, "stroke-dasharray": l[0] === "boosted_rising" ? "8 5" : "none" }, svg);
      var last = D.values[l[0]][D.areas.length - 1] / 1e6;
      txt(svg, R + 10, Y(last) + 5 + (l[0] === "boosted_rising" ? 16 : 0), l[1], { fill: l[2], size: 14, weight: 700 });
    });
    el("line", { x1: X(a), y1: T, x2: X(a), y2: B, stroke: "#141413", "stroke-width": 1.2, "stroke-dasharray": "4 4" }, svg);
    var parts = [];
    LINES.forEach(function (l) { if (l[0] === "boosted_rising" && !showRise) return; var v = at(l[0], a); el("circle", { cx: X(a), cy: Y(v), r: 6, fill: l[2], stroke: "#fcfcfb", "stroke-width": 2 }, svg); parts.push(l[1] + " <b>" + v.toFixed(2) + "</b>"); });
    document.getElementById("bg-av").textContent = a + " m²";
    document.getElementById("bg-read").innerHTML = "At " + a + " m², in million kroner: " + parts.join(" · ");
  }
  document.getElementById("bg-a").addEventListener("input", draw);
  document.getElementById("bg-rise").addEventListener("click", function (e) { showRise = !showRise; e.target.classList.toggle("on", showRise); draw(); });
  draw();
})();
</script>
"""


def bigger(curve):
    """`curve` is deck13_data's houses()["curve"]."""
    vmax = max(max(v) for v in curve["values"].values()) / 1e6
    vmax = 2 * int(vmax / 2 + 1)
    return _fill(BIGGER, data={"areas": curve["areas"], "values": curve["values"], "data_max": curve["data_max"], "vmax": vmax})


# ============================================================ the eight hours, scored again
SCORE = r"""
<style>__BASECSS__
.w-sc13 table { border-collapse: collapse; font: 14.5px "Segoe UI", sans-serif; width: 100%; }
.w-sc13 th { color: #52514e; font-weight: 600; text-align: right !important; padding: 4px 8px; border-bottom: 1.5px solid #c3c2b7; line-height: 1.2; vertical-align: bottom; }
.w-sc13 td { text-align: right !important; padding: 4px 8px; border-bottom: 1px solid #ecebe6; font-family: "Cascadia Mono", Consolas, monospace; color: #52514e; white-space: nowrap; }
.w-sc13 td.t, .w-sc13 th.t { text-align: left !important; font-family: "Segoe UI", sans-serif; }
.w-sc13 tr.sum td { border-top: 1.5px solid #c3c2b7; color: #0d366b; font-weight: 700; }
.w-sc13 td.best { background: #eaf1fa; color: #0d366b; font-weight: 700; }
.w-sc13 td.unset { color: #c3c2b7; }
.w-sc13-note { font: 0.44em "Segoe UI", sans-serif; color: #52514e; margin-top: 0.35em; }
</style>
<div class="w-sc13" id="sc13"><div id="sc13-body" style="min-height: 330px"></div><div class="w-sc13-note" id="sc13-note"></div></div>
<script>
(function () {
__HELPERS__
  var G = __DATA__, K1 = "mlfin10-updown", K2 = "mlfin10-updown-revised", WHO = ["first", "rev", "logit", "knn", "forest", "boosted"];
  var body = document.getElementById("sc13-body"); if (!body) return;
  function brier(p, y) { return (p - y) * (p - y); }
  function logl(p, y) { p = Math.min(0.99, Math.max(0.01, p)); return -(y * Math.log(p) + (1 - y) * Math.log(1 - p)); }
  function got(a, k) { return a && a[k] !== null && a[k] !== undefined; }
  function render() {
    var first = recall(K1) || [], rev = recall(K2) || [], m = G.rounds.length, rows = "", unset = 0, tot = {};
    WHO.forEach(function (w) { tot[w] = [0, 0]; });
    G.rounds.forEach(function (r, k) {
      var f = got(first, k) ? first[k] : 0.5, ps = { first: f, rev: got(rev, k) ? rev[k] : f, logit: r.logit, knn: r.knn, forest: r.forest, boosted: r.boosted };
      if (!got(first, k)) unset++;
      var best = Math.min.apply(null, WHO.map(function (w) { return brier(ps[w], r.truth); }));
      WHO.forEach(function (w) { tot[w][0] += brier(ps[w], r.truth); tot[w][1] += logl(ps[w], r.truth); });
      rows += "<tr><td class='t'>" + r.short + ", " + hh(r.hour) + "</td><td class='t'>" + (r.truth ? "higher" : "not higher") + "</td>" +
        WHO.map(function (w) { var cls = Math.abs(brier(ps[w], r.truth) - best) < 1e-9 ? "best" : (w === "first" && !got(first, k) ? "unset" : ""); return "<td class='" + cls + "'>" + Math.round(ps[w] * 100) + "%</td>"; }).join("") + "</tr>";
    });
    function sumRow(label, j) {
      var v = WHO.map(function (w) { return tot[w][j] / m; }), best = Math.min.apply(null, v);
      return "<tr class='sum'><td class='t' colspan='2'>" + label + "</td>" + v.map(function (x) { return "<td" + (Math.abs(x - best) < 1e-9 ? " class='best'" : "") + ">" + x.toFixed(3) + "</td>"; }).join("") + "</tr>";
    }
    body.innerHTML = "<table><tr><th class='t'>tomorrow, at the hour</th><th class='t'>what happened</th><th>your first<br>answer</th><th>your revised<br>answer</th><th>logistic<br>regression</th><th>k-NN</th><th>a forest</th><th>boosted<br>trees</th></tr>" + rows +
      sumRow("Brier score, average", 0) + sumRow("log loss, average", 1) + "</table>";
    var note = "Shaded: the closest forecast on each hour, and the lowest score. Your answers come from the Session 10 deck, opened in this browser. ";
    if (unset) note += "Answers not set count as 50 percent. ";
    document.getElementById("sc13-note").textContent = note;
  }
  function again() { render(); try { if (window.Reveal && Reveal.layout) Reveal.layout(); } catch (e) {} }
  render();
  onShow(body, again);
})();
</script>
"""


def scoreboard(game):
    """`game` is deck13_data's hours()["game"]."""
    keep = ("short", "hour", "truth", "logit", "knn", "forest", "boosted")
    return _fill(SCORE, data={"rounds": [{k: r[k] for k in keep} for r in game["rounds"]]})
