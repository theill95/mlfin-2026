# -*- coding: utf-8 -*-
"""The interactive pieces of Session 10's deck, each returned as one block of
HTML: a style block, the markup, and a script that runs when the page loads.

Every widget is self-contained (its own helpers, its own element ids) and reads
its data from JSON embedded at render time, so the deck needs no network and no
server beyond the static files. Nothing plays by itself: every widget waits
for the viewer to click, drag or press a button (the map's turbines turn while
its slide is shown, and a curve eases to its new place after a click). The
scoreboard alone asks Reveal to re-centre its slide, because it fills in after
the slide has been laid out.

    import deck10_widgets as W
    print(W.emit(W.explorer(DATA["explorer"])))       # inside a  #| output: asis  cell

Colours are the course palette: deep blue #0d366b, blue #1c5cab, light blue
#86b6ef, amber #b8860b, brick #b3402f, greys #c3c2b7 / #898781 / #52514e.
Throughout, red (brick) means higher than the day before and blue means not.
"""
import json

HELPERS = r"""
var NS = "http://www.w3.org/2000/svg";
function el(tag, attrs, parent, text) { var e = document.createElementNS(NS, tag); for (var k in attrs) e.setAttribute(k, attrs[k]); if (text !== undefined) e.textContent = text; if (parent) parent.appendChild(e); return e; }
function txt(parent, x, y, s, o) { o = o || {}; return el("text", { x: x, y: y, "font-size": o.size || 12.5, "font-weight": o.weight || 400, fill: o.fill || "#898781", "text-anchor": o.anchor || "start", "font-family": "Segoe UI, sans-serif", transform: o.rotate ? "rotate(" + o.rotate + " " + x + " " + y + ")" : "", stroke: o.halo ? "#fcfcfb" : "none", "stroke-width": o.halo ? 4 : 0, "paint-order": "stroke" }, parent, s); }
function onShow(node, fn) { var sec = node.closest("section"); if (!sec) { fn(); return; } var was = false; function check() { var now = sec.classList.contains("present"); if (now && !was) fn(); was = now; } new MutationObserver(check).observe(sec, { attributes: true, attributeFilter: ["class"] }); check(); }
function visible(node) { var sec = node.closest("section"); return !sec || sec.classList.contains("present"); }
function lerp(a, b, t) { return a + (b - a) * t; }
function ramp(stops, t) { t = Math.max(0, Math.min(1, t)); var n = stops.length - 1, i = Math.min(n - 1, Math.floor(t * n)), u = t * n - i, a = stops[i], b = stops[i + 1]; return "rgb(" + Math.round(lerp(a[0], b[0], u)) + "," + Math.round(lerp(a[1], b[1], u)) + "," + Math.round(lerp(a[2], b[2], u)) + ")"; }
var BLUES = [[241, 245, 250], [134, 182, 239], [28, 92, 171], [13, 54, 107]];
var AMBERS = [[240, 222, 170], [184, 134, 11]];
var BRICKS = [[251, 241, 240], [224, 160, 149], [179, 64, 47]];
function priceColour(v) { if (v === null || v === undefined) return "#ffffff"; if (v < 0) return ramp(AMBERS, Math.min(1, -v / 300)); return ramp(BLUES, Math.min(1, v / 1500)); }
function fmt(n) { var s = Math.round(Math.abs(n)).toLocaleString("en-GB"); return (n < -0.5 ? "−" : "") + s; }
function hh(h) { return (h < 10 ? "0" : "") + h + ":00"; }
function niceTicks(lo, hi, n) { var span = hi - lo, raw = span / Math.max(1, n), mag = Math.pow(10, Math.floor(Math.log10(raw))), step = [1, 2, 2.5, 5, 10].map(function (m) { return m * mag; }).filter(function (s) { return s >= raw; })[0]; var out = []; for (var v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(Math.round(v * 1000) / 1000); return out; }
function store(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch (e) { window["__" + key] = value; } return value; }
function recall(key) { try { var v = localStorage.getItem(key); if (v) return JSON.parse(v); } catch (e) {} return window["__" + key] || null; }
function hidpi(canvas, w, h) { var r = 2; canvas.width = w * r; canvas.height = h * r; canvas.style.width = w + "px"; canvas.style.height = h + "px"; var c = canvas.getContext("2d"); c.setTransform(r, 0, 0, r, 0, 0); return c; }
var MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
var WEEKDAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
function longDate(iso) { var d = new Date(iso + "T12:00:00"); return WEEKDAYS[d.getDay()] + " " + d.getDate() + " " + MONTHS[d.getMonth()] + " " + d.getFullYear(); }
var WDNAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
function dayMonth(iso) { var d = new Date(iso + "T12:00:00"); return WEEKDAYS[d.getDay()] + " " + d.getDate() + " " + MONTHS[d.getMonth()]; }
function signed(v) { return (v > 0 ? "+" : "") + fmt(v); }
function diverge(v, span) { if (v === null || v === undefined) return "#ffffff"; var t = Math.max(-1, Math.min(1, v / span)); return t < 0 ? ramp([[241, 240, 236], [134, 182, 239], [28, 92, 171]], -t) : ramp([[241, 240, 236], [224, 160, 149], [179, 64, 47]], t); }
"""

BASE_CSS = r"""
.w-btn { font: 600 0.54em "Segoe UI", sans-serif; color: #0d366b; background: #ffffff; border: 1.5px solid #86b6ef; border-radius: 7px; padding: 0 0.75em; height: 2em; box-sizing: border-box; display: inline-flex; align-items: center; justify-content: center; line-height: 1; cursor: pointer; white-space: nowrap; vertical-align: middle; }
.w-marks { display: inline-flex; align-items: center; gap: 0.45em; flex-wrap: wrap; }
.w-btn:hover { border-color: #1c5cab; }
.w-btn.on { background: #0d366b; color: #ffffff; border-color: #0d366b; }
.w-read { font: 0.54em "Segoe UI", sans-serif; color: #52514e; }
.w-read b, .w-val { font-family: "Cascadia Mono", Consolas, monospace; color: #0d366b; font-weight: 600; }
.w-range { accent-color: #1c5cab; }
.w-tip { position: absolute; pointer-events: none; background: #ffffff; border: 1px solid #c3c2b7; border-radius: 6px; padding: 4px 8px; font: 13px "Segoe UI", sans-serif; color: #141413; box-shadow: 0 2px 6px rgba(0,0,0,0.08); white-space: nowrap; display: none; z-index: 5; }
"""


def emit(html):
    """Wrap the widget in a raw HTML block, so pandoc passes it through untouched
    (without it, the text between tags is read as markdown and wrapped in <p>)."""
    body = "\n".join(line for line in html.strip("\n").split("\n") if line.strip() != "")
    return "```{=html}\n" + body + "\n```"


def _fill(template, **values):
    out = template.replace("__HELPERS__", HELPERS).replace("__BASECSS__", BASE_CSS)
    for key, value in values.items():
        out = out.replace("__" + key.upper() + "__", value if isinstance(value, str)
                          else json.dumps(value, separators=(",", ":")))
    return out


# ============================================================ 2. the map
MAP = r"""
<style>__BASECSS__
.w-map { display: flex; gap: 24px; align-items: flex-start; }
.w-map-panel { flex: 1 1 0; min-width: 0; }
.w-map-days { display: flex; gap: 0.5em; margin-bottom: 0.3em; }
.w-map-row { display: flex; align-items: center; gap: 0.6em; margin: 0.2em 0; }
.w-map-row input { flex: 1 1 0; }
.w-map-big { font: 700 0.8em "Cascadia Mono", Consolas, monospace; color: #0d366b; min-width: 3.1em; }
.w-map-legend { font: 13px "Segoe UI", sans-serif; color: #52514e; margin-top: 2px; }
.map-flow { animation: mapdash linear infinite; }
@keyframes mapdash { to { stroke-dashoffset: -32; } }
</style>
<div class="w-map" id="map">
<svg id="map-svg" width="520" height="__DISPH__" viewBox="0 0 __WIDTH__ __HEIGHT__" role="img" aria-label="Denmark's two price areas coloured by the hour's price, with the power flowing on each cable abroad"></svg>
<div class="w-map-panel">
<div class="w-map-days" id="map-days"></div>
<div class="w-map-row"><span class="w-map-big" id="map-hour">00:00</span><input class="w-range" type="range" id="map-range" min="0" max="23" value="0"><button class="w-btn" id="map-play">play the day</button></div>
<div class="w-read" id="map-read"></div>
<svg id="map-day" width="560" height="176" role="img" aria-label="The day's hourly price in West Denmark with the chosen hour marked"></svg>
<svg id="map-key" width="560" height="40" role="img" aria-label="Colour key for the price"></svg>
<div class="w-map-legend">Arrows: power on each cable, MW, pointing the way it flows. Cables drawn schematically.</div>
</div>
</div>
<script>
(function () {
__HELPERS__
  var M = __DATA__;
  var svg = document.getElementById("map-svg"); if (!svg) return;
  var day = 0, hour = 0, playing = false, timer = null, spin = [0, 0, 0], lastT = 0;
  var defs = el("defs", {}, svg);
  var mk = el("marker", { id: "map-head", viewBox: "0 0 10 10", refX: 6, refY: 5, markerWidth: 15, markerHeight: 15, markerUnits: "userSpaceOnUse", orient: "auto" }, defs);
  el("path", { d: "M0,0 L10,5 L0,10 z", fill: "#3d3c39" }, mk);
  el("rect", { x: 0, y: 0, width: M.width, height: M.height, fill: "#dce5ef" }, svg);
  Object.keys(M.shapes).forEach(function (k) { if (k !== "DK1" && k !== "DK2") el("path", { d: M.shapes[k], fill: "#eeede8", stroke: "#cfcdc4", "stroke-width": 0.8 }, svg); });
  var dk1 = el("path", { d: M.shapes.DK1, stroke: "#3d3c39", "stroke-width": 1.1 }, svg);
  var dk2 = el("path", { d: M.shapes.DK2, stroke: "#3d3c39", "stroke-width": 1.1 }, svg);
  var lines = {}, tags = {};
  Object.keys(M.cables).forEach(function (k) {
    lines[k] = el("line", { stroke: "#3d3c39", "stroke-linecap": "butt", "stroke-dasharray": "10 6", "marker-end": "url(#map-head)", class: "map-flow" }, svg);
  });
  var rotors = M.turbines.map(function (p) {
    el("line", { x1: p[0], y1: p[1], x2: p[0], y2: p[1] + 28, stroke: "#52514e", "stroke-width": 2.4 }, svg);
    var g = el("g", {}, svg);
    [0, 120, 240].forEach(function (a) { var r = a * Math.PI / 180; el("line", { x1: p[0], y1: p[1], x2: p[0] + 17 * Math.sin(r), y2: p[1] - 17 * Math.cos(r), stroke: "#0d366b", "stroke-width": 2.6, "stroke-linecap": "round" }, g); });
    el("circle", { cx: p[0], cy: p[1], r: 2.8, fill: "#0d366b" }, svg);
    return { g: g, x: p[0], y: p[1] };
  });
  Object.keys(M.cables).forEach(function (k) { tags[k] = txt(svg, 0, 0, "", { size: 15, fill: "#141413", anchor: "middle", halo: true, weight: 600 }); });
  Object.keys(M.labels).forEach(function (k) {
    var dk = k.indexOf("Denmark") >= 0, p = M.labels[k];
    txt(svg, p[0], p[1], k, { size: dk ? 15 : 14, weight: dk ? 700 : 400, fill: dk ? "#0d366b" : "#898781", anchor: "middle", halo: dk });
  });
  var names = ["Windy: " + longDate(M.days[0].date).split(" ").slice(0, 3).join(" "), "Still: " + longDate(M.days[1].date).split(" ").slice(0, 3).join(" ")];
  var bar = document.getElementById("map-days");
  names.forEach(function (n, i) { var b = document.createElement("button"); b.className = "w-btn"; b.textContent = n; b.addEventListener("click", function () { day = i; render(); }); bar.appendChild(b); });
  (function key() {
    var k = document.getElementById("map-key"); if (!k) return;
    for (var i = 0; i < 60; i++) el("rect", { x: 80 + i * 5.5, y: 4, width: 5.8, height: 13, fill: priceColour(i * 25) }, k);
    el("rect", { x: 8, y: 4, width: 50, height: 13, fill: priceColour(-200) }, k);
    txt(k, 33, 34, "below 0", { anchor: "middle", fill: "#52514e" });
    [0, 500, 1000, 1500].forEach(function (v) { txt(k, 80 + v / 25 * 5.5, 34, fmt(v) + (v === 1500 ? "+" : ""), { anchor: "middle", fill: "#52514e" }); });
    txt(k, 425, 15, "kr per MWh", { fill: "#52514e" });
  })();
  function dayChart() {
    var g = document.getElementById("map-day"); while (g.firstChild) g.removeChild(g.firstChild);
    var d = M.days[day], v = d.dk1, lo = Math.min(0, Math.min.apply(null, v)), hi = Math.max(200, Math.max.apply(null, v)) * 1.08;
    var L = 56, R = 540, T = 10, B = 128;
    function X(h) { return L + h / 23 * (R - L); } function Y(q) { return B - (q - lo) / (hi - lo) * (B - T); }
    niceTicks(lo, hi, 3).forEach(function (q) { el("line", { x1: L, y1: Y(q), x2: R, y2: Y(q), stroke: q === 0 ? "#c3c2b7" : "#ecebe6" }, g); txt(g, L - 7, Y(q) + 4, fmt(q), { anchor: "end" }); });
    el("polyline", { points: v.map(function (q, h) { return X(h) + "," + Y(q); }).join(" "), fill: "none", stroke: "#1c5cab", "stroke-width": 2.4, "stroke-linejoin": "round" }, g);
    el("line", { x1: X(hour), y1: T, x2: X(hour), y2: B, stroke: "#b3402f", "stroke-width": 1.6 }, g);
    el("circle", { cx: X(hour), cy: Y(v[hour]), r: 5.5, fill: "#b3402f" }, g);
    [0, 6, 12, 18].forEach(function (h) { txt(g, X(h), B + 18, hh(h), { anchor: "middle" }); });
    txt(g, X(23), B + 18, hh(23), { anchor: "end" });
    txt(g, L, B + 40, "West Denmark through the day, kr per MWh", { fill: "#52514e" });
  }
  function render() {
    var d = M.days[day];
    Array.prototype.forEach.call(bar.children, function (b, i) { b.classList.toggle("on", i === day); });
    dk1.setAttribute("fill", priceColour(d.dk1[hour])); dk2.setAttribute("fill", priceColour(d.dk2[hour]));
    Object.keys(M.cables).forEach(function (k) {
      var f = d.flows[k][hour], c = M.cables[k], ln = lines[k], a = f >= 0 ? c.from : c.to, b = f >= 0 ? c.to : c.from;
      var w = Math.abs(f), dx = b[0] - a[0], dy = b[1] - a[1], len = Math.sqrt(dx * dx + dy * dy);
      var ex = b[0] - dx / len * 9, ey = b[1] - dy / len * 9;
      ln.setAttribute("x1", a[0]); ln.setAttribute("y1", a[1]); ln.setAttribute("x2", ex); ln.setAttribute("y2", ey);
      ln.setAttribute("stroke-width", (2 + w / 1700 * 7).toFixed(2));
      ln.style.display = w < 20 ? "none" : "";
      ln.style.animationDuration = Math.max(0.35, 2.6 - w / 800).toFixed(2) + "s";
      tags[k].setAttribute("x", c.tag[0]); tags[k].setAttribute("y", c.tag[1] + 4);
      tags[k].textContent = w < 20 ? "" : fmt(w);
    });
    document.getElementById("map-hour").textContent = hh(hour);
    document.getElementById("map-range").value = hour;
    document.getElementById("map-read").innerHTML = "West Denmark <b>" + fmt(d.dk1[hour]) + "</b> kr per MWh &middot; East Denmark <b>" + fmt(d.dk2[hour]) + "</b> &middot; wind forecast <b>" + fmt(d.wind[hour]) + "</b> MW";
    dayChart();
  }
  function spinLoop(t) {
    var dt = lastT ? (t - lastT) / 1000 : 0; lastT = t;
    var speed = 0.4 + 7 * M.days[day].wind[hour] / M.wind_max;
    rotors.forEach(function (r, i) { spin[i] = (spin[i] + speed * dt * 57.3) % 360; r.g.setAttribute("transform", "rotate(" + spin[i].toFixed(1) + " " + r.x + " " + r.y + ")"); });
    if (visible(svg)) requestAnimationFrame(spinLoop); else lastT = 0;
  }
  document.getElementById("map-range").addEventListener("input", function (e) { hour = parseInt(e.target.value, 10); render(); });
  document.getElementById("map-play").addEventListener("click", function () {
    var btn = this;
    if (playing) { clearInterval(timer); playing = false; btn.textContent = "play the day"; return; }
    playing = true; btn.textContent = "pause"; if (hour === 23) hour = -1;
    timer = setInterval(function () { hour++; if (hour >= 23) { hour = 23; clearInterval(timer); playing = false; btn.textContent = "play the day"; } render(); }, 650);
  });
  render();
  onShow(svg, function () { lastT = 0; requestAnimationFrame(spinLoop); });
})();
</script>
"""


def power_map(data):
    return _fill(MAP, data=data, width=str(data["width"]), height=str(data["height"]),
                 disph=str(round(520 * data["height"] / data["width"])))


# ============================================================ 3. merit order
MERIT = r"""
<style>__BASECSS__
.w-merit-ctl { display: flex; gap: 2em; align-items: center; margin-bottom: 0.15em; }
.w-merit-ctl label { font: 600 0.54em "Segoe UI", sans-serif; color: #0d366b; display: flex; align-items: center; gap: 0.5em; }
.w-merit-ctl input { width: 11em; }
.w-merit-out { font: 600 0.62em "Segoe UI", sans-serif; color: #b3402f; margin: 0.1em 0 0; }
.w-merit-out b { font-family: "Cascadia Mono", Consolas, monospace; }
</style>
<div id="merit">
<div class="w-merit-ctl"><label>wind forecast <input class="w-range" type="range" id="merit-wind" min="0" max="4500" step="50" value="1300"> <span class="w-val" id="merit-wv"></span></label><label>demand <input class="w-range" type="range" id="merit-dem" min="2000" max="4800" step="50" value="3400"> <span class="w-val" id="merit-dv"></span></label></div>
<div class="w-merit-out" id="merit-out"></div>
<svg id="merit-svg" width="1040" height="356" viewBox="0 0 1040 356" role="img" aria-label="A stylised supply curve of blocks priced from cheapest to most expensive, with demand as a vertical line and the clearing price where they meet"></svg>
</div>
<script>
(function () {
__HELPERS__
  var svg = document.getElementById("merit-svg"); if (!svg) return;
  var BLOCKS = [
    { name: "wind and sun", bid: 0, mw: 0, fill: "#86b6ef" },
    { name: "heat plants", bid: 200, mw: 900, fill: "#c3c2b7" },
    { name: "hydro imports", bid: 450, mw: 1500, fill: "#1c5cab" },
    { name: "gas", bid: 900, mw: 1500, fill: "#b8860b" },
    { name: "oil", bid: 2500, mw: 1000, fill: "#b3402f" }];
  var L = 90, R = 1020, T = 26, B = 300, XMAX = 9500, YMIN = -250, YMAX = 2700;
  function X(mw) { return L + mw / XMAX * (R - L); } function Y(p) { return B - (p - YMIN) / (YMAX - YMIN) * (B - T); }
  function draw() {
    var w = +document.getElementById("merit-wind").value, d = +document.getElementById("merit-dem").value;
    document.getElementById("merit-wv").textContent = fmt(w) + " MW"; document.getElementById("merit-dv").textContent = fmt(d) + " MW";
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    [0, 500, 1000, 1500, 2000, 2500].forEach(function (p) { el("line", { x1: L, y1: Y(p), x2: R, y2: Y(p), stroke: p === 0 ? "#c3c2b7" : "#ecebe6" }, svg); txt(svg, L - 8, Y(p) + 4, fmt(p), { size: 13, anchor: "end" }); });
    [0, 2000, 4000, 6000, 8000].forEach(function (m) { txt(svg, X(m), B + 22, fmt(m), { size: 13, anchor: "middle" }); });
    txt(svg, (L + R) / 2, B + 46, "MW offered, cheapest first", { size: 13, anchor: "middle", fill: "#52514e" });
    txt(svg, 22, (T + B) / 2, "bid, kr per MWh", { size: 13, anchor: "middle", fill: "#52514e", rotate: -90 });
    var start = 0, price = null, setter = null;
    BLOCKS.forEach(function (b, i) {
      var mw = i === 0 ? w : b.mw, x0 = X(start), x1 = X(start + mw), top = b.bid === 0 ? Y(0) - 7 : Y(b.bid);
      if (mw > 0) {
        el("rect", { x: x0 + 1, y: top, width: Math.max(0, x1 - x0 - 2), height: Math.max(3, Y(0) - top), fill: b.fill, rx: 2 }, svg);
        if (x1 - x0 > b.name.length * 7.2 + 8) txt(svg, (x0 + x1) / 2, top - 8, b.name, { size: 13, anchor: "middle", fill: "#141413" });
      }
      if (price === null && start + mw >= d) { price = b.bid; setter = b.name; }
      start += mw;
    });
    el("line", { x1: X(d), y1: T - 14, x2: X(d), y2: B, stroke: "#141413", "stroke-width": 2, "stroke-dasharray": "6 4" }, svg);
    txt(svg, X(d) + 7, T - 4, "demand", { size: 13, weight: 600, fill: "#141413" });
    el("line", { x1: L, y1: Y(price), x2: X(d), y2: Y(price), stroke: "#b3402f", "stroke-width": 2.2 }, svg);
    el("circle", { cx: X(d), cy: Y(price), r: 6.5, fill: "#b3402f", stroke: "#fcfcfb", "stroke-width": 2 }, svg);
    document.getElementById("merit-out").innerHTML = w >= d ? "Price <b>0</b> kr per MWh or below: wind and sun alone cover demand" : "Price <b>" + fmt(price) + "</b> kr per MWh for every MWh sold, set by " + setter;
  }
  ["merit-wind", "merit-dem"].forEach(function (id) { document.getElementById(id).addEventListener("input", draw); });
  draw();
})();
</script>
"""


def merit():
    return _fill(MERIT)


# ============================================================ 8. the crisis toggle
CRISIS = r"""
<style>__BASECSS__
.w-cr { display: flex; gap: 30px; align-items: flex-start; }
.w-cr table { border-collapse: collapse; font: 16px "Segoe UI", sans-serif; margin-top: 10px; }
.w-cr td { padding: 7px 12px; border-bottom: 1px solid #ecebe6; color: #52514e; }
.w-cr td.n { text-align: right; font-family: "Cascadia Mono", Consolas, monospace; color: #0d366b; font-weight: 600; }
</style>
<div id="crisis">
<div style="display:flex;gap:0.6em;margin-bottom:0.15em"><button class="w-btn on" id="crisis-a">trained on 2022 and 2023</button><button class="w-btn" id="crisis-b">trained on 2023 only</button><button class="w-btn" id="crisis-c">trained on 2022 only</button></div>
<div class="w-cr">
<svg id="crisis-svg" width="450" height="396" viewBox="0 0 450 396" role="img" aria-label="For 2024's hours grouped by predicted probability, the average prediction against the share that happened, with the diagonal where they would agree"></svg>
<table id="crisis-tab" data-quarto-disable-processing="true"></table>
</div>
</div>
<script>
(function () {
__HELPERS__
  var D = __DATA__;
  var svg = document.getElementById("crisis-svg"); if (!svg) return;
  var L = 64, R = 436, T = 12, B = 346, pts = null, anim = null;
  function X(p) { return L + p * (R - L); } function Y(p) { return B - p * (B - T); }
  function base() {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    [0, 0.25, 0.5, 0.75, 1].forEach(function (p) { el("line", { x1: L, y1: Y(p), x2: R, y2: Y(p), stroke: "#ecebe6" }, svg); txt(svg, L - 8, Y(p) + 4, p.toFixed(2), { size: 13, anchor: "end" }); txt(svg, X(p), B + 20, p.toFixed(2), { size: 13, anchor: "middle" }); });
    el("line", { x1: X(0), y1: Y(0), x2: X(1), y2: Y(1), stroke: "#898781", "stroke-dasharray": "6 5" }, svg);
    txt(svg, X(0.58), Y(0.66), "predicted = actual", { size: 12.5, rotate: -42 });
    txt(svg, (L + R) / 2, B + 42, "the model's probability, average in the group", { size: 13, anchor: "middle", fill: "#52514e" });
    txt(svg, 16, (T + B) / 2, "share that went above 1,000 kr", { size: 13, anchor: "middle", fill: "#52514e", rotate: -90 });
  }
  function target(k) { return D[k].buckets.filter(function (b) { return b.n > 0; }).map(function (b) { return [b.said, b.happened, b.n]; }); }
  function draw() {
    base();
    el("polyline", { points: pts.map(function (q) { return X(q[0]) + "," + Y(q[1]); }).join(" "), fill: "none", stroke: "#b3402f", "stroke-width": 2.4 }, svg);
    pts.forEach(function (q) { el("circle", { cx: X(q[0]), cy: Y(q[1]), r: 4 + Math.sqrt(q[2]) / 7, fill: "#b3402f", "fill-opacity": 0.85, stroke: "#fcfcfb", "stroke-width": 2 }, svg); });
  }
  function go(k) {
    var to = target(k), from = pts || to, t0 = null; if (anim) cancelAnimationFrame(anim);
    while (from.length < to.length) from.push(from[from.length - 1]);
    function step(t) { if (t0 === null) t0 = t; var u = Math.min(1, (t - t0) / 900); pts = to.map(function (q, i) { var f = from[Math.min(i, from.length - 1)]; return [lerp(f[0], q[0], u), lerp(f[1], q[1], u), lerp(f[2], q[2], u)]; }); draw(); if (u < 1) anim = requestAnimationFrame(step); }
    anim = requestAnimationFrame(step);
    var d = D[k], row = function (a, b) { return "<tr><td>" + a + "</td><td class='n'>" + b + "</td></tr>"; };
    document.getElementById("crisis-tab").innerHTML = row("hours above 1,000 kr in the training years", (100 * d.base).toFixed(1) + "%") + row("the model's average probability for 2024", (100 * d.mean_p).toFixed(1) + "%") + row("hours above 1,000 kr in 2024", (100 * d.actual).toFixed(1) + "%");
    document.getElementById("crisis-a").classList.toggle("on", k === "crisis"); document.getElementById("crisis-b").classList.toggle("on", k === "calm"); document.getElementById("crisis-c").classList.toggle("on", k === "y2022");
  }
  document.getElementById("crisis-a").addEventListener("click", function () { go("crisis"); });
  document.getElementById("crisis-b").addEventListener("click", function () { go("calm"); });
  document.getElementById("crisis-c").addEventListener("click", function () { go("y2022"); });
  go("crisis");
})();
</script>
"""


def crisis(data):
    return _fill(CRISIS, data=data["crisis"])


# ============================================================ the shape of a day, month by month
MONTHLY = r"""
<style>__BASECSS__
.w-mo-top { display: flex; align-items: center; gap: 0.5em; margin-bottom: 0.1em; }
.w-mo-name { font: 700 0.66em "Segoe UI", sans-serif; color: #0d366b; min-width: 8.6em; text-align: center; }
.w-mo-top input { width: 15em; }
</style>
<div id="mo">
<div class="w-mo-top"><button class="w-btn" id="mo-prev" aria-label="previous month">&#8249;</button><span class="w-mo-name" id="mo-name"></span><button class="w-btn" id="mo-next" aria-label="next month">&#8250;</button><input class="w-range" type="range" id="mo-range" min="0" max="__NMAX__" value="__START__" aria-label="the month"><span class="w-marks" id="mo-types" style="margin-left: 1em"></span></div>
<svg id="mo-price" width="1040" height="206" viewBox="0 0 1040 206" role="img" aria-label="The month's average price at each hour of the day"></svg>
<svg id="mo-sun" width="1040" height="124" viewBox="0 0 1040 124" role="img" aria-label="The month's average sun forecast at each hour of the day"></svg>
</div>
<script>
(function () {
__HELPERS__
  var D = __DATA__, i = __START__, TYPE = "";
  var sp = document.getElementById("mo-price"); if (!sp) return;
  function nn(q) { return q !== null && q !== undefined; }
  var SMAX = 0; ["solar", "solar_wd", "solar_we"].forEach(function (k) { D[k].forEach(function (r) { r.forEach(function (q) { if (nn(q)) SMAX = Math.max(SMAX, q); }); }); }); SMAX *= 1.1;
  function X(h) { return 70 + (h + 0.5) / 24 * 960; }
  function draw() {
    var v = D["price" + TYPE][i], s = D["solar" + TYPE][i], all3 = D.price[i].concat(D.price_wd[i], D.price_we[i]).filter(nn);
    while (sp.firstChild) sp.removeChild(sp.firstChild);
    var lo = Math.min(0, Math.min.apply(null, all3)), hi = Math.max.apply(null, all3) * 1.1, T = 26, B = 200;
    function Y(q) { return B - (q - lo) / (hi - lo) * (B - T); }
    niceTicks(lo, hi, 4).forEach(function (q) { el("line", { x1: 70, y1: Y(q), x2: 1030, y2: Y(q), stroke: q === 0 ? "#c3c2b7" : "#ecebe6" }, sp); txt(sp, 62, Y(q) + 4, fmt(q), { anchor: "end" }); });
    v.forEach(function (q, h) { if (!nn(q)) return; var top = Math.min(Y(q), Y(0)), ht = Math.abs(Y(q) - Y(0)); el("rect", { x: X(h) - 15, y: top, width: 30, height: Math.max(1.5, ht), rx: 2, fill: "#1c5cab" }, sp); });
    txt(sp, 70, 14, "the month's average price at each hour, kr per MWh", { fill: "#52514e", size: 13.5, weight: 600 });
    var ss = document.getElementById("mo-sun"); while (ss.firstChild) ss.removeChild(ss.firstChild);
    var T2 = 24, B2 = 96; function Y2(q) { return B2 - q / SMAX * (B2 - T2); }
    [0, 500, 1000, 1500].forEach(function (q) { if (q > SMAX) return; el("line", { x1: 70, y1: Y2(q), x2: 1030, y2: Y2(q), stroke: q === 0 ? "#c3c2b7" : "#ecebe6" }, ss); txt(ss, 62, Y2(q) + 4, fmt(q), { anchor: "end" }); });
    var d = "M" + X(0) + "," + Y2(0);
    s.forEach(function (q, h) { d += " L" + X(h).toFixed(1) + "," + Y2(nn(q) ? q : 0).toFixed(1); });
    d += " L" + X(23) + "," + Y2(0) + " Z";
    el("path", { d: d, fill: "#b8860b", "fill-opacity": 0.28, stroke: "#b8860b", "stroke-width": 1.8 }, ss);
    [0, 6, 12, 18, 23].forEach(function (h) { txt(ss, X(h), B2 + 18, hh(h), { anchor: "middle" }); });
    txt(ss, 70, 14, "the month's average sun forecast at each hour, MW", { fill: "#52514e", size: 13.5, weight: 600 });
    document.getElementById("mo-name").textContent = D.labels[i];
    Array.prototype.forEach.call(document.getElementById("mo-types").children, function (b) { b.classList.toggle("on", b.dataset.t === TYPE); });
    document.getElementById("mo-range").value = i;
  }
  document.getElementById("mo-prev").addEventListener("click", function () { i = Math.max(0, i - 1); draw(); });
  document.getElementById("mo-next").addEventListener("click", function () { i = Math.min(D.labels.length - 1, i + 1); draw(); });
  document.getElementById("mo-range").addEventListener("input", function (e) { i = +e.target.value; draw(); });
  [["", "all days"], ["_wd", "weekdays"], ["_we", "weekends"]].forEach(function (t) { var b = document.createElement("button"); b.className = "w-btn"; b.dataset.t = t[0]; b.textContent = t[1]; b.addEventListener("click", function () { TYPE = t[0]; draw(); }); document.getElementById("mo-types").appendChild(b); });
  draw();
})();
</script>
"""


def monthly(data, start=24):
    return _fill(MONTHLY, data=data, nmax=str(len(data["labels"]) - 1), start=str(start))


# ============================================================ 1. two years of prices, one day at a time
EXPLORER = r"""
<style>__BASECSS__
.w-ex-ctl { display: flex; align-items: center; gap: 0.45em; margin: 0.12em 0; }
.w-ex-ctl input { flex: 1 1 0; }
.w-ex-date { font: 700 0.6em "Segoe UI", sans-serif; color: #0d366b; }
</style>
<div id="ex">
<svg id="ex-line" width="1040" height="138" viewBox="0 0 1040 138" role="img" aria-label="Each day's average price in West Denmark, 2022 and 2023, with the chosen day marked" style="cursor: crosshair"></svg>
<div class="w-ex-ctl"><button class="w-btn" id="ex-prev" aria-label="the day before">&#9664;</button><input class="w-range" type="range" id="ex-range" min="0" max="__NMAX__" value="0" aria-label="the day"><button class="w-btn" id="ex-next" aria-label="the day after">&#9654;</button><span class="w-marks" id="ex-marks"></span></div>
<div class="w-ex-ctl"><span class="w-ex-date" id="ex-date"></span><span class="w-read" id="ex-read"></span></div>
<svg id="ex-day" width="1040" height="176" viewBox="0 0 1040 176" role="img" aria-label="The chosen day's 24 hourly prices"></svg>
</div>
<script>
(function () {
__HELPERS__
  var D = __DATA__;
  var line = document.getElementById("ex-line"); if (!line) return;
  var N = D.days.length, sel = Math.max(0, D.days.indexOf(D.start)), L = 70, R = 1030, lo = -600, hi = 5600, T = 8, B = 110;
  function X(i) { return L + i / (N - 1) * (R - L); } function Y(v) { return B - (v - lo) / (hi - lo) * (B - T); }
  var g0 = el("g", {}, line);
  [0, 2000, 4000].forEach(function (v) { el("line", { x1: L, y1: Y(v), x2: R, y2: Y(v), stroke: v === 0 ? "#c3c2b7" : "#ecebe6" }, g0); txt(g0, L - 8, Y(v) + 4, fmt(v), { anchor: "end" }); });
  [["2022", 0], ["2023", D.days.indexOf("2023-01-01")]].forEach(function (y) { el("line", { x1: X(y[1]), y1: B, x2: X(y[1]), y2: B + 5, stroke: "#c3c2b7" }, g0); txt(g0, X(y[1]) + 3, B + 19, y[0]); });
  txt(g0, 16, (T + B) / 2, "day's average, kr", { anchor: "middle", rotate: -90, fill: "#52514e" });
  el("polyline", { points: D.means.map(function (v, i) { return X(i).toFixed(1) + "," + Y(v).toFixed(1); }).join(" "), fill: "none", stroke: "#1c5cab", "stroke-width": 1.1, "stroke-linejoin": "round" }, g0);
  var mk = el("g", {}, line);
  function day() {
    var g = document.getElementById("ex-day"); while (g.firstChild) g.removeChild(g.firstChild);
    var v = D.curves[sel], ok = v.filter(function (q) { return q !== null; });
    var mn = Math.min.apply(null, ok), mx = Math.max.apply(null, ok), avg = ok.reduce(function (a, b) { return a + b; }, 0) / ok.length;
    var dlo = Math.min(0, mn) * 1.15, dhi = Math.max(mx * 1.1, 60), L2 = 70, R2 = 1030, T2 = 8, B2 = 150;
    function X2(h) { return L2 + (h + 0.5) / 24 * (R2 - L2); } function Y2(q) { return B2 - (q - dlo) / (dhi - dlo) * (B2 - T2); }
    niceTicks(dlo, dhi, 4).forEach(function (q) { el("line", { x1: L2, y1: Y2(q), x2: R2, y2: Y2(q), stroke: q === 0 ? "#c3c2b7" : "#ecebe6" }, g); txt(g, L2 - 8, Y2(q) + 4, fmt(q), { anchor: "end" }); });
    v.forEach(function (q, h) { if (q === null) return; var top = Math.min(Y2(q), Y2(0)), ht = Math.abs(Y2(q) - Y2(0)); el("rect", { x: X2(h) - 14, y: top, width: 28, height: Math.max(1.5, ht), rx: 2, fill: "#1c5cab" }, g); });
    [0, 6, 12, 18, 23].forEach(function (h) { txt(g, X2(h), B2 + 18, hh(h), { anchor: "middle" }); });
    txt(g, 16, (T2 + B2) / 2, "kr per MWh", { anchor: "middle", rotate: -90, fill: "#52514e" });
    document.getElementById("ex-date").textContent = longDate(D.days[sel]);
    document.getElementById("ex-read").innerHTML = "&nbsp; average <b>" + fmt(avg) + "</b> kr per MWh";
  }
  function draw() {
    while (mk.firstChild) mk.removeChild(mk.firstChild);
    el("line", { x1: X(sel), y1: T - 2, x2: X(sel), y2: B, stroke: "#b3402f", "stroke-width": 1.5 }, mk);
    el("circle", { cx: X(sel), cy: Y(D.means[sel]), r: 5, fill: "#b3402f", stroke: "#fcfcfb", "stroke-width": 1.5 }, mk);
    document.getElementById("ex-range").value = sel;
    day();
  }
  function pick(e) { var r = line.getBoundingClientRect(), x = (e.clientX - r.left) / r.width * 1040; sel = Math.max(0, Math.min(N - 1, Math.round((x - L) / (R - L) * (N - 1)))); draw(); }
  var down = false;
  line.addEventListener("mousedown", function (e) { down = true; pick(e); e.preventDefault(); });
  window.addEventListener("mouseup", function () { down = false; });
  line.addEventListener("mousemove", function (e) { if (down) pick(e); });
  document.getElementById("ex-range").addEventListener("input", function (e) { sel = +e.target.value; draw(); });
  document.getElementById("ex-prev").addEventListener("click", function () { sel = Math.max(0, sel - 1); draw(); });
  document.getElementById("ex-next").addEventListener("click", function () { sel = Math.min(N - 1, sel + 1); draw(); });
  var bar = document.getElementById("ex-marks");
  Object.keys(D.marks).forEach(function (name) { var b = document.createElement("button"); b.className = "w-btn"; b.textContent = name; b.addEventListener("click", function () { sel = Math.max(0, D.days.indexOf(D.marks[name])); draw(); }); bar.appendChild(b); bar.appendChild(document.createTextNode(" ")); });
  draw();
})();
</script>
"""


def explorer(data, start="2022-08-26"):
    payload = dict(data)
    payload["start"] = start
    return _fill(EXPLORER, data=payload, nmax=str(len(data["days"]) - 1))


# ============================================================ today and tomorrow, 2024: two lines and the gap between them
TODAY = r"""
<style>__BASECSS__
.w-tt-top { display: flex; align-items: center; gap: 0.45em; margin-bottom: 0.1em; }
.w-tt-date { font: 700 0.6em "Segoe UI", sans-serif; color: #0d366b; min-width: 16.4em; text-align: center; white-space: nowrap; }
.w-tt-top input { flex: 1 1 0; }
</style>
<div id="tt">
<div class="w-tt-top"><button class="w-btn" id="tt-prev" aria-label="the day before">&#9664;</button><span class="w-tt-date" id="tt-date"></span><button class="w-btn" id="tt-next" aria-label="the day after">&#9654;</button><input class="w-range" type="range" id="tt-range" min="0" max="1" value="0" aria-label="the day"></div>
<div class="w-tt-top" id="tt-marks"></div>
<svg id="tt-price" width="1040" height="176" viewBox="0 0 1040 176" role="img" aria-label="Today's and tomorrow's price by hour, red between the lines where tomorrow's is higher and blue where it is lower"></svg>
<svg id="tt-wind" width="1040" height="170" viewBox="0 0 1040 170" role="img" aria-label="Today's and tomorrow's wind forecast by hour, blue between the lines where tomorrow has more wind and red where it has less"></svg>
</div>
<script>
(function () {
__HELPERS__
  var D = __DATA__;
  var sp = document.getElementById("tt-price"); if (!sp) return;
  var RED = "#b3402f", BLU = "#1c5cab", L = 70, R = 946;
  var VALID = []; D.price.forEach(function (row, k) { if (row.some(function (q) { return q !== null; })) VALID.push(k); });
  var pos = Math.max(0, VALID.indexOf(D.days.indexOf(D.start)));
  function X(h) { return L + h / 23 * (R - L); }
  function nn(q) { return q !== null; }
  function clear(n) { while (n.firstChild) n.removeChild(n.firstChild); }
  function poly(g, pts, fill) { el("polygon", { points: pts.map(function (q) { return q[0].toFixed(1) + "," + q[1].toFixed(1); }).join(" "), fill: fill, "fill-opacity": 0.3 }, g); }
  function panel(g, a, b, T, B, title, upCol, downCol, upWord, downWord, withHours, zoom) {
    var all = a.concat(b).filter(nn), lo = Math.min.apply(null, all), hi = Math.max.apply(null, all), span = Math.max(hi - lo, 50);
    if (zoom) { lo -= span * 0.15; hi += span * 0.15; } else { lo = Math.min(0, lo); hi = hi * 1.1 + 1; }
    function Y(q) { return B - (q - lo) / (hi - lo) * (B - T); }
    niceTicks(lo, hi, 3).forEach(function (q) { el("line", { x1: L, y1: Y(q), x2: R, y2: Y(q), stroke: q === 0 ? "#c3c2b7" : "#ecebe6" }, g); txt(g, L - 8, Y(q) + 4, fmt(q), { anchor: "end" }); });
    txt(g, L, 13, title, { fill: "#52514e", size: 13, weight: 600 });
    for (var h = 0; h < 23; h++) {
      if (!nn(a[h]) || !nn(a[h + 1]) || !nn(b[h]) || !nn(b[h + 1])) continue;
      var d0 = b[h] - a[h], d1 = b[h + 1] - a[h + 1];
      if (d0 * d1 >= 0) poly(g, [[X(h), Y(a[h])], [X(h + 1), Y(a[h + 1])], [X(h + 1), Y(b[h + 1])], [X(h), Y(b[h])]], d0 + d1 > 0 ? upCol : downCol);
      else {
        var t = d0 / (d0 - d1), xm = h + t, ym = a[h] + t * (a[h + 1] - a[h]);
        poly(g, [[X(h), Y(a[h])], [X(xm), Y(ym)], [X(h), Y(b[h])]], d0 > 0 ? upCol : downCol);
        poly(g, [[X(xm), Y(ym)], [X(h + 1), Y(a[h + 1])], [X(h + 1), Y(b[h + 1])]], d1 > 0 ? upCol : downCol);
      }
    }
    function path(s) { var out = ""; s.forEach(function (q, h) { if (nn(q)) out += (out ? " L" : "M") + X(h).toFixed(1) + "," + Y(q).toFixed(1); }); return out; }
    el("path", { d: path(a), fill: "none", stroke: "#898781", "stroke-width": 2.2, "stroke-linejoin": "round" }, g);
    el("path", { d: path(b), fill: "none", stroke: "#0d366b", "stroke-width": 2.8, "stroke-linejoin": "round" }, g);
    var last = 23; while (last > 0 && (!nn(a[last]) || !nn(b[last]))) last--;
    var ya = Y(a[last]), yb = Y(b[last]);
    if (Math.abs(ya - yb) < 16) { var mid = (ya + yb) / 2, s = ya <= yb ? -1 : 1; ya = mid + s * 8; yb = mid - s * 8; }
    txt(g, R + 8, ya + 4, "today", { fill: "#898781", weight: 700, size: 13.5 });
    txt(g, R + 8, yb + 4, "tomorrow", { fill: "#0d366b", weight: 700, size: 13.5 });
    [[1, upWord, upCol], [-1, downWord, downCol]].forEach(function (w) {
      var best = null, start = null;
      for (var h2 = 0; h2 <= 24; h2++) {
        var inRun = h2 < 24 && nn(a[h2]) && nn(b[h2]) && (b[h2] - a[h2]) * w[0] > 0;
        if (inRun && start === null) start = h2;
        if (!inRun && start !== null) { if (!best || h2 - start > best[1] - best[0]) best = [start, h2]; start = null; }
      }
      if (!best || best[1] - best[0] < 3) return;
      var lo3 = Math.max(best[0], 3), hi3 = Math.min(best[1] - 1, 20), m = Math.max(3, Math.min(20, Math.round((best[0] + best[1] - 1) / 2)));
      for (var h3 = lo3; h3 <= hi3; h3++) if (Math.abs(Y(a[h3]) - Y(b[h3])) > Math.abs(Y(a[m]) - Y(b[m]))) m = h3;
      var y1 = Y(a[m]), y2 = Y(b[m]), gap = Math.abs(y1 - y2);
      var y = gap > 22 ? (y1 + y2) / 2 + 5 : Math.min(y1, y2) - 9;
      txt(g, X(m), y, w[1], { anchor: "middle", fill: w[2], weight: 700, size: 13.5, halo: true });
    });
    if (withHours) [0, 6, 12, 18, 23].forEach(function (h3) { txt(g, X(h3), B + 18, hh(h3), { anchor: "middle" }); });
  }
  function draw() {
    var k = VALID[pos], sw = document.getElementById("tt-wind"); clear(sp); clear(sw);
    panel(sp, D.before[k], D.price[k], 24, 168, "price, kr per MWh", RED, BLU, "higher tomorrow", "lower tomorrow", false, true);
    panel(sw, D.wind_before[k], D.wind[k], 24, 142, "wind forecast, MW", BLU, RED, "more wind tomorrow", "less wind tomorrow", true, false);
    document.getElementById("tt-date").textContent = "tomorrow: " + longDate(D.days[k]);
    document.getElementById("tt-range").value = pos;
  }
  var rng = document.getElementById("tt-range"); rng.max = VALID.length - 1;
  rng.addEventListener("input", function (e) { pos = +e.target.value; draw(); });
  document.getElementById("tt-prev").addEventListener("click", function () { pos = Math.max(0, pos - 1); draw(); });
  document.getElementById("tt-next").addEventListener("click", function () { pos = Math.min(VALID.length - 1, pos + 1); draw(); });
  var bar = document.getElementById("tt-marks");
  Object.keys(D.marks).forEach(function (name) { var b = document.createElement("button"); b.className = "w-btn"; b.textContent = name; b.addEventListener("click", function () { pos = Math.max(0, VALID.indexOf(D.days.indexOf(D.marks[name]))); draw(); }); bar.appendChild(b); bar.appendChild(document.createTextNode(" ")); });
  draw();
})();
</script>
"""


def today_tomorrow(heat_data, start="2024-01-30", marks=None):
    payload = {k: heat_data[k] for k in ("days", "price", "before", "wind", "wind_before")}
    payload["start"] = start
    payload["marks"] = marks or {}
    return _fill(TODAY, data=payload)


# ============================================================ 2024 as a grid of hours
HEAT = r"""
<style>__BASECSS__
.w-heat { position: relative; }
.w-heat-bar { display: flex; gap: 0.5em; align-items: center; margin-bottom: 0.25em; }
.w-heat-key { display: flex; gap: 1.3em; align-items: center; font: 14px "Segoe UI", sans-serif; color: #52514e; margin: 3px 0 0 70px; flex-wrap: wrap; }
.w-heat-key i { display: inline-block; width: 15px; height: 13px; margin-right: 6px; vertical-align: -1px; border-radius: 2px; }
.w-heat-ctl { display: flex; align-items: center; gap: 0.45em; margin: 0.25em 0 0; }
.w-heat-ctl input { flex: 1 1 0; }
</style>
<div class="w-heat" id="__ID__">
__BAR__
<canvas id="__ID__-cv" role="img" aria-label="__LABEL__"></canvas>
<div class="w-heat-key" id="__ID__-key"></div>
<div class="w-tip" id="__ID__-tip"></div>
__CTL__
__CURVE__
</div>
<script>
(function () {
__HELPERS__
  var D = __DATA__, ID = "__ID__", MODES = __MODES__, MODE = MODES[0];
  var cv = document.getElementById(ID + "-cv"); if (!cv) return;
  var N = D.days.length, CW = 2.62, CH = __CELLH__, L = 70, T = 6, W = L + N * CW + 8, H = T + 24 * CH + 24;
  var c = hidpi(cv, W, H), sel = null;
  var TWO = { up: ["#b3402f", "#86b6ef", "higher than the day before", "not higher"],
              pred: ["#b3402f", "#86b6ef", "predicted higher", "predicted not higher"],
              wrong: ["#141413", "#e6e5df", "wrong", "right"] };
  var LABEL = { price: "the prices", up: "what happened", pred: "what the model predicted", wrong: "where it was wrong" };
  function value(d, h) {
    if (MODE === "wrong") { var u = D.grids.up[d][h], q = D.grids.pred[d][h]; return u === null ? null : (u !== q ? 1 : 0); }
    return D.grids[MODE][d][h];
  }
  function colour(v) { if (v === null || v === undefined) return "#ffffff"; if (MODE === "price") return priceColour(v); return v ? TWO[MODE][0] : TWO[MODE][1]; }
  function paint() {
    c.clearRect(0, 0, W, H);
    for (var d = 0; d < N; d++) for (var h = 0; h < 24; h++) { c.fillStyle = colour(value(d, h)); c.fillRect(L + d * CW, T + h * CH, CW + 0.35, CH + 0.35); }
    if (sel !== null) { c.strokeStyle = "#141413"; c.lineWidth = 1.4; c.strokeRect(L + sel * CW - 1, T - 1, CW + 2, 24 * CH + 2); }
    c.font = "12.5px Segoe UI, sans-serif"; c.fillStyle = "#898781"; c.textAlign = "right"; c.textBaseline = "middle";
    [0, 6, 12, 18, 23].forEach(function (h) { c.fillText(hh(h), L - 7, T + (h + 0.5) * CH); });
    c.textAlign = "left"; c.textBaseline = "top";
    D.days.forEach(function (iso, d) { if (iso.slice(8) === "01") { c.fillStyle = "#c3c2b7"; c.fillRect(L + d * CW, T + 24 * CH, 1, 5); c.fillStyle = "#898781"; c.fillText(MONTHS[+iso.slice(5, 7) - 1].slice(0, 3), L + d * CW + 2, T + 24 * CH + 6); } });
    var k = document.getElementById(ID + "-key"), sw = function (col, t) { return '<span><i style="background:' + col + '"></i>' + t + "</span>"; };
    if (MODE === "price") k.innerHTML = sw(priceColour(-200), "below zero") + sw(priceColour(150), "150") + sw(priceColour(600), "600") + sw(priceColour(1100), "1,100") + sw(priceColour(1600), "1,500 and above, kr per MWh");
    else k.innerHTML = sw(TWO[MODE][0], TWO[MODE][2]) + sw(TWO[MODE][1], TWO[MODE][3]);
  }
  var bar = document.getElementById(ID + "-bar");
  if (bar) MODES.forEach(function (m) { var b = document.createElement("button"); b.className = "w-btn" + (m === MODE ? " on" : ""); b.textContent = LABEL[m]; b.addEventListener("click", function () { MODE = m; Array.prototype.forEach.call(bar.children, function (x) { x.classList.toggle("on", x === b); }); paint(); }); bar.appendChild(b); });
  var tip = document.getElementById(ID + "-tip");
  function cell(e) { var r = cv.getBoundingClientRect(), x = (e.clientX - r.left) / r.width * W, y = (e.clientY - r.top) / r.height * H; var d = Math.floor((x - L) / CW), h = Math.floor((y - T) / CH); return (d >= 0 && d < N && h >= 0 && h < 24) ? [d, h] : null; }
  cv.addEventListener("mousemove", function (e) {
    var q = cell(e); if (!q) { tip.style.display = "none"; return; }
    var v = D.price[q[0]][q[1]], s = longDate(D.days[q[0]]) + ", " + hh(q[1]) + ": ";
    if (v === null) s += "no data (no forecast published, or the clocks changed)";
    else {
      s += fmt(v) + " kr per MWh";
      if (MODE !== "price" && D.before[q[0]][q[1]] !== null) s += ", the day before " + fmt(D.before[q[0]][q[1]]);
      if (MODE !== "price" && D.prob) s += " · the model's probability " + D.prob[q[0]][q[1]].toFixed(2);
    }
    tip.textContent = s; tip.style.display = "block";
    var box = cv.parentNode.getBoundingClientRect(), sc = box.width / cv.parentNode.offsetWidth;
    tip.style.left = Math.min((e.clientX - box.left) / sc + 14, cv.parentNode.offsetWidth - 560) + "px"; tip.style.top = ((e.clientY - box.top) / sc + 14) + "px";
  });
  cv.addEventListener("mouseleave", function () { tip.style.display = "none"; });
  function curve(d) {
    var g = document.getElementById(ID + "-curve"); if (!g) return;
    while (g.firstChild) g.removeChild(g.firstChild);
    var v = D.price[d], ok = v.filter(function (q) { return q !== null; });
    if (!ok.length) { txt(g, 70, 18, longDate(D.days[d]) + ": no data for this day", { size: 14, weight: 600, fill: "#0d366b" }); return; }
    var avg = ok.reduce(function (a, b) { return a + b; }, 0) / ok.length;
    var lo = Math.min(0, Math.min.apply(null, ok)) * 1.15, hi = Math.max.apply(null, ok) * 1.1 + 20, L2 = 70, R2 = 1010, T2 = 30, B2 = 134;
    function X(h) { return L2 + (h + 0.5) / 24 * (R2 - L2); } function Y(q) { return B2 - (q - lo) / (hi - lo) * (B2 - T2); }
    el("line", { x1: L2, y1: Y(0), x2: R2, y2: Y(0), stroke: "#c3c2b7" }, g);
    v.forEach(function (q, h) { if (q === null) return; var top = Math.min(Y(q), Y(0)), ht = Math.abs(Y(q) - Y(0)); el("rect", { x: X(h) - 14, y: top, width: 28, height: Math.max(1.5, ht), rx: 2, fill: "#1c5cab" }, g); });
    [0, 6, 12, 18, 23].forEach(function (h) { txt(g, X(h), B2 + 17, hh(h), { anchor: "middle" }); });
    txt(g, L2, 17, longDate(D.days[d]), { size: 14.5, weight: 700, fill: "#0d366b" });
    txt(g, L2 + 290, 17, "average " + fmt(avg) + " kr per MWh", { size: 14, fill: "#52514e" });
  }
  function select(d) { sel = Math.max(0, Math.min(N - 1, d)); paint(); curve(sel); var r = document.getElementById(ID + "-range"); if (r) r.value = sel; }
  cv.addEventListener("click", function (e) { var q = cell(e); if (!q || !document.getElementById(ID + "-curve")) return; select(q[0]); });
  var rng = document.getElementById(ID + "-range");
  if (rng) {
    rng.addEventListener("input", function (e) { select(+e.target.value); });
    document.getElementById(ID + "-prev").addEventListener("click", function () { select(sel - 1); });
    document.getElementById(ID + "-next").addEventListener("click", function () { select(sel + 1); });
    Object.keys(D.marks || {}).forEach(function (name) { var b = document.createElement("button"); b.className = "w-btn"; b.textContent = name; b.addEventListener("click", function () { select(D.days.indexOf(D.marks[name])); }); document.getElementById(ID + "-marks").appendChild(b); });
  }
  if (document.getElementById(ID + "-curve")) select(__START__); else paint();
})();
</script>
"""


def heat(prefix, data, modes, label="", with_curve=False, start_day=0, cell_h=9, marks=None):
    need = [m for m in modes if m != "wrong"] + (["up", "pred"] if "wrong" in modes else [])
    payload = {"days": data["days"], "grids": {m: data[m] for m in dict.fromkeys(need)},
               "price": data["price"], "before": data["before"]}
    if any(m != "price" for m in modes):
        payload["prob"] = data["prob"]
    curve = (f'<svg id="{prefix}-curve" width="1040" height="156" role="img" '
             f'aria-label="The chosen day\'s 24 hourly prices"></svg>') if with_curve else ""
    bar = f'<div class="w-heat-bar" id="{prefix}-bar"></div>' if len(modes) > 1 else ""
    ctl = (f'<div class="w-heat-ctl"><button class="w-btn" id="{prefix}-prev" aria-label="the day before">&#9664;</button>'
           f'<input class="w-range" type="range" id="{prefix}-range" min="0" max="{len(data["days"]) - 1}" value="{int(start_day)}" aria-label="the day">'
           f'<button class="w-btn" id="{prefix}-next" aria-label="the day after">&#9654;</button>'
           f'<span class="w-marks" id="{prefix}-marks"></span></div>') if with_curve else ""
    payload["marks"] = marks or {}
    return _fill(HEAT, data=payload, id=prefix, modes=list(modes), label=label, curve=curve, bar=bar, ctl=ctl,
                 start=str(int(start_day)), cellh=str(cell_h))


# ============================================================ the forecasts
FORECAST = r"""
<style>__BASECSS__
.w-fc table { border-collapse: collapse; width: 100%; font: 17px "Segoe UI", sans-serif; }
.w-fc th { font: 600 13.5px "Segoe UI", sans-serif; color: #898781; padding: 0 12px 6px; vertical-align: bottom; line-height: 1.25; text-align: right !important; border-bottom: 1.5px solid #c3c2b7; }
.w-fc th.l { text-align: left !important; }
.w-fc td { padding: 7px 12px; border-bottom: 1px solid #ecebe6; white-space: nowrap; }
.w-fc td.when { font-weight: 600; color: #0d366b; }
.w-fc td.n { text-align: right; font-family: "Cascadia Mono", Consolas, monospace; color: #141413; font-weight: 600; }
.w-fc td.n.m { color: #898781; font-weight: 400; }
.w-fc td.gap, .w-fc th.gap { padding-left: 26px; }
.w-fc td.p { width: 34%; padding-left: 26px; }
.w-fc .slot { display: flex; align-items: center; gap: 0.7em; }
.w-fc input[type=range] { flex: 1 1 0; accent-color: #1c5cab; }
.w-fc input[type=range].unset { opacity: 0.35; }
.w-fc .v { font: 700 17px "Cascadia Mono", Consolas, monospace; color: #0d366b; min-width: 4.6em; text-align: right; }
.w-fc .v.unset { color: #c3c2b7; font-weight: 400; }
.w-fc-foot { display: flex; justify-content: space-between; align-items: baseline; font: 13.5px "Segoe UI", sans-serif; color: #898781; margin-top: 8px; }
.w-fc-foot button { font: 13.5px "Segoe UI", sans-serif; color: #1c5cab; background: none; border: none; padding: 0; cursor: pointer; text-decoration: underline; }
</style>
<div class="w-fc" id="fc">
<table id="fc-table" data-quarto-disable-processing="true">
<tr class="h1"><th class="l">tomorrow,<br>at the hour</th><th>price<br>yesterday, kr</th><th>price<br>today, kr</th><th class="gap">wind forecast<br>today, MW</th><th>wind forecast<br>tomorrow, MW</th><th class="l gap">how likely is tomorrow's price<br>higher than today's?</th></tr>
</table>
<div class="w-fc-foot"><span>The models see the same numbers and the weekday. Your answers stay in this browser.</span><button id="fc-clear">clear my answers</button></div>
</div>
<script>
(function () {
__HELPERS__
  var G = __DATA__, KEY = "__KEY__";
  var table = document.getElementById("fc-table"); if (!table) return;
  var n = G.rounds.length, saved = recall(KEY), a = [];
  for (var k = 0; k < n; k++) a.push(saved && saved[k] !== undefined ? saved[k] : null);
  var sliders = [], shown = [];
  G.rounds.forEach(function (r, k) {
    var tr = document.createElement("tr");
    tr.innerHTML = "<td class='when'>" + r.short + ", " + hh(r.hour) + "</td><td class='n m'>" + fmt(r.price_2before) + "</td><td class='n'>" + fmt(r.price_before) + "</td><td class='n m gap'>" + fmt(r.wind_before) + "</td><td class='n'>" + fmt(r.wind) + "</td><td class='p'><div class='slot'><input type='range' min='1' max='99' aria-label='your probability for " + r.short + ", " + hh(r.hour) + "'><span class='v'></span></div></td>";
    table.appendChild(tr);
    var s = tr.querySelector("input"), v = tr.querySelector(".v");
    sliders.push(s); shown.push(v);
    s.addEventListener("input", function () { a[k] = s.value / 100; store(KEY, a); paint(k); });
  });
  function paint(k) {
    var s = sliders[k], v = shown[k], set = a[k] !== null;
    s.value = set ? Math.round(a[k] * 100) : 50;
    s.classList.toggle("unset", !set); v.classList.toggle("unset", !set);
    v.textContent = set ? Math.round(a[k] * 100) + "%" : "not set";
  }
  for (var k2 = 0; k2 < n; k2++) paint(k2);
  document.getElementById("fc-clear").addEventListener("click", function () { for (var k = 0; k < n; k++) { a[k] = null; paint(k); } store(KEY, a); });
})();
</script>
"""

GAME_KEY = "mlfin10-updown"


def forecast(data):
    keep = ("short", "hour", "price_2before", "price_before", "wind_before", "wind")
    return _fill(FORECAST, data={"rounds": [{k: r[k] for k in keep} for r in data["rounds"]]}, key=GAME_KEY)


# ============================================================ the forecasts, revised at the end
REVISE = r"""
<style>__BASECSS__
.w-rv table { border-collapse: collapse; width: 100%; font: 17px "Segoe UI", sans-serif; }
.w-rv th { font: 600 13.5px "Segoe UI", sans-serif; color: #898781; padding: 0 12px 6px; vertical-align: bottom; line-height: 1.25; text-align: right !important; border-bottom: 1.5px solid #c3c2b7; }
.w-rv th.l { text-align: left !important; }
.w-rv td { padding: 7px 12px; border-bottom: 1px solid #ecebe6; white-space: nowrap; }
.w-rv td.when { font-weight: 600; color: #0d366b; }
.w-rv td.n { text-align: right; font-family: "Cascadia Mono", Consolas, monospace; color: #141413; font-weight: 600; }
.w-rv td.first { text-align: right; font-family: "Cascadia Mono", Consolas, monospace; color: #52514e; }
.w-rv td.p { width: 31%; padding-left: 26px; }
.w-rv .slot { display: flex; align-items: center; gap: 0.7em; }
.w-rv input[type=range] { flex: 1 1 0; accent-color: #1c5cab; }
.w-rv input[type=range].same { opacity: 0.4; }
.w-rv .v { font: 700 17px "Cascadia Mono", Consolas, monospace; color: #0d366b; min-width: 5.6em; text-align: right; }
.w-rv .v.same { color: #c3c2b7; font-weight: 400; }
.w-rv-foot { display: flex; justify-content: space-between; align-items: baseline; font: 13.5px "Segoe UI", sans-serif; color: #898781; margin-top: 8px; }
.w-rv-foot button { font: 13.5px "Segoe UI", sans-serif; color: #1c5cab; background: none; border: none; padding: 0; cursor: pointer; text-decoration: underline; }
</style>
<div class="w-rv" id="rv">
<table id="rv-table" data-quarto-disable-processing="true">
<tr><th class="l">tomorrow, at the hour</th><th>wind change, MW</th><th>today's move, kr</th><th>your first answer</th><th class="l" style="padding-left: 26px">your answer now</th></tr>
</table>
<div class="w-rv-foot"><span>An answer you leave alone stays as it was.</span><button id="rv-reset">back to my first answers</button></div>
</div>
<script>
(function () {
__HELPERS__
  var G = __DATA__, K1 = "__KEY__", K2 = "__KEY__-revised";
  var table = document.getElementById("rv-table"); if (!table) return;
  var n = G.rounds.length, sliders = [], shown = [], firsts = [];
  G.rounds.forEach(function (r, k) {
    var tr = document.createElement("tr");
    tr.innerHTML = "<td class='when'>" + r.short + ", " + hh(r.hour) + "</td><td class='n'>" + signed(r.wind - r.wind_before) + "</td><td class='n'>" + signed(r.price_before - r.price_2before) + "</td><td class='first'></td><td class='p'><div class='slot'><input type='range' min='1' max='99' aria-label='your answer now for " + r.short + ", " + hh(r.hour) + "'><span class='v'></span></div></td>";
    table.appendChild(tr);
    firsts.push(tr.querySelector(".first")); sliders.push(tr.querySelector("input")); shown.push(tr.querySelector(".v"));
    tr.querySelector("input").addEventListener("input", function (e) { var rev = recall(K2) || []; while (rev.length < n) rev.push(null); rev[k] = e.target.value / 100; store(K2, rev); paint(); });
  });
  function got(a, k) { return a[k] !== null && a[k] !== undefined; }
  function paint() {
    var first = recall(K1) || [], rev = recall(K2) || [];
    for (var k = 0; k < n; k++) {
      var f = got(first, k) ? first[k] : null, r = got(rev, k) ? rev[k] : null, now = r !== null ? r : (f !== null ? f : 0.5);
      firsts[k].textContent = f === null ? "not set" : Math.round(f * 100) + "%";
      sliders[k].value = Math.round(now * 100);
      sliders[k].classList.toggle("same", r === null); shown[k].classList.toggle("same", r === null);
      shown[k].textContent = r === null ? "unchanged" : Math.round(r * 100) + "%";
    }
  }
  document.getElementById("rv-reset").addEventListener("click", function () { store(K2, []); paint(); });
  paint();
  onShow(table, paint);
})();
</script>
"""


def revise(data):
    keep = ("short", "hour", "price_2before", "price_before", "wind_before", "wind")
    return _fill(REVISE, data={"rounds": [{k: r[k] for k in keep} for r in data["rounds"]]}, key=GAME_KEY)


SCORE = r"""
<style>__BASECSS__
.w-score table { border-collapse: collapse; font: 15px "Segoe UI", sans-serif; width: 100%; }
.w-score th { color: #52514e; font-weight: 600; text-align: right !important; padding: 4px 10px; border-bottom: 1.5px solid #c3c2b7; line-height: 1.2; vertical-align: bottom; }
.w-score td { text-align: right !important; padding: 4px 10px; border-bottom: 1px solid #ecebe6; font-family: "Cascadia Mono", Consolas, monospace; color: #52514e; white-space: nowrap; }
.w-score td.t, .w-score th.t { text-align: left !important; font-family: "Segoe UI", sans-serif; }
.w-score tr.sum td { border-top: 1.5px solid #c3c2b7; color: #0d366b; font-weight: 700; font-size: 15.5px; }
.w-score td.best { background: #eaf1fa; color: #0d366b; font-weight: 700; }
.w-score td.unset { color: #c3c2b7; }
.w-score-note { font: 0.46em "Segoe UI", sans-serif; color: #52514e; margin-top: 0.35em; }
.w-score-note b { font-family: "Cascadia Mono", Consolas, monospace; color: #0d366b; }
</style>
<div class="w-score" id="score"><div id="score-body" style="min-height: 318px"></div><div class="w-score-note" id="score-note"></div></div>
<script>
(function () {
__HELPERS__
  var G = __DATA__, K1 = "__KEY__", K2 = "__KEY__-revised", WHO = ["first", "rev", "logit", "knn"];
  var body = document.getElementById("score-body"); if (!body) return;
  function brier(p, y) { return (p - y) * (p - y); }
  function logl(p, y) { p = Math.min(0.99, Math.max(0.01, p)); return -(y * Math.log(p) + (1 - y) * Math.log(1 - p)); }
  function got(a, k) { return a[k] !== null && a[k] !== undefined; }
  function render() {
    var first = recall(K1) || [], rev = recall(K2) || [], m = G.rounds.length, rows = "", unset = 0, tot = {};
    WHO.forEach(function (w) { tot[w] = [0, 0]; });
    G.rounds.forEach(function (r, k) {
      var f = got(first, k) ? first[k] : 0.5, ps = { first: f, rev: got(rev, k) ? rev[k] : f, logit: r.logit, knn: r.knn };
      if (!got(first, k)) unset++;
      var best = Math.min.apply(null, WHO.map(function (w) { return brier(ps[w], r.truth); }));
      WHO.forEach(function (w) { tot[w][0] += brier(ps[w], r.truth); tot[w][1] += logl(ps[w], r.truth); });
      rows += "<tr><td class='t'>" + r.short + ", " + hh(r.hour) + "</td><td class='t'>" + (r.truth ? "higher: " : "not higher: ") + fmt(r.price_before) + " &rarr; " + fmt(r.price) + " kr</td>" +
        WHO.map(function (w) { var cls = Math.abs(brier(ps[w], r.truth) - best) < 1e-9 ? "best" : (w === "first" && !got(first, k) ? "unset" : ""); return "<td class='" + cls + "'>" + Math.round(ps[w] * 100) + "%</td>"; }).join("") + "</tr>";
    });
    function sumRow(label, j) {
      var v = WHO.map(function (w) { return tot[w][j] / m; }), best = Math.min.apply(null, v);
      return "<tr class='sum'><td class='t' colspan='2'>" + label + "</td>" + v.map(function (x) { return "<td" + (Math.abs(x - best) < 1e-9 ? " class='best'" : "") + ">" + x.toFixed(3) + "</td>"; }).join("") + "</tr>";
    }
    body.innerHTML = "<table><tr><th class='t'>tomorrow, at the hour</th><th class='t'>what happened</th><th>your first<br>answer</th><th>your answer<br>now</th><th>logistic<br>regression</th><th>k-NN</th></tr>" + rows +
      sumRow("Brier score, average over the " + m + " hours", 0) + sumRow("log loss, average over the " + m + " hours", 1) + "</table>";
    var note = "Shaded: the closest forecast on each hour, and the lowest score. ";
    if (unset) note += "Answers you did not set count as 50 percent. ";
    note += "Saying 50 percent every time scores <b>0.250</b> and <b>0.693</b>.";
    document.getElementById("score-note").innerHTML = note;
  }
  function again() { render(); try { if (window.Reveal && Reveal.layout) Reveal.layout(); } catch (e) {} }
  render();
  onShow(body, again);
})();
</script>
"""


def scoreboard(data):
    keep = ("short", "hour", "truth", "logit", "knn", "price_before", "price")
    return _fill(SCORE, data={"rounds": [{k: r[k] for k in keep} for r in data["rounds"]]}, key=GAME_KEY)


# ============================================================ the logistic curve of the wind change
SCURVE = r"""
<style>__BASECSS__
.w-sc-top { display: flex; align-items: center; gap: 1em; margin-bottom: 0.1em; }
.w-sc-top label { font: 600 0.52em "Segoe UI", sans-serif; color: #0d366b; display: flex; align-items: center; gap: 0.5em; white-space: nowrap; }
.w-sc-top input { width: 18em; }
</style>
<div id="sc">
<div class="w-sc-top"><label>wind change <input class="w-range" type="range" id="sc-w" min="-3000" max="3000" step="50" value="1000" aria-label="the wind change"> <span class="w-val" id="sc-wv"></span></label></div>
<svg id="sc-svg" width="1040" height="344" viewBox="0 0 1040 344" role="img" aria-label="The share of 2022-23 hours whose price rose, by the change in the wind forecast, as dots, with the logistic regression's probability as a red curve and the chosen wind change marked on it"></svg>
</div>
<script>
(function () {
__HELPERS__
  var D = __DATA__;
  var svg = document.getElementById("sc-svg"); if (!svg) return;
  var curve = D.curves.alone[0], dots = D.dots.all, L = 76, R = 1000, T = 10, B = 290, XMIN = -3000, XMAX = 3000;
  function X(w) { return L + (w - XMIN) / (XMAX - XMIN) * (R - L); } function Y(p) { return B - p * (B - T); }
  function pAt(w) { var step = D.grid[1] - D.grid[0], i = (w - D.grid[0]) / step, i0 = Math.max(0, Math.min(curve.length - 2, Math.floor(i))); return lerp(curve[i0], curve[i0 + 1], i - i0); }
  function draw() {
    var w = +document.getElementById("sc-w").value, p = pAt(w);
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    [0, 0.25, 0.5, 0.75, 1].forEach(function (q) { el("line", { x1: L, y1: Y(q), x2: R, y2: Y(q), stroke: q === 0 ? "#c3c2b7" : "#ecebe6" }, svg); txt(svg, L - 8, Y(q) + 4, q.toFixed(2), { anchor: "end" }); });
    [-3000, -2000, -1000, 0, 1000, 2000, 3000].forEach(function (x) { txt(svg, X(x), B + 19, (x > 0 ? "+" : "") + fmt(x), { anchor: "middle" }); });
    el("line", { x1: X(0), y1: T, x2: X(0), y2: B, stroke: "#c3c2b7", "stroke-dasharray": "4 4" }, svg);
    txt(svg, (L + R) / 2, B + 42, "wind change: tomorrow's wind forecast minus today's, MW", { anchor: "middle", fill: "#52514e", size: 13.5 });
    txt(svg, 18, (T + B) / 2, "share higher than the day before", { anchor: "middle", rotate: -90, fill: "#52514e", size: 13 });
    D.centres.forEach(function (x, i) { var s = dots.share[i], n = dots.n[i]; if (s === null || n < 15) return; el("circle", { cx: X(x), cy: Y(s), r: 2.5 + Math.sqrt(n) / 4.5, fill: "#0d366b", "fill-opacity": 0.72, stroke: "#fcfcfb", "stroke-width": 1.5 }, svg); });
    var d = ""; D.grid.forEach(function (x, i) { if (x < XMIN || x > XMAX) return; d += (d ? " L" : "M") + X(x).toFixed(1) + "," + Y(curve[i]).toFixed(1); });
    el("path", { d: d, fill: "none", stroke: "#b3402f", "stroke-width": 3.2 }, svg);
    txt(svg, X(500), Y(0.92), "dots: the share of hours that rose, 2022 and 2023", { fill: "#0d366b", size: 14, weight: 600 });
    txt(svg, X(-2900), Y(0.07), "red line: a logistic regression on the wind change", { fill: "#b3402f", size: 14, weight: 600 });
    el("line", { x1: X(w), y1: B, x2: X(w), y2: Y(p), stroke: "#141413", "stroke-width": 1.2, "stroke-dasharray": "4 4" }, svg);
    el("line", { x1: L, y1: Y(p), x2: X(w), y2: Y(p), stroke: "#141413", "stroke-width": 1.2, "stroke-dasharray": "4 4" }, svg);
    el("circle", { cx: X(w), cy: Y(p), r: 7.5, fill: "#b3402f", stroke: "#fcfcfb", "stroke-width": 2 }, svg);
    var right = w < 1600;
    txt(svg, X(w) + (right ? 14 : -14), Y(p) - 12, "probability " + p.toFixed(2), { anchor: right ? "start" : "end", fill: "#141413", weight: 700, size: 15, halo: true });
    document.getElementById("sc-wv").textContent = signed(w) + " MW";
  }
  document.getElementById("sc-w").addEventListener("input", draw);
  draw();
})();
</script>
"""


def scurve(data):
    s = data["scurve"]
    return _fill(SCURVE, data={"grid": s["grid"], "centres": s["centres"], "curves": {"alone": s["curves"]["alone"][:1]},
                               "dots": {"all": s["dots"]["all"]}})


# ============================================================ the weekday: one number or seven columns
WEEKDAY = r"""
<style>__BASECSS__
.w-wk-top { display: flex; align-items: center; gap: 0.45em; margin-bottom: 0.1em; }
</style>
<div id="wk">
<div class="w-wk-top" id="wk-modes"><span class="w-read">&nbsp;AUC on 2024 <b id="wk-auc"></b></span></div>
<svg id="wk-svg" width="1040" height="330" viewBox="0 0 1040 330" role="img" aria-label="By weekday, the share of 2022-23 hours whose price rose, as bars, with the model's average probability for each weekday as red dots"></svg>
</div>
<script>
(function () {
__HELPERS__
  var D = __DATA__;
  var svg = document.getElementById("wk-svg"); if (!svg) return;
  var MODE = "alone", cur = D.avg_p.alone.slice(), aucNow = D.auc.alone, anim = null;
  var NAMES = { alone: "without the weekday", number: "the weekday as one number", columns: "the weekday as 7 columns" };
  var L = 76, R = 1000, T = 36, B = 288;
  function X(d) { return L + (d + 0.5) / 7 * (R - L); } function Y(p) { return B - p * (B - T); }
  function frame() {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    [0, 0.25, 0.5, 0.75, 1].forEach(function (q) { el("line", { x1: L, y1: Y(q), x2: R, y2: Y(q), stroke: q === 0 ? "#c3c2b7" : "#ecebe6" }, svg); txt(svg, L - 8, Y(q) + 4, q.toFixed(2), { anchor: "end" }); });
    D.share_by_day.forEach(function (s, d) { el("rect", { x: X(d) - 46, y: Y(s), width: 92, height: B - Y(s), rx: 3, fill: "#e3e2dc" }, svg); txt(svg, X(d), B + 21, WDNAMES[d], { anchor: "middle", fill: "#52514e", size: 14 }); });
    el("polyline", { points: cur.map(function (p, d) { return X(d) + "," + Y(p); }).join(" "), fill: "none", stroke: "#b3402f", "stroke-width": 2.4 }, svg);
    cur.forEach(function (p, d) { el("circle", { cx: X(d), cy: Y(p), r: 7.5, fill: "#b3402f", stroke: "#fcfcfb", "stroke-width": 2 }, svg); });
    el("rect", { x: L, y: 6, width: 18, height: 15, rx: 2, fill: "#e3e2dc" }, svg); txt(svg, L + 26, 19, "bars: the share of hours that rose, 2022 and 2023", { fill: "#52514e", size: 14 });
    el("circle", { cx: L + 452, cy: 13.5, r: 7, fill: "#b3402f" }, svg); txt(svg, L + 466, 19, "dots: the model's average probability", { fill: "#52514e", size: 14 });
    txt(svg, 18, (T + B) / 2, "share higher than the day before", { anchor: "middle", rotate: -90, fill: "#52514e", size: 13 });
    document.getElementById("wk-auc").textContent = aucNow.toFixed(3);
  }
  var modes = document.getElementById("wk-modes");
  function go(m) {
    var from = cur.slice(), a0 = aucNow, t0 = null; MODE = m; if (anim) cancelAnimationFrame(anim);
    function step(t) { if (t0 === null) t0 = t; var u = Math.min(1, (t - t0) / 800), e = u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2; cur = from.map(function (p, d) { return lerp(p, D.avg_p[m][d], e); }); aucNow = lerp(a0, D.auc[m], e); frame(); if (u < 1) anim = requestAnimationFrame(step); }
    anim = requestAnimationFrame(step);
    Array.prototype.forEach.call(modes.querySelectorAll("button"), function (b) { b.classList.toggle("on", b.dataset.m === m); });
  }
  ["alone", "number", "columns"].forEach(function (m) { var b = document.createElement("button"); b.className = "w-btn" + (m === MODE ? " on" : ""); b.dataset.m = m; b.textContent = NAMES[m]; b.addEventListener("click", function () { go(m); }); modes.insertBefore(b, modes.lastElementChild); });
  frame();
})();
</script>
"""


def weekday(data):
    s = data["scurve"]
    return _fill(WEEKDAY, data={"share_by_day": s["share_by_day"], "avg_p": s["avg_p"], "auc": s["auc"]})


# ============================================================ a big move either way
VSHAPE = r"""
<style>__BASECSS__
.w-v-top { display: flex; align-items: center; gap: 0.4em; margin-bottom: 0.15em; flex-wrap: nowrap; }
.w-v-top .w-btn { font-size: 0.47em; }
.w-v-top .w-read { white-space: nowrap; }
</style>
<div id="vs">
<div class="w-v-top" id="vs-modes"><span class="w-read">&nbsp;AUC on 2024 <b id="vs-auc"></b></span></div>
<svg id="vs-svg" width="1040" height="340" viewBox="0 0 1040 340" role="img" aria-label="The share of 2022-23 hours whose price moved more than 150 kroner either way, by the change in the wind forecast, as dots, with a model's probability as a red curve"></svg>
</div>
<script>
(function () {
__HELPERS__
  var D = __DATA__;
  var svg = document.getElementById("vs-svg"); if (!svg) return;
  var MODE = "linear", cur = D.curves.linear.slice(), aucNow = D.auc.linear, anim = null;
  var NAMES = { linear: "logistic regression, the wind change", size: "logistic regression, the size of the wind change", knn: "k-NN, the wind change" };
  var L = 76, R = 1000, T = 10, B = 288, XMIN = -3000, XMAX = 3000;
  function X(w) { return L + (w - XMIN) / (XMAX - XMIN) * (R - L); } function Y(p) { return B - p * (B - T); }
  function at(w) { var step = D.grid[1] - D.grid[0], i = Math.round((w - D.grid[0]) / step); return cur[Math.max(0, Math.min(cur.length - 1, i))]; }
  function frame() {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    [0, 0.25, 0.5, 0.75, 1].forEach(function (p) { el("line", { x1: L, y1: Y(p), x2: R, y2: Y(p), stroke: p === 0 ? "#c3c2b7" : "#ecebe6" }, svg); txt(svg, L - 8, Y(p) + 4, p.toFixed(2), { anchor: "end" }); });
    [-3000, -2000, -1000, 0, 1000, 2000, 3000].forEach(function (w) { txt(svg, X(w), B + 19, (w > 0 ? "+" : "") + fmt(w), { anchor: "middle" }); });
    el("line", { x1: X(0), y1: T, x2: X(0), y2: B, stroke: "#c3c2b7", "stroke-dasharray": "4 4" }, svg);
    txt(svg, (L + R) / 2, B + 42, "wind change: tomorrow's wind forecast minus today's, MW", { anchor: "middle", fill: "#52514e", size: 13.5 });
    txt(svg, 18, (T + B) / 2, "share with a move above 150 kr", { anchor: "middle", rotate: -90, fill: "#52514e", size: 13 });
    D.centres.forEach(function (w, i) { var s = D.dots.share[i], n = D.dots.n[i]; if (s === null || n < 15) return; el("circle", { cx: X(w), cy: Y(s), r: 2.5 + Math.sqrt(n) / 4, fill: "#0d366b", "fill-opacity": 0.72, stroke: "#fcfcfb", "stroke-width": 1.5 }, svg); });
    var s = ""; D.grid.forEach(function (w, i) { if (w < XMIN || w > XMAX) return; s += (s ? " L" : "M") + X(w).toFixed(1) + "," + Y(cur[i]).toFixed(1); });
    el("path", { d: s, fill: "none", stroke: "#b3402f", "stroke-width": 3.2 }, svg);
    txt(svg, X(0), Y(0.93), "dots: the share of hours that moved more than 150 kr, 2022 and 2023", { anchor: "middle", fill: "#0d366b", size: 14, weight: 600 });
    txt(svg, X(1500), Y(at(1500)) + 26, "red line: the model", { fill: "#b3402f", size: 14, weight: 600, halo: true });
    document.getElementById("vs-auc").textContent = aucNow.toFixed(3);
  }
  function go(m) {
    var from = cur.slice(), a0 = aucNow, t0 = null; MODE = m; if (anim) cancelAnimationFrame(anim);
    function step(t) { if (t0 === null) t0 = t; var u = Math.min(1, (t - t0) / 800), e = u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2; cur = from.map(function (p, i) { return lerp(p, D.curves[m][i], e); }); aucNow = lerp(a0, D.auc[m], e); frame(); if (u < 1) anim = requestAnimationFrame(step); }
    anim = requestAnimationFrame(step);
    Array.prototype.forEach.call(modes.querySelectorAll("button"), function (b) { b.classList.toggle("on", b.dataset.m === m); });
  }
  var modes = document.getElementById("vs-modes");
  ["linear", "size", "knn"].forEach(function (m) { var b = document.createElement("button"); b.className = "w-btn" + (m === MODE ? " on" : ""); b.dataset.m = m; b.textContent = NAMES[m]; b.addEventListener("click", function () { go(m); }); modes.insertBefore(b, modes.lastElementChild); });
  frame();
})();
</script>
"""


def vshape(data):
    return _fill(VSHAPE, data=data["vshape"])


# ============================================================ the threshold, in kroner (a cost of 50 per trade)
THRESH = r"""
<style>__BASECSS__
.w-th-top { display: flex; align-items: center; gap: 1.2em; margin-bottom: 0.1em; }
.w-th-top label { font: 600 0.52em "Segoe UI", sans-serif; color: #0d366b; display: flex; align-items: center; gap: 0.5em; white-space: nowrap; }
.w-th-top input { width: 15em; }
.w-th { display: flex; gap: 30px; align-items: flex-start; }
.w-th table { border-collapse: collapse; font: 16px "Segoe UI", sans-serif; margin-top: 18px; }
.w-th td, .w-th th { padding: 9px 16px; text-align: right !important; border: 1px solid #ecebe6; }
.w-th th { color: #52514e; font-weight: 600; background: #f5f5f2; }
.w-th th.l { text-align: left !important; }
.w-th td b { font-family: "Cascadia Mono", Consolas, monospace; color: #0d366b; display: block; font-size: 17px; }
.w-th td span { font-family: "Cascadia Mono", Consolas, monospace; font-size: 14px; color: #52514e; white-space: nowrap; }
</style>
<div id="th">
<div class="w-th-top"><label>threshold <input class="w-range" type="range" id="th-t" min="50" max="95" value="50" aria-label="the threshold"> <span class="w-val" id="th-tv"></span></label><span class="w-read" id="th-read"></span></div>
<div class="w-th">
<svg id="th-svg" width="470" height="312" viewBox="0 0 470 312" role="img" aria-label="Kroner earned in 2024 against the threshold, at a cost of 50 kroner per trade"></svg>
<table id="th-tab" data-quarto-disable-processing="true"></table>
</div>
</div>
<script>
(function () {
__HELPERS__
  var D = __DATA__, C = D.cost;
  var svg = document.getElementById("th-svg"); if (!svg) return;
  var L = 90, R = 456, T = 10, B = 262;
  function trades(r) { return r.lu[0] + r.ld[0] + r.su[0] + r.sd[0]; }
  var vals = D.rows.map(function (r) { return Math.round(r.lu[1] + r.ld[1] + r.su[1] + r.sd[1] - C * trades(r)); });
  var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals), pad = (hi - lo) * 0.15; lo -= pad; hi += pad;
  function X(t) { return L + (t - 0.5) / 0.45 * (R - L); } function Y(k) { return B - (k - lo) / (hi - lo) * (B - T); }
  function draw() {
    var ti = +document.getElementById("th-t").value - 50, r = D.rows[ti];
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    niceTicks(lo, hi, 4).forEach(function (k) { el("line", { x1: L, y1: Y(k), x2: R, y2: Y(k), stroke: "#ecebe6" }, svg); txt(svg, L - 8, Y(k) + 4, fmt(k), { anchor: "end" }); });
    [0.5, 0.6, 0.7, 0.8, 0.9].forEach(function (t) { txt(svg, X(t), B + 20, t.toFixed(2), { anchor: "middle" }); });
    txt(svg, (L + R) / 2, B + 44, "threshold", { anchor: "middle", fill: "#52514e", size: 13.5 });
    txt(svg, 16, (T + B) / 2, "kroner earned in 2024", { anchor: "middle", rotate: -90, fill: "#52514e", size: 13 });
    el("polyline", { points: D.rows.map(function (q, i) { return X(q.thr) + "," + Y(vals[i]); }).join(" "), fill: "none", stroke: "#1c5cab", "stroke-width": 2.8, "stroke-linejoin": "round" }, svg);
    el("line", { x1: X(r.thr), y1: T, x2: X(r.thr), y2: B, stroke: "#b3402f", "stroke-width": 1.3 }, svg);
    el("circle", { cx: X(r.thr), cy: Y(vals[ti]), r: 7, fill: "#b3402f", stroke: "#fcfcfb", "stroke-width": 2 }, svg);
    document.getElementById("th-tv").textContent = r.thr.toFixed(2);
    document.getElementById("th-read").innerHTML = "<b>" + fmt(vals[ti]) + "</b> kr from <b>" + fmt(trades(r)) + "</b> of the " + fmt(D.n) + " hours";
    function cellT(q) { var k = Math.round(q[1] - C * q[0]); return "<td><b>" + fmt(q[0]) + "</b><span>" + (k > 0 ? "+" : "") + fmt(k) + " kr</span></td>"; }
    document.getElementById("th-tab").innerHTML = "<tr><th></th><th>went up</th><th>did not go up</th></tr>" +
      "<tr><th class='l'>long: a bet on higher</th>" + cellT(r.lu) + cellT(r.ld) + "</tr>" +
      "<tr><th class='l'>short: a bet on lower</th>" + cellT(r.su) + cellT(r.sd) + "</tr>" +
      "<tr><th class='l'>no trade</th><td><b>" + fmt(r.nu) + "</b><span>0 kr</span></td><td><b>" + fmt(r.nd) + "</b><span>0 kr</span></td></tr>";
  }
  document.getElementById("th-t").addEventListener("input", draw);
  draw();
})();
</script>
"""


def threshold(data):
    return _fill(THRESH, data=data["trade"])


# ============================================================ the cost of trading moves the best threshold
COST = r"""
<style>__BASECSS__
.w-co-top { display: flex; align-items: center; gap: 1.2em; margin-bottom: 0.1em; }
.w-co-top label { font: 600 0.52em "Segoe UI", sans-serif; color: #0d366b; display: flex; align-items: center; gap: 0.5em; white-space: nowrap; }
.w-co-top input { width: 15em; }
</style>
<div id="co">
<div class="w-co-top"><label>cost per MWh traded <input class="w-range" type="range" id="co-c" min="0" max="150" step="5" value="__COST__" aria-label="the cost"> <span class="w-val" id="co-cv"></span></label><span class="w-read" id="co-read"></span></div>
<svg id="co-svg" width="1040" height="324" viewBox="0 0 1040 324" role="img" aria-label="Kroner earned in 2024 against the threshold at the chosen cost, with the best threshold marked"></svg>
</div>
<script>
(function () {
__HELPERS__
  var D = __DATA__;
  var svg = document.getElementById("co-svg"); if (!svg) return;
  var L = 100, R = 1000, T = 26, B = 270;
  function X(t) { return L + (t - 0.5) / 0.45 * (R - L); }
  function trades(r) { return r.lu[0] + r.ld[0] + r.su[0] + r.sd[0]; }
  function draw() {
    var c = +document.getElementById("co-c").value, vals = D.rows.map(function (r) { return Math.round(r.lu[1] + r.ld[1] + r.su[1] + r.sd[1] - c * trades(r)); }), best = 0;
    vals.forEach(function (v, i) { if (v > vals[best]) best = i; });
    var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals), pad = Math.max(15000, (hi - lo) * 0.15); lo -= pad; hi += pad;
    function Y(k) { return B - (k - lo) / (hi - lo) * (B - T); }
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    niceTicks(lo, hi, 4).forEach(function (k) { el("line", { x1: L, y1: Y(k), x2: R, y2: Y(k), stroke: Math.abs(k) < 1 ? "#c3c2b7" : "#ecebe6" }, svg); txt(svg, L - 8, Y(k) + 4, fmt(k), { anchor: "end" }); });
    [0.5, 0.6, 0.7, 0.8, 0.9].forEach(function (t) { txt(svg, X(t), B + 20, t.toFixed(2), { anchor: "middle" }); });
    txt(svg, (L + R) / 2, B + 44, "threshold", { anchor: "middle", fill: "#52514e", size: 13.5 });
    txt(svg, 18, (T + B) / 2, "kroner earned in 2024", { anchor: "middle", rotate: -90, fill: "#52514e", size: 13 });
    el("polyline", { points: D.rows.map(function (q, i) { return X(q.thr) + "," + Y(vals[i]); }).join(" "), fill: "none", stroke: "#1c5cab", "stroke-width": 2.8, "stroke-linejoin": "round" }, svg);
    var bt = D.rows[best].thr;
    el("line", { x1: X(bt), y1: Y(vals[best]), x2: X(bt), y2: B, stroke: "#b3402f", "stroke-width": 1.3, "stroke-dasharray": "5 4" }, svg);
    el("circle", { cx: X(bt), cy: Y(vals[best]), r: 7.5, fill: "#b3402f", stroke: "#fcfcfb", "stroke-width": 2 }, svg);
    txt(svg, X(bt), Y(vals[best]) - 14, "best: " + bt.toFixed(2), { anchor: "middle", fill: "#b3402f", weight: 700, size: 15, halo: true });
    document.getElementById("co-cv").textContent = c + " kr";
    document.getElementById("co-read").innerHTML = "at the best threshold <b>" + fmt(vals[best]) + "</b> kr; at 0.50 <b>" + fmt(vals[0]) + "</b> kr";
  }
  document.getElementById("co-c").addEventListener("input", draw);
  draw();
})();
</script>
"""


def cost(data):
    return _fill(COST, data=data["trade"], cost=str(data["trade"]["cost"]))


# ============================================================ similar hours
KNN = r"""
<style>__BASECSS__
.w-knn-ctl { display: grid; grid-template-columns: repeat(3, auto); gap: 0.1em 1.3em; align-items: center; justify-content: start; margin-bottom: 0.1em; }
.w-knn-ctl label { font: 600 0.5em "Segoe UI", sans-serif; color: #0d366b; white-space: nowrap; display: flex; align-items: center; gap: 0.45em; }
.w-knn-ctl input[type=range] { width: 8.5em; }
.w-knn-ctl .w-val { min-width: 5.4em; }
.w-knn-row { display: flex; gap: 18px; align-items: flex-start; }
.w-knn-row > * { flex: 0 0 auto; }
</style>
<div id="knn">
<div class="w-knn-ctl">
<label>wind change <input class="w-range" type="range" id="knn-w" min="-3000" max="3000" step="50" value="1000"> <span class="w-val" id="knn-wv"></span></label>
<label>today's move <input class="w-range" type="range" id="knn-m" min="-1000" max="1000" step="10" value="0"> <span class="w-val" id="knn-mv"></span></label>
<label>neighbours, k <input class="w-range" type="range" id="knn-k" min="1" max="501" step="2" value="201"> <span class="w-val" id="knn-kv"></span></label>
<label>weekday <input class="w-range" type="range" id="knn-d" min="0" max="6" step="1" value="2"> <span class="w-val" id="knn-dv"></span></label>
<label>hour <input class="w-range" type="range" id="knn-h" min="0" max="23" value="18"> <span class="w-val" id="knn-hv"></span></label>
<label><input type="checkbox" id="knn-sc" checked> scale the columns first</label>
</div>
<div class="w-read" id="knn-read" style="margin-bottom:4px"></div>
<div class="w-knn-row">
<canvas id="knn-cv" role="img" aria-label="Every hour of 2022 and 2023 placed by its wind change and today's move, with the k most similar hours to the chosen one coloured by whether the price went up"></canvas>
<svg id="knn-side" width="320" height="330" viewBox="0 0 320 330" role="img" aria-label="How many of the neighbours fall on each weekday"></svg>
</div>
</div>
<script>
(function () {
__HELPERS__
  var D = __DATA__;
  var cv = document.getElementById("knn-cv"); if (!cv) return;
  var N = D.up.length, W = 690, H = 330, L = 62, R = 678, T = 8, B = 284, XM = 3500, YM = 1200;
  var c = hidpi(cv, W, H);
  function X(w) { return L + (Math.max(-XM, Math.min(XM, w)) + XM) / (2 * XM) * (R - L); } function Y(m) { return B - (Math.max(-YM, Math.min(YM, m)) + YM) / (2 * YM) * (B - T); }
  var cols = [D.wind_change, D.change_before, D.weekday, D.hour], dist = new Float64Array(N), idx = new Uint32Array(N);
  function run() {
    var q = [+document.getElementById("knn-w").value, +document.getElementById("knn-m").value, +document.getElementById("knn-d").value, +document.getElementById("knn-h").value];
    var k = +document.getElementById("knn-k").value, sc = document.getElementById("knn-sc").checked;
    document.getElementById("knn-wv").textContent = signed(q[0]) + " MW"; document.getElementById("knn-mv").textContent = signed(q[1]) + " kr";
    document.getElementById("knn-dv").textContent = WDNAMES[q[2]]; document.getElementById("knn-hv").textContent = hh(q[3]); document.getElementById("knn-kv").textContent = k;
    for (var i = 0; i < N; i++) { var t = 0; for (var j = 0; j < 4; j++) { var z = cols[j][i] - q[j]; if (sc) z /= D.sd[j]; t += z * z; } dist[i] = t; idx[i] = i; }
    var order = Array.prototype.slice.call(idx).sort(function (a, b) { return dist[a] - dist[b]; }).slice(0, k);
    var near = new Uint8Array(N), ups = 0, byDay = [0, 0, 0, 0, 0, 0, 0];
    order.forEach(function (i) { near[i] = 1; ups += D.up[i]; byDay[D.weekday[i]]++; });
    c.clearRect(0, 0, W, H);
    c.lineWidth = 1; c.font = "12px Segoe UI, sans-serif";
    [-3000, -2000, -1000, 0, 1000, 2000, 3000].forEach(function (w) { c.beginPath(); c.moveTo(X(w), T); c.lineTo(X(w), B); c.strokeStyle = w === 0 ? "#c3c2b7" : "#ecebe6"; c.stroke(); c.fillStyle = "#898781"; c.textAlign = "center"; c.textBaseline = "top"; c.fillText((w > 0 ? "+" : "") + fmt(w), X(w), B + 5); });
    [-1000, -500, 0, 500, 1000].forEach(function (m) { c.beginPath(); c.moveTo(L, Y(m)); c.lineTo(R, Y(m)); c.strokeStyle = m === 0 ? "#c3c2b7" : "#ecebe6"; c.stroke(); c.fillStyle = "#898781"; c.textAlign = "right"; c.textBaseline = "middle"; c.fillText((m > 0 ? "+" : "") + fmt(m), L - 6, Y(m)); });
    c.fillStyle = "#52514e"; c.font = "13px Segoe UI, sans-serif"; c.textAlign = "center"; c.textBaseline = "top"; c.fillText("wind change, MW", (L + R) / 2, B + 24);
    c.save(); c.translate(14, (T + B) / 2); c.rotate(-Math.PI / 2); c.fillText("today's move, kr", 0, -6); c.restore();
    c.fillStyle = "#d9d8d2"; for (var i2 = 0; i2 < N; i2++) if (!near[i2]) c.fillRect(X(cols[0][i2]) - 0.9, Y(cols[1][i2]) - 0.9, 1.8, 1.8);
    order.forEach(function (i) { c.fillStyle = D.up[i] ? "#b3402f" : "#1c5cab"; c.fillRect(X(cols[0][i]) - 2.2, Y(cols[1][i]) - 2.2, 4.4, 4.4); });
    c.strokeStyle = "#141413"; c.lineWidth = 1.8; c.beginPath(); c.arc(X(q[0]), Y(q[1]), 8, 0, 2 * Math.PI); c.stroke();
    c.beginPath(); c.moveTo(X(q[0]) - 13, Y(q[1])); c.lineTo(X(q[0]) + 13, Y(q[1])); c.moveTo(X(q[0]), Y(q[1]) - 13); c.lineTo(X(q[0]), Y(q[1]) + 13); c.stroke();
    var g = document.getElementById("knn-side"); while (g.firstChild) g.removeChild(g.firstChild);
    var topD = Math.max.apply(null, byDay) || 1;
    txt(g, 8, 18, "the neighbours, by weekday", { size: 14, fill: "#52514e", weight: 600 });
    byDay.forEach(function (n, d) { var x = 10 + d * 43; el("rect", { x: x, y: 262 - n / topD * 214, width: 33, height: Math.max(0.5, n / topD * 214), rx: 2, fill: d === q[2] ? "#0d366b" : "#86b6ef" }, g); txt(g, x + 16.5, 282, WDNAMES[d].slice(0, 3), { anchor: "middle", size: 12.5, fill: d === q[2] ? "#141413" : "#898781", weight: d === q[2] ? 700 : 400 }); });
    document.getElementById("knn-read").innerHTML = "Of the <b>" + k + "</b> most similar hours of 2022 and 2023, <b>" + ups + "</b> went up (red), so k-NN says <b>" + (ups / k).toFixed(2) + "</b>.";
  }
  ["knn-w", "knn-m", "knn-d", "knn-h", "knn-k"].forEach(function (id) { document.getElementById(id).addEventListener("input", run); });
  document.getElementById("knn-sc").addEventListener("change", run);
  run();
})();
</script>
"""


def knn(data):
    k = data["knn"]
    return _fill(KNN, data={x: k[x] for x in ("wind_change", "change_before", "weekday", "hour", "up", "sd")})


# ============================================================ hours above or below a price, by year
HOURS = r"""
<style>__BASECSS__
.w-hr-top { display: flex; align-items: center; gap: 1.2em; margin-bottom: 0.15em; }
.w-hr-top label { font: 600 0.52em "Segoe UI", sans-serif; color: #0d366b; display: flex; align-items: center; gap: 0.5em; white-space: nowrap; }
.w-hr-top input { width: 22em; }
</style>
<div id="hr">
<div class="w-hr-top"><span class="w-marks" id="hr-side"></span><label>price <input class="w-range" type="range" id="hr-p" min="0" max="__NMAX__" value="__START__" aria-label="the price"> <span class="w-val" id="hr-pv"></span></label></div>
<svg id="hr-svg" width="1040" height="320" viewBox="0 0 1040 320" role="img" aria-label="For 2022, 2023 and 2024, the number of hours with a price below or above the chosen level"></svg>
</div>
<script>
(function () {
__HELPERS__
  var D = __DATA__;
  var svg = document.getElementById("hr-svg"); if (!svg) return;
  var side = "below", cur = [0, 0, 0], curTop = 1000, anim = null, L = 90, R = 1000, T = 40, B = 280;
  function niceTop(v) { var t = niceTicks(0, Math.max(10, v) * 1.18, 4); return t[t.length - 1] >= v * 1.1 ? t[t.length - 1] : v * 1.2; }
  function frame(vals, top) {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    var lev = D.levels[+document.getElementById("hr-p").value];
    function Y(v) { return B - v / top * (B - T); }
    niceTicks(0, top, 4).forEach(function (q) { if (q > top) return; el("line", { x1: L, y1: Y(q), x2: R, y2: Y(q), stroke: q === 0 ? "#c3c2b7" : "#ecebe6" }, svg); txt(svg, L - 8, Y(q) + 4, fmt(q), { anchor: "end" }); });
    txt(svg, L, 20, "hours " + side + " " + fmt(lev) + " kr per MWh", { fill: "#0d366b", size: 16, weight: 700 });
    vals.forEach(function (v, k) {
      var x = L + (k + 0.5) / 3 * (R - L);
      el("rect", { x: x - 90, y: Y(v), width: 180, height: Math.max(0, B - Y(v)), rx: 3, fill: "#1c5cab" }, svg);
      txt(svg, x, Y(v) - 10, fmt(v) + " hours, " + (100 * v / D.total[k]).toFixed(1) + "%", { anchor: "middle", fill: "#141413", size: 15, weight: 700, halo: true });
      txt(svg, x, B + 24, String(D.years[k]), { anchor: "middle", fill: "#52514e", size: 15, weight: 600 });
    });
  }
  function go() {
    var i = +document.getElementById("hr-p").value, to = D[side].map(function (row) { return row[i]; }), top = niceTop(Math.max.apply(null, to));
    var from = cur.slice(), t0 = null, top0 = curTop; if (anim) cancelAnimationFrame(anim);
    document.getElementById("hr-pv").textContent = fmt(D.levels[i]) + " kr";
    function step(t) { if (t0 === null) t0 = t; var u = Math.min(1, (t - t0) / 350), e = 1 - Math.pow(1 - u, 3); cur = from.map(function (v, k) { return lerp(v, to[k], e); }); curTop = lerp(top0, top, e); frame(cur.map(Math.round), curTop); if (u < 1) anim = requestAnimationFrame(step); }
    anim = requestAnimationFrame(step);
    Array.prototype.forEach.call(document.getElementById("hr-side").children, function (b) { b.classList.toggle("on", b.dataset.s === side); });
  }
  ["below", "above"].forEach(function (s) { var b = document.createElement("button"); b.className = "w-btn"; b.dataset.s = s; b.textContent = s; b.addEventListener("click", function () { side = s; go(); }); document.getElementById("hr-side").appendChild(b); });
  document.getElementById("hr-p").addEventListener("input", go);
  cur = D.below.map(function (row) { return row[__START__]; }); curTop = niceTop(Math.max.apply(null, cur));
  go();
})();
</script>
"""


def hours(data):
    h = data["hours"]
    return _fill(HOURS, data=h, nmax=str(len(h["levels"]) - 1), start=str(h["levels"].index(0)))


# ============================================================ the median price by band of the wind or the sun forecast
BANDS = r"""
<style>__BASECSS__
.w-bd-top { display: flex; align-items: center; gap: 1.4em; margin-bottom: 0.15em; }
</style>
<div id="bd">
<div class="w-bd-top"><span class="w-marks" id="bd-col"></span><span class="w-marks" id="bd-year"></span></div>
<svg id="bd-svg" width="1040" height="330" viewBox="0 0 1040 330" role="img" aria-label="The median price in each band of the wind or the sun forecast"></svg>
</div>
<script>
(function () {
__HELPERS__
  var D = __DATA__;
  var svg = document.getElementById("bd-svg"); if (!svg) return;
  var COL = "wind", YEAR = "all", cur = null, curTop = 1, anim = null, L = 90, R = 1000, T = 40, B = 262;
  var COLS = { wind: "the wind forecast", solar: "the sun forecast" }, YEARS = { all: "2022 to 2024", "2022": "2022", "2023": "2023", "2024": "2024" }, ORDER = ["all", "2022", "2023", "2024"];
  function frame(vals, top) {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    function Y(v) { return B - v / top * (B - T); }
    niceTicks(0, top, 4).forEach(function (q) { if (q > top) return; el("line", { x1: L, y1: Y(q), x2: R, y2: Y(q), stroke: q === 0 ? "#c3c2b7" : "#ecebe6" }, svg); txt(svg, L - 8, Y(q) + 4, fmt(q), { anchor: "end" }); });
    txt(svg, L, 20, "the median price, kr per MWh, by " + COLS[COL] + ", " + YEARS[YEAR], { fill: "#0d366b", size: 16, weight: 700 });
    var S = D[COL];
    vals.forEach(function (v, k) {
      var x = L + (k + 0.5) / 4 * (R - L);
      el("rect", { x: x - 80, y: Y(v), width: 160, height: Math.max(0, B - Y(v)), rx: 3, fill: COL === "wind" ? "#1c5cab" : "#b8860b" }, svg);
      txt(svg, x, Y(v) - 10, fmt(v) + " kr", { anchor: "middle", fill: "#141413", size: 15, weight: 700, halo: true });
      txt(svg, x, B + 22, S.names[k], { anchor: "middle", fill: "#141413", size: 14.5, weight: 600 });
      txt(svg, x, B + 42, fmt(S[YEAR].n[k]) + " hours", { anchor: "middle", fill: "#898781", size: 13 });
    });
  }
  function go() {
    var to = D[COL][YEAR].median.map(function (v) { return v === null ? 0 : v; }), both = D.wind[YEAR].median.concat(D.solar[YEAR].median).filter(function (v) { return v !== null; });
    var top = niceTicks(0, Math.max.apply(null, both) * 1.15, 4).pop(), from = cur || to.slice(), top0 = cur ? curTop : top, t0 = null; if (anim) cancelAnimationFrame(anim);
    function step(t) { if (t0 === null) t0 = t; var u = Math.min(1, (t - t0) / 450), e = 1 - Math.pow(1 - u, 3); cur = from.map(function (v, k) { return lerp(v, to[k], e); }); curTop = lerp(top0, top, e); frame(cur.map(Math.round), curTop); if (u < 1) anim = requestAnimationFrame(step); }
    anim = requestAnimationFrame(step);
    Array.prototype.forEach.call(document.getElementById("bd-col").children, function (b) { b.classList.toggle("on", b.dataset.v === COL); });
    Array.prototype.forEach.call(document.getElementById("bd-year").children, function (b) { b.classList.toggle("on", b.dataset.v === YEAR); });
  }
  Object.keys(COLS).forEach(function (c) { var b = document.createElement("button"); b.className = "w-btn"; b.dataset.v = c; b.textContent = COLS[c]; b.addEventListener("click", function () { COL = c; go(); }); document.getElementById("bd-col").appendChild(b); });
  ORDER.forEach(function (y) { var b = document.createElement("button"); b.className = "w-btn"; b.dataset.v = y; b.textContent = y === "all" ? "all three years" : y; b.addEventListener("click", function () { YEAR = y; go(); }); document.getElementById("bd-year").appendChild(b); });
  go();
})();
</script>
"""


def bands(data):
    return _fill(BANDS, data=data["bands"])


# ============================================================ C as a slider: the coefficients and the probabilities
CPATH = r"""
<style>__BASECSS__
.w-cp-top { display: flex; align-items: center; gap: 1.4em; margin-bottom: 0.1em; }
.w-cp-top label { font: 600 0.52em "Segoe UI", sans-serif; color: #0d366b; display: flex; align-items: center; gap: 0.5em; white-space: nowrap; }
.w-cp-top input { width: 16em; }
.w-cp-row { display: flex; gap: 22px; align-items: flex-start; }
</style>
<div id="cp">
<div class="w-cp-top"><label>C <input class="w-range" type="range" id="cp-c" min="0" max="__NMAX__" value="__NMAX__" aria-label="C, from 0.0001 to 10"> <span class="w-val" id="cp-cv"></span></label><span class="w-read" id="cp-read"></span></div>
<div class="w-cp-row">
<svg id="cp-coef" width="520" height="326" viewBox="0 0 520 326" role="img" aria-label="The logistic regression's nine coefficients at the chosen C"></svg>
<svg id="cp-hist" width="490" height="326" viewBox="0 0 490 326" role="img" aria-label="How the model's probabilities for 2024's hours are spread at the chosen C"></svg>
</div>
</div>
<script>
(function () {
__HELPERS__
  var D = __DATA__;
  var sc = document.getElementById("cp-coef"); if (!sc) return;
  var HTOP = 0; D.rows.forEach(function (r) { r.hist.forEach(function (n) { HTOP = Math.max(HTOP, n); }); }); HTOP *= 1.08;
  function cText(c) { return c >= 1 ? String(Math.round(c * 100) / 100) : String(Number(c.toPrecision(2))); }
  function draw() {
    var r = D.rows[+document.getElementById("cp-c").value];
    while (sc.firstChild) sc.removeChild(sc.firstChild);
    var L = 120, R = 508, T = 34, ROW = 30, XM = 1.6;
    function X(v) { return (L + R) / 2 + v / XM * (R - L) / 2; }
    txt(sc, 0, 16, "the coefficients, on scaled columns", { fill: "#0d366b", size: 15, weight: 700 });
    [-1.5, -1, -0.5, 0, 0.5, 1, 1.5].forEach(function (v) { el("line", { x1: X(v), y1: T - 6, x2: X(v), y2: T + 9 * ROW, stroke: v === 0 ? "#898781" : "#ecebe6" }, sc); txt(sc, X(v), T + 9 * ROW + 18, (v > 0 ? "+" : "") + v, { anchor: "middle", size: 12 }); });
    r.coef.forEach(function (v, k) {
      var y = T + k * ROW;
      txt(sc, L - 10, y + ROW / 2 + 5, D.names[k], { anchor: "end", fill: "#141413", size: 14 });
      el("rect", { x: Math.min(X(0), X(v)), y: y + 6, width: Math.max(1, Math.abs(X(v) - X(0))), height: ROW - 12, rx: 2, fill: v > 0 ? "#b3402f" : "#1c5cab" }, sc);
    });
    var sh = document.getElementById("cp-hist"); while (sh.firstChild) sh.removeChild(sh.firstChild);
    var L2 = 44, R2 = 466, T2 = 34, B2 = 270;
    txt(sh, 0, 16, "the model's probabilities for the hours of 2024", { fill: "#0d366b", size: 15, weight: 700 });
    function X2(p) { return L2 + p * (R2 - L2); } function Y2(n) { return B2 - n / HTOP * (B2 - T2); }
    el("line", { x1: L2, y1: B2, x2: R2, y2: B2, stroke: "#c3c2b7" }, sh);
    r.hist.forEach(function (n, b) { el("rect", { x: X2(b / 20) + 1, y: Y2(n), width: (R2 - L2) / 20 - 2, height: B2 - Y2(n), rx: 1.5, fill: "#1c5cab" }, sh); });
    [0, 0.25, 0.5, 0.75, 1].forEach(function (p) { txt(sh, X2(p), B2 + 18, p.toFixed(2), { anchor: "middle", size: 12 }); });
    txt(sh, (L2 + R2) / 2, B2 + 42, "the probability that the price rises", { anchor: "middle", fill: "#52514e", size: 13 });
    document.getElementById("cp-cv").textContent = cText(r.C);
    document.getElementById("cp-read").innerHTML = "AUC on the folds <b>" + r.folds.toFixed(3) + "</b> &middot; Brier score on 2024 <b>" + r.brier.toFixed(3) + "</b>";
  }
  document.getElementById("cp-c").addEventListener("input", draw);
  draw();
})();
</script>
"""


def cpath(data):
    c = data["cpath"]
    return _fill(CPATH, data=c, nmax=str(len(c["rows"]) - 1))


# ============================================================ the feature lab: a checklist of columns, and the score
LABW = r"""
<style>__BASECSS__
.w-lb-labels { display: flex; align-items: center; gap: 0.45em; margin-bottom: 0.15em; }
.w-lb-labels span.k { font: 600 0.46em "Segoe UI", sans-serif; color: #898781; margin-right: 0.3em; }
.w-lb { display: flex; gap: 30px; align-items: flex-start; }
.w-lb-list { flex: 0 0 660px; }
.w-lb-h { font: 700 13px "Segoe UI", sans-serif; color: #898781; letter-spacing: 0.06em; text-transform: uppercase; margin: 6px 0 2px 6px; }
.w-lb-item { display: flex; align-items: center; gap: 11px; padding: 3px 6px; border-radius: 6px; cursor: pointer; font: 18px "Segoe UI", sans-serif; user-select: none; }
.w-lb-item:hover { background: #f1f0ec; }
.w-lb-box { width: 18px; height: 18px; border: 2px solid #86b6ef; border-radius: 4px; flex: 0 0 16px; position: relative; background: #fff; }
.w-lb-item.on .w-lb-box { background: #0d366b; border-color: #0d366b; }
.w-lb-item.on .w-lb-box::after { content: ""; position: absolute; left: 5px; top: 1px; width: 5px; height: 10px; border: solid #fff; border-width: 0 2.5px 2.5px 0; transform: rotate(45deg); }
.w-lb-name { font-weight: 600; color: #0d366b; flex: 0 0 13.2em; }
.w-lb-desc { color: #52514e; font-size: 16.5px; }
.w-lb-code { font: 14px "Cascadia Mono", Consolas, monospace; color: #52514e; background: #f5f5f2; border-radius: 6px; padding: 6px 10px; margin-top: 8px; white-space: nowrap; overflow: hidden; min-height: 17px; }
.w-lb-res { flex: 1 1 auto; font: 16px "Segoe UI", sans-serif; color: #52514e; padding-top: 4px; }
.w-lb-model { margin: 2px 0 18px; }
.w-lb-model .nm { font-weight: 700; color: #141413; font-size: 17.5px; }
.w-lb-big { display: flex; align-items: baseline; gap: 12px; margin: 0 0 5px; }
.w-lb-big b { font: 700 36px "Cascadia Mono", Consolas, monospace; color: #0d366b; }
.w-lb-delta { font: 700 18px "Cascadia Mono", Consolas, monospace; }
.w-lb-track { position: relative; height: 14px; background: #f1f0ec; border-radius: 4px; }
.w-lb-fill { position: absolute; left: 0; top: 0; bottom: 0; width: 0; border-radius: 4px; background: #1c5cab; transition: width 0.5s ease; }
.w-lb-mark { position: absolute; top: -5px; bottom: -5px; width: 0; border-left: 2px dashed #b3402f; }
.w-lb-sub { font-size: 15px; color: #898781; margin-top: 4px; }
.w-lb-foot { font-size: 15px; color: #52514e; margin-top: 4px; }
.w-lb-foot b { font-family: "Cascadia Mono", Consolas, monospace; color: #0d366b; }
.w-lb-foot button { font: inherit; color: #1c5cab; background: none; border: none; padding: 0; cursor: pointer; text-decoration: underline; }
</style>
<div id="__ID__">
<div class="w-lb-labels" id="__ID__-labels"></div>
<div class="w-lb">
<div class="w-lb-list"><div id="__ID__-list"></div><div class="w-lb-code" id="__ID__-code"></div></div>
<div class="w-lb-res"><div class="w-lb-sub" style="margin: 0 0 10px">AUC on the folds of 2022 and 2023, where the columns are chosen. <span style="color:#b3402f">Dashed: the model on the slides.</span></div><div id="__ID__-models"></div><div class="w-lb-foot" id="__ID__-foot"></div></div>
</div>
</div>
<script>
(function () {
__HELPERS__
  var D = __DATA__, ID = "__ID__", KEY = "mlfin10-" + ID, LO = 0.45, HI = 0.9;
  var root = document.getElementById(ID); if (!root) return;
  var label = D.labels[0], mask = D.start, last = null, prev = null, best = recall(KEY) || {};
  var LABELS = { up: "tomorrow higher than today", big: "a move of more than 150 kr either way" };
  var MODELS = { logit: ["logistic regression", 0], knn: ["k-NN, k = 201", 1] };
  function pct(v) { return ((Math.max(LO, v) - LO) / (HI - LO) * 100).toFixed(2) + "%"; }
  var lab = document.getElementById(ID + "-labels");
  if (D.labels.length > 1) {
    var k0 = document.createElement("span"); k0.className = "k"; k0.textContent = "predict"; lab.appendChild(k0);
    D.labels.forEach(function (l) { var b = document.createElement("button"); b.className = "w-btn"; b.dataset.l = l; b.textContent = LABELS[l]; b.addEventListener("click", function () { label = l; prev = null; render(false); }); lab.appendChild(b); });
  } else lab.style.display = "none";
  var list = document.getElementById(ID + "-list");
  [["raw", "columns in the table"], ["built", "columns built from them"]].forEach(function (grp) {
    var h = document.createElement("div"); h.className = "w-lb-h"; h.textContent = grp[1]; list.appendChild(h);
    D.chips.forEach(function (ch, i) {
      if (ch.kind !== grp[0] || D.hide.indexOf(i) >= 0) return;
      var it = document.createElement("div"); it.className = "w-lb-item"; it.dataset.i = i;
      it.innerHTML = "<span class='w-lb-box'></span><span class='w-lb-name'>" + ch.name + "</span><span class='w-lb-desc'>" + ch.desc + "</span>";
      it.addEventListener("click", function () { prev = D[label][mask]; mask ^= (1 << i); last = i; render(true); });
      list.appendChild(it);
    });
  });
  var models = document.getElementById(ID + "-models");
  D.models.forEach(function (m) {
    var j = MODELS[m][1], blk = document.createElement("div"); blk.className = "w-lb-model";
    blk.innerHTML = "<div class='nm'>" + MODELS[m][0] + "</div><div class='w-lb-big'><b id='" + ID + "-v" + j + "'></b><span class='w-lb-delta' id='" + ID + "-d" + j + "'></span></div><div class='w-lb-track'><div class='w-lb-fill' id='" + ID + "-f" + j + "'" + (j ? " style='background:#0d366b'" : "") + "></div><div class='w-lb-mark' id='" + ID + "-m" + j + "'></div></div><div class='w-lb-sub' id='" + ID + "-t" + j + "'></div>";
    models.appendChild(blk);
  });
  function render(changed) {
    Array.prototype.forEach.call(list.querySelectorAll(".w-lb-item"), function (it) { it.classList.toggle("on", !!(mask >> +it.dataset.i & 1)); });
    Array.prototype.forEach.call(lab.querySelectorAll("button"), function (b) { b.classList.toggle("on", b.dataset.l === label); });
    document.getElementById(ID + "-code").textContent = last === null ? "Tick a column to use it; the pandas line that builds it shows here." : D.chips[last].name + ":   " + D.chips[last].code;
    var r = D[label][mask];
    D.models.forEach(function (m) {
      var j = MODELS[m][1], v = document.getElementById(ID + "-v" + j), d = document.getElementById(ID + "-d" + j), f = document.getElementById(ID + "-f" + j), t = document.getElementById(ID + "-t" + j);
      document.getElementById(ID + "-m" + j).style.left = pct(D[label][D.slides[label][j]][2 * j]);
      if (!r) { v.textContent = "-"; d.textContent = ""; f.style.width = "0%"; t.textContent = "tick at least one column"; return; }
      v.textContent = r[2 * j].toFixed(3); f.style.width = pct(r[2 * j]); t.textContent = "on 2024: " + r[2 * j + 1].toFixed(3);
      if (changed && prev) { var dv = r[2 * j] - prev[2 * j]; d.textContent = (dv >= 0 ? "+" : "−") + Math.abs(dv).toFixed(3); d.style.color = dv >= 0 ? "#1c5cab" : "#b3402f"; } else d.textContent = "";
      var key = label + j; if (!best[key] || r[2 * j] > best[key][0]) best[key] = [r[2 * j], mask];
    });
    store(KEY, best);
    var parts = D.models.map(function (m) { var j = MODELS[m][1], b = best[label + j]; return b ? MODELS[m][0] + " <b>" + b[0].toFixed(3) + "</b>" : ""; }).filter(function (s) { return s; });
    document.getElementById(ID + "-foot").innerHTML = (parts.length ? "Your best on the folds: " + parts.join(", ") + ". " : "") + "<button>clear</button>";
    document.getElementById(ID + "-foot").querySelector("button").addEventListener("click", function () { best = {}; store(KEY, best); render(false); });
  }
  render(false);
})();
</script>
"""

LAB_DESC = {"tomorrow's wind": "the wind forecast for the hour", "today's price": "the price at the same hour today",
            "weekday": "0 for Monday to 6 for Sunday", "hour": "the clock hour, 0 to 23",
            "wind change": "tomorrow's wind forecast minus today's", "today's move": "today's price minus yesterday's",
            "sun change": "tomorrow's sun forecast minus today's", "weekday as 7 columns": "one column of 0 and 1 per weekday",
            "size of the wind change": "the wind change without its sign"}


def lab(data, prefix="lab", models=("logit", "knn"), labels=("up", "big"), hide=()):
    L = data["lab"]
    names = [c["name"] for c in L["chips"]]

    def mask(*chips):
        return sum(1 << names.index(c) for c in chips)

    chips = [dict(c, desc=LAB_DESC[c["name"]]) for c in L["chips"]]
    payload = {"chips": chips, "up": L["up"], "big": L["big"] if "big" in labels else None,
               "labels": list(labels), "models": list(models), "hide": [names.index(h) for h in hide],
               "start": mask("wind change"),
               "slides": {"up": [mask("wind change", "today's move", "weekday as 7 columns"),
                                 mask("wind change", "today's move", "weekday", "hour")],
                          "big": [mask("wind change", "today's move", "weekday as 7 columns", "size of the wind change"),
                                  mask("wind change", "today's move", "weekday", "hour")]}}
    return _fill(LABW, data=payload, id=prefix)
