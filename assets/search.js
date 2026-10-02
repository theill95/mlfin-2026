/* The course search, on search.html. It searches assets/search-index.json,
   which tools/build_search.py builds: every lecture slide, exercise, case
   question, cheatsheet row, glossary term and page section. Every word typed
   has to appear; a word in a title counts most. */
(function () {
  "use strict";

  var KINDS = [
    ["all", "Everything"], ["lecture", "Lectures"], ["exercise", "Exercises"],
    ["case", "The case"], ["cheatsheet", "Cheatsheet"], ["glossary", "Glossary"],
    ["maths", "Maths"], ["page", "Pages"]
  ];
  // Short, exact entries first when the words match equally well.
  var BOOST = { glossary: 6, cheatsheet: 4, maths: 3, page: 2, lecture: 1, "case": 0, exercise: 0 };
  var TRY = ["GridSearchCV", "out-of-bag", "threshold", "lasso", "groupby", "AUC", "persistence"];
  // The pages spell the British way and the libraries the American way: a word
  // typed either way finds both.
  var SPELLINGS = [["standardiz", "standardis"], ["normaliz", "normalis"], ["regulariz", "regularis"],
    ["penaliz", "penalis"], ["optimiz", "optimis"], ["minimiz", "minimis"], ["maximiz", "maximis"],
    ["generaliz", "generalis"], ["summariz", "summaris"], ["visualiz", "visualis"], ["analyz", "analys"],
    ["initializ", "initialis"], ["neighbor", "neighbour"], ["color", "colour"], ["behavior", "behaviour"],
    ["modeling", "modelling"], ["center", "centre"]];
  var STEP = 30;

  var form = document.getElementById("search-form");
  var input = document.getElementById("q");
  var kindsEl = document.getElementById("kinds");
  var statusEl = document.getElementById("status");
  var listEl = document.getElementById("results");
  var moreEl = document.getElementById("more");
  var items = null, kind = "all", limit = STEP, timer = null;

  function norm(s) {
    return (s || "").toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "");
  }
  function esc(s) {
    return s.replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;" }[c];
    });
  }
  function reEsc(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }

  // Escape a piece of text and mark every word in it.
  function marked(text, re) {
    if (!re) return esc(text);
    var out = "", at = 0, m;
    re.lastIndex = 0;
    while ((m = re.exec(text)) !== null) {
      if (!m[0]) { re.lastIndex++; continue; }
      out += esc(text.slice(at, m.index)) + "<mark>" + esc(m[0]) + "</mark>";
      at = m.index + m[0].length;
    }
    return out + esc(text.slice(at));
  }

  // About two lines of the text, around the first word found in it.
  function snippet(text, words, re) {
    if (!text) return "";
    if (text.length <= 240) return marked(text, re);
    var low = norm(text), at = -1;
    for (var i = 0; i < words.length && at === -1; i++) at = low.indexOf(words[i]);
    var start = at > 80 ? at - 80 : 0;
    var end = Math.min(text.length, start + 230);
    if (start > 0) {
      var sp = text.indexOf(" ", start);
      if (sp !== -1 && sp < (at === -1 ? end : at)) start = sp + 1;
    }
    if (end < text.length) {
      var sp2 = text.lastIndexOf(" ", end);
      if (sp2 > start) end = sp2;
    }
    return (start > 0 ? "… " : "") + marked(text.slice(start, end), re) + (end < text.length ? " …" : "");
  }

  // A word, and the same word in the other spelling if it has one.
  function spellings(w) {
    for (var i = 0; i < SPELLINGS.length; i++) {
      var a = SPELLINGS[i][0], b = SPELLINGS[i][1];
      if (w.indexOf(b) !== -1) return [w, w.replace(b, a)];
      if (w.indexOf(a) !== -1) return [w, w.replace(a, b)];
    }
    return [w];
  }

  function found(hay, forms) {
    for (var i = 0; i < forms.length; i++) if (hay.indexOf(forms[i]) !== -1) return forms[i];
    return null;
  }

  function score(it, words, whole, phrase) {
    var s = 0;
    for (var i = 0; i < words.length; i++) {
      var w = found(it.nt, words[i]);
      if (w) {
        s += 10;
        if (it.nt.indexOf(w) === 0) s += 4;
        if (whole[i].test(it.nt)) s += 4;
      } else if (found(it.nx, words[i])) {
        s += 3;
      } else if (found(it.nc, words[i])) {
        s += 2;
      } else if (found(it.nl, words[i])) {
        s += 1;
      } else {
        return -1;
      }
    }
    if (words.length > 1) {
      if (it.nt.indexOf(phrase) !== -1) s += 12;
      else if (it.nx.indexOf(phrase) !== -1) s += 4;
    }
    return s + (BOOST[it.k] || 0);
  }

  function setUrl() {
    var q = input.value.trim(), params = [];
    if (q) params.push("q=" + encodeURIComponent(q));
    if (kind !== "all") params.push("k=" + kind);
    try { history.replaceState(null, "", location.pathname + (params.length ? "?" + params.join("&") : "")); } catch (e) {}
  }

  function showTries() {
    kindsEl.hidden = true;
    listEl.innerHTML = "";
    moreEl.hidden = true;
    statusEl.innerHTML = "Try " + TRY.map(function (t) {
      return '<button type="button" class="sr-try" data-q="' + esc(t) + '">' + esc(t) + "</button>";
    }).join(" ");
  }

  function render() {
    var typed = input.value.trim();
    var words = norm(typed).split(/\s+/).filter(Boolean);
    if (!items) { statusEl.textContent = "Loading the index…"; return; }
    if (!words.length) { showTries(); return; }

    var phrase = words.join(" ");
    var forms = words.map(spellings);
    var whole = forms.map(function (f) {
      return new RegExp("(^|[^a-z0-9_])(" + f.map(reEsc).join("|") + ")($|[^a-z0-9_])");
    });
    var hits = [];
    for (var i = 0; i < items.length; i++) {
      var s = score(items[i], forms, whole, phrase);
      if (s >= 0) hits.push([s, i]);
    }
    hits.sort(function (a, b) { return b[0] - a[0] || a[1] - b[1]; });

    var counts = { all: hits.length };
    hits.forEach(function (f) { var k = items[f[1]].k; counts[k] = (counts[k] || 0) + 1; });
    if (kind !== "all" && !counts[kind]) kind = "all";

    kindsEl.hidden = hits.length === 0;
    kindsEl.innerHTML = KINDS.filter(function (k) { return counts[k[0]]; }).map(function (k) {
      return '<li><button type="button" data-kind="' + k[0] + '" aria-pressed="' + (k[0] === kind) + '">' +
        esc(k[1]) + ' <span class="n">' + counts[k[0]] + "</span></button></li>";
    }).join("");

    var shown = kind === "all" ? hits : hits.filter(function (f) { return items[f[1]].k === kind; });
    statusEl.textContent = shown.length
      ? shown.length + (shown.length === 1 ? " result" : " results")
      : "Nothing found for “" + typed + "”. Fewer or shorter words find more.";

    var all = [].concat.apply([], forms);
    var re = new RegExp(all.map(reEsc).sort(function (a, b) { return b.length - a.length; }).join("|"), "gi");
    listEl.innerHTML = shown.slice(0, limit).map(function (f) {
      var it = items[f[1]];
      var away = /^https?:/.test(it.u);
      return '<li class="sr-item"><a class="sr-title" href="' + esc(it.u) + '"' +
        (away ? ' target="_blank" rel="noopener"' : "") + ">" + marked(it.t, re) + "</a>" +
        '<span class="sr-label">' + esc(it.l) + (away ? " · opens in Colab" : "") + "</span>" +
        (it.x ? '<p class="sr-snip">' + snippet(it.x, all, re) + "</p>" : "") + "</li>";
    }).join("");
    moreEl.hidden = shown.length <= limit;
    moreEl.textContent = "Show " + Math.min(STEP, shown.length - limit) + " more";
  }

  function later() {
    clearTimeout(timer);
    timer = setTimeout(function () { limit = STEP; render(); setUrl(); }, 120);
  }

  function links() { return [].slice.call(listEl.querySelectorAll("a.sr-title")); }

  input.addEventListener("input", later);
  form.addEventListener("submit", function (e) {
    e.preventDefault();
    var first = links()[0];
    if (first) first.focus();
  });
  input.addEventListener("keydown", function (e) {
    if (e.key === "ArrowDown") {
      var first = links()[0];
      if (first) { e.preventDefault(); first.focus(); }
    }
  });
  listEl.addEventListener("keydown", function (e) {
    if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
    var all = links(), i = all.indexOf(document.activeElement);
    if (i === -1) return;
    e.preventDefault();
    if (e.key === "ArrowDown" && i + 1 < all.length) all[i + 1].focus();
    else if (e.key === "ArrowUp") (i > 0 ? all[i - 1] : input).focus();
  });
  kindsEl.addEventListener("click", function (e) {
    var b = e.target.closest("button[data-kind]");
    if (!b) return;
    kind = b.getAttribute("data-kind");
    limit = STEP;
    render();
    setUrl();
  });
  statusEl.addEventListener("click", function (e) {
    var b = e.target.closest("button[data-q]");
    if (!b) return;
    input.value = b.getAttribute("data-q");
    limit = STEP;
    render();
    setUrl();
    input.focus();
  });
  moreEl.addEventListener("click", function () { limit += STEP; render(); });

  var given = new URLSearchParams(location.search);
  input.value = given.get("q") || "";
  if (given.get("k")) kind = given.get("k");
  render();

  fetch("assets/search-index.json")
    .then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); })
    .then(function (data) {
      items = data.items;
      items.forEach(function (it) {
        it.nt = norm(it.t); it.nx = norm(it.x); it.nc = norm(it.c); it.nl = norm(it.l);
      });
      render();
    })
    .catch(function () {
      statusEl.textContent = "The search could not load its index. The cheatsheet and the revision page work without it.";
    });
})();
