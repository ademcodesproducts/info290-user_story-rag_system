// Quorum demo — single-page client.
// Posts queries to /api/query and renders the structured response.

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const form = $("#query-form");
const input = $("#query-input");
const submitBtn = $("#submit-btn");
const result = $("#result");
const errorPanel = $("#error-panel");
const errorText = $("#error-text");

const summaryEl = $("#summary-text");
const painsGrid = $("#pains-grid");
const storiesList = $("#stories-list");
const sourcesList = $("#sources-list");
const painsCount = $("#pains-count");
const storiesCount = $("#stories-count");
const sourcesCount = $("#sources-count");

const metaVariant = $("#meta-variant");
const metaTopK = $("#meta-topk");
const metaTime = $("#meta-time");

const examplesRow = $("#examples-row");
const kbStat = $("#kb-stat");

// ── Bootstrap ─────────────────────────────────────────────
async function bootstrap() {
  try {
    const [healthRes, examplesRes] = await Promise.all([
      fetch("/api/health").then((r) => r.json()),
      fetch("/api/examples").then((r) => r.json()),
    ]);
    renderExamples(examplesRes.examples || []);
    renderKbStat(healthRes);
  } catch (err) {
    console.warn("bootstrap failed:", err);
  }
}

function renderExamples(examples) {
  examplesRow.innerHTML = "";
  examples.forEach((ex) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "example-chip";
    chip.textContent = ex.title;
    chip.title = ex.query;
    chip.addEventListener("click", () => {
      input.value = ex.query;
      input.focus();
    });
    examplesRow.appendChild(chip);
  });
}

function renderKbStat(health) {
  if (!health.ok) {
    kbStat.textContent = "API key missing";
    return;
  }
  const total = health.stats?.total_chunks;
  if (total != null) {
    kbStat.textContent = `${total} chunks indexed`;
  } else {
    kbStat.textContent = "ready";
  }
}

// ── Submit ────────────────────────────────────────────────
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = input.value.trim();
  if (!query) return;

  const variant = $('input[name="variant"]:checked').value;
  const topK = parseInt($('input[name="topk"]:checked').value, 10);
  const filterRaw = $('input[name="filter"]:checked').value;
  const filter = filterRaw === "" ? null : filterRaw;

  setLoading(true);
  hideError();

  const startedAt = performance.now();

  try {
    const res = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        variant,
        top_k: topK,
        filter_type: filter,
      }),
    });

    if (!res.ok) {
      const detail = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(detail.detail || `Request failed (${res.status})`);
    }

    const data = await res.json();
    const elapsedMs = Math.round(performance.now() - startedAt);
    renderResult(data, elapsedMs);
  } catch (err) {
    showError(err.message || String(err));
  } finally {
    setLoading(false);
  }
});

function setLoading(loading) {
  submitBtn.classList.toggle("loading", loading);
  submitBtn.disabled = loading;
  $(".btn-label").textContent = loading ? "Thinking…" : "Ask Quorum";
}

function showError(msg) {
  errorText.textContent = msg;
  errorPanel.hidden = false;
  result.hidden = true;
}

function hideError() {
  errorPanel.hidden = true;
}

// ── Render ────────────────────────────────────────────────
function renderResult(data, elapsedMs) {
  result.hidden = false;

  // Meta chips
  metaVariant.textContent = data.variant_label || data.model;
  metaTopK.textContent = `top-k = ${data.top_k}`;
  metaTime.textContent = `${(elapsedMs / 1000).toFixed(1)}s`;

  // Summary
  summaryEl.textContent = data.summary || "(no summary returned)";

  // Pain points
  painsGrid.innerHTML = "";
  const pains = data.pain_points || [];
  painsCount.textContent = `${pains.length} found`;
  pains.forEach((p) => painsGrid.appendChild(painCard(p)));

  // User stories
  storiesList.innerHTML = "";
  const stories = data.user_stories || [];
  storiesCount.textContent = `${stories.length} found`;
  stories.forEach((s) => storiesList.appendChild(storyCard(s)));

  // Sources
  sourcesList.innerHTML = "";
  const sources = data.sources || [];
  sourcesCount.textContent = `${sources.length} chunks`;
  sources.forEach((src, i) => sourcesList.appendChild(sourceCard(src, i + 1)));

  result.scrollIntoView({ behavior: "smooth", block: "start" });
}

function painCard(p) {
  const sev = (p.severity || "").toLowerCase();
  const div = document.createElement("div");
  div.className = `pain ${sev}`;

  const row = document.createElement("div");
  row.className = "pain-row";
  const sevSpan = document.createElement("span");
  sevSpan.className = `severity ${sev}`;
  sevSpan.textContent = sev || "—";
  row.appendChild(sevSpan);
  div.appendChild(row);

  const text = document.createElement("div");
  text.className = "pain-text";
  text.textContent = p.description || "";
  div.appendChild(text);

  if (p.sources?.length) div.appendChild(sourceChipRow(p.sources));
  return div;
}

function storyCard(s) {
  const div = document.createElement("div");
  div.className = "story";

  const story = document.createElement("div");
  story.className = "story-text";
  story.textContent = s.story || "";
  div.appendChild(story);

  if (s.rationale) {
    const rat = document.createElement("div");
    rat.className = "story-rationale";
    const lbl = document.createElement("strong");
    lbl.textContent = "Rationale";
    rat.appendChild(lbl);
    rat.appendChild(document.createTextNode(s.rationale));
    div.appendChild(rat);
  }

  if (s.sources?.length) div.appendChild(sourceChipRow(s.sources));
  return div;
}

function sourceChipRow(sources) {
  const row = document.createElement("div");
  row.className = "source-chips";
  sources.forEach((s) => {
    const chip = document.createElement("span");
    chip.className = "source-chip";
    chip.textContent = s;
    row.appendChild(chip);
  });
  return row;
}

function sourceCard(src, idx) {
  const div = document.createElement("div");
  div.className = "source";

  const meta = document.createElement("div");
  meta.className = "source-meta";

  const i = document.createElement("span");
  i.className = "source-idx";
  i.textContent = `[${idx}]`;
  meta.appendChild(i);

  if (src.doc_type) {
    const dt = document.createElement("span");
    dt.className = "source-doctype";
    dt.textContent = src.doc_type;
    meta.appendChild(dt);
  }

  const lbl = document.createElement("span");
  lbl.className = "source-label";
  lbl.textContent = src.label || src.source_file || "";
  meta.appendChild(lbl);

  if (src.distance != null) {
    const d = document.createElement("span");
    d.className = "source-distance";
    d.textContent = `dist ${src.distance.toFixed(3)}`;
    meta.appendChild(d);
  }

  div.appendChild(meta);

  const snip = document.createElement("div");
  snip.className = "source-snippet";
  snip.textContent = src.snippet || "";
  div.appendChild(snip);

  return div;
}

// Cmd/Ctrl + Enter to submit
input.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
    e.preventDefault();
    form.requestSubmit();
  }
});

// ── Tabs (Live Demo / Benchmarks) ─────────────────────────
const pageDemo = $("#page-demo");
const pageBench = $("#page-benchmarks");
let benchmarksLoaded = false;

$$(".tab").forEach((tab) => {
  tab.addEventListener("click", async () => {
    $$(".tab").forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    const target = tab.dataset.tab;
    if (target === "demo") {
      pageDemo.hidden = false;
      pageBench.hidden = true;
    } else {
      pageDemo.hidden = true;
      pageBench.hidden = false;
      if (!benchmarksLoaded) {
        await loadBenchmarks();
        benchmarksLoaded = true;
      }
    }
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
});

// ── Benchmarks ────────────────────────────────────────────
async function loadBenchmarks() {
  try {
    const [bench, casesRes] = await Promise.all([
      fetch("/api/benchmarks").then((r) => r.json()),
      fetch("/api/benchmarks/cases?run=test_final").then((r) => r.json()),
    ]);
    renderBenchHeadline(bench.headline);
    renderPromptCompare(bench.prompt_compare);
    renderTopkCompare(bench.topk_compare);
    renderCases(casesRes.cases || []);
  } catch (err) {
    console.warn("benchmarks load failed:", err);
  }
}

function renderBenchHeadline(headline) {
  const tilesEl = $("#bench-stat-tiles");
  const cfgEl = $("#bench-headline-config");
  if (!headline) {
    tilesEl.innerHTML = '<div class="case-empty">No final-test results found in results/test_final_topk10.json.</div>';
    return;
  }

  cfgEl.innerHTML = "";
  const cfgChips = [
    `n = ${headline.n_cases ?? "?"}`,
    `model = ${headline.config.model}`,
    `top-k = ${headline.config.top_k}`,
    `eval = ${headline.config.eval_set?.split("/").pop() ?? "?"}`,
  ];
  cfgChips.forEach((c) => {
    const chip = document.createElement("span");
    chip.className = "meta-chip";
    chip.textContent = c;
    cfgEl.appendChild(chip);
  });

  const r = headline.retrieval || {};
  const g = headline.generation || {};
  const j = headline.llm_judge || {};

  const tiles = [
    { value: pct(r.recall_at_k), label: "Recall @ k", sub: "fraction of expected sources retrieved" },
    { value: r.mrr?.toFixed(3), label: "MRR", sub: "mean reciprocal rank of first hit" },
    { value: j.faithfulness?.toFixed(2), label: "Faithfulness", sub: "RAGAS · 0–5", cls: "judge" },
    { value: j.relevance?.toFixed(2), label: "Relevance", sub: "RAGAS · 0–5", cls: "judge" },
    { value: j.invest_mean_total?.toFixed(2), label: "INVEST", sub: "user-story quality · 0–18", cls: "judge" },
    { value: pct(g.story_overall), label: "Story-overall", sub: "format · feature · benefit", cls: "results" },
  ];

  tilesEl.innerHTML = "";
  tiles.forEach((t) => {
    const div = document.createElement("div");
    div.className = `stat-tile ${t.cls || ""}`;
    div.innerHTML = `
      <div class="stat-value">${t.value ?? "—"}</div>
      <div class="stat-label">${t.label}</div>
      <div class="stat-sub">${t.sub}</div>
    `;
    tilesEl.appendChild(div);
  });
}

function renderPromptCompare(runs) {
  const target = $("#prompt-compare");
  target.innerHTML = "";
  if (!runs || !runs[0] || !runs[1]) {
    target.innerHTML = '<div class="case-empty">No V1/V2 validation runs found.</div>';
    return;
  }
  const [v1, v2] = runs;
  const metrics = [
    { key: "story_overall",         path: "generation.story_overall",         label: "Story Overall",        sub: "format + feature + benefit" },
    { key: "story_format_compliance", path: "generation.story_format_compliance", label: "Format Compliance", sub: "Connextra structure" },
    { key: "story_named_feature",   path: "generation.story_named_feature",   label: "Named Feature",        sub: "names a real Databricks product" },
    { key: "story_benefit_detail",  path: "generation.story_benefit_detail",  label: "Benefit Detail",       sub: "measurable so-that clause" },
    { key: "faithfulness",          path: "llm_judge.faithfulness",           label: "Faithfulness",         sub: "RAGAS judge · 0–5", scale: 5 },
    { key: "relevance",             path: "llm_judge.relevance",              label: "Relevance",            sub: "RAGAS judge · 0–5", scale: 5 },
    { key: "invest_mean_total",     path: "llm_judge.invest_mean_total",      label: "INVEST",               sub: "story quality · 0–18", scale: 18 },
  ];

  metrics.forEach((m) => {
    const v1val = pluck(v1, m.path);
    const v2val = pluck(v2, m.path);
    target.appendChild(metricGroupLabel(`${m.label} · ${m.sub}`));
    target.appendChild(barRow("V1 baseline", v1val, m.scale, false, refValue(v2val, v1val)));
    target.appendChild(barRow("V2 enhanced ★", v2val, m.scale, true, deltaValue(v1val, v2val)));
  });
}

function renderTopkCompare(runs) {
  const target = $("#topk-compare");
  target.innerHTML = "";
  if (!runs || runs.length === 0) {
    target.innerHTML = '<div class="case-empty">No top-k ablation runs found.</div>';
    return;
  }

  const metrics = [
    { path: "retrieval.recall_at_k", label: "Recall @ k", sub: "fraction of expected sources retrieved" },
    { path: "retrieval.mrr",         label: "MRR",        sub: "mean reciprocal rank · 0–1" },
    { path: "generation.story_overall", label: "Story Overall", sub: "stays stable across k" },
  ];

  metrics.forEach((m) => {
    target.appendChild(metricGroupLabel(`${m.label} · ${m.sub}`));
    runs.forEach((r, i) => {
      if (!r) return;
      const v = pluck(r, m.path);
      const k = r.config?.top_k ?? "?";
      const cls = i === 0 ? "k3" : i === 1 ? "k5" : "k10";
      target.appendChild(barRow(`k = ${k}`, v, null, false, formatValue(v), cls));
    });
  });
}

function metricGroupLabel(text) {
  const div = document.createElement("div");
  div.className = "compare-group-label";
  div.textContent = text;
  return div;
}

function barRow(label, value, scale, winner, valueLabel, cls) {
  const wrap = document.createElement("div");
  wrap.className = "compare-row";

  const lbl = document.createElement("div");
  lbl.className = "compare-row-label";
  lbl.textContent = label;
  wrap.appendChild(lbl);

  const bg = document.createElement("div");
  bg.className = "compare-bar-bg";
  const fill = document.createElement("div");
  fill.className = `compare-bar-fill ${cls || (winner ? "winner" : "")}`;
  const pctNum = value == null ? 0 : (scale ? (value / scale) * 100 : value * 100);
  fill.style.width = Math.min(100, Math.max(2, pctNum)) + "%";
  fill.textContent = formatValue(value, scale);
  bg.appendChild(fill);
  wrap.appendChild(bg);

  const val = document.createElement("div");
  val.className = "compare-row-value" + (typeof valueLabel === "object" ? ` delta${valueLabel.neg ? " neg" : ""}` : "");
  val.textContent = typeof valueLabel === "object" ? valueLabel.text : (valueLabel || "");
  wrap.appendChild(val);

  return wrap;
}

function deltaValue(baseline, candidate) {
  if (baseline == null || candidate == null) return "";
  const delta = candidate - baseline;
  const sign = delta >= 0 ? "+" : "";
  const text = Math.abs(delta) < 0.01 ? "≈" : `${sign}${delta.toFixed(3)}`;
  return { text, neg: delta < 0 };
}

function refValue(_a, _b) { return ""; }

function formatValue(v, scale) {
  if (v == null) return "—";
  if (scale && scale > 1) return v.toFixed(2);
  if (v <= 1) return (v * 100).toFixed(1) + "%";
  return v.toFixed(2);
}

function pct(v) {
  if (v == null) return "—";
  return (v * 100).toFixed(1) + "%";
}

function pluck(obj, path) {
  return path.split(".").reduce((o, k) => (o ? o[k] : undefined), obj);
}

// ── Cases ─────────────────────────────────────────────────
let allCases = [];

function renderCases(cases) {
  allCases = cases;
  filterCases();
}

function filterCases() {
  const q = $("#cases-search").value.trim().toLowerCase();
  const filterMode = ($('input[name="case-filter"]:checked') || {}).value || "all";
  const list = $("#cases-list");
  const empty = $("#cases-empty");
  list.innerHTML = "";
  let shown = 0;

  allCases.forEach((c) => {
    if (filterMode === "hit" && c.recall_at_k < 1) return;
    if (filterMode === "miss" && c.recall_at_k >= 1) return;
    if (q) {
      const haystack = `${c.id || ""} ${c.query || ""} ${(c.sources || []).map((s) => s.label || "").join(" ")}`.toLowerCase();
      if (!haystack.includes(q)) return;
    }
    list.appendChild(caseEl(c));
    shown++;
  });

  $("#cases-count").textContent = `${shown} / ${allCases.length} cases`;
  empty.hidden = shown !== 0;
}

function caseEl(c) {
  const det = document.createElement("details");
  det.className = "case";

  const sum = document.createElement("summary");

  const id = document.createElement("span");
  id.className = "case-id";
  id.textContent = c.id || "?";

  const q = document.createElement("span");
  q.className = "case-query";
  q.textContent = c.query || "";

  const badges = document.createElement("span");
  badges.className = "case-badges";

  const recallCls = c.recall_at_k >= 1 ? "pass" : c.recall_at_k > 0 ? "warn" : "fail";
  badges.appendChild(badge(`recall ${formatValue(c.recall_at_k)}`, recallCls));

  if (c.faithfulness != null) {
    const fcls = c.faithfulness >= 4.5 ? "pass" : c.faithfulness >= 3.5 ? "warn" : "fail";
    badges.appendChild(badge(`faith ${c.faithfulness}/5`, fcls));
  }
  if (c.invest_total != null) {
    const icls = c.invest_total >= 16 ? "pass" : c.invest_total >= 13 ? "warn" : "fail";
    badges.appendChild(badge(`INVEST ${c.invest_total}/18`, icls));
  }

  sum.append(id, q, badges);
  det.appendChild(sum);

  const body = document.createElement("div");
  body.className = "case-body";

  if (c.summary) {
    body.appendChild(sectionLabel("Summary"));
    const s = document.createElement("div");
    s.className = "case-summary-text";
    s.textContent = c.summary;
    body.appendChild(s);
  }

  if (c.user_stories?.length) {
    body.appendChild(sectionLabel(`User Stories (${c.user_stories.length})`));
    const wrap = document.createElement("div");
    wrap.className = "case-stories";
    c.user_stories.forEach((us) => {
      const s = document.createElement("div");
      s.className = "case-story";
      s.textContent = us.story || "";
      wrap.appendChild(s);
    });
    body.appendChild(wrap);
  }

  if (c.pain_points?.length) {
    body.appendChild(sectionLabel(`Pain Points (${c.pain_points.length})`));
    const wrap = document.createElement("div");
    wrap.className = "case-pains";
    c.pain_points.forEach((p) => {
      const div = document.createElement("div");
      div.className = `case-pain ${(p.severity || "").toLowerCase()}`;
      div.textContent = p.description || "";
      wrap.appendChild(div);
    });
    body.appendChild(wrap);
  }

  // LLM judge breakdown
  if (c.faithfulness != null || c.relevance != null || c.invest_total != null) {
    body.appendChild(sectionLabel("LLM Judge"));
    const judge = document.createElement("div");
    judge.className = "case-judge";
    if (c.faithfulness != null) judge.appendChild(judgeItem("Faithfulness", `${c.faithfulness}/5`));
    if (c.relevance != null) judge.appendChild(judgeItem("Relevance", `${c.relevance}/5`));
    if (c.invest_total != null) judge.appendChild(judgeItem("INVEST", `${c.invest_total}/18`));
    if (c.story_overall != null) judge.appendChild(judgeItem("Story-overall", formatValue(c.story_overall)));
    if (c.pain_keyword_recall != null) judge.appendChild(judgeItem("Pain-kw recall", formatValue(c.pain_keyword_recall)));
    body.appendChild(judge);
  }

  if (c.sources?.length) {
    body.appendChild(sectionLabel(`Retrieved Sources (${c.sources.length})`));
    const wrap = document.createElement("div");
    wrap.className = "case-sources";
    c.sources.forEach((src, i) => {
      const div = document.createElement("div");
      div.className = "case-source";
      const label = src.label || src.source_file || `[${i + 1}]`;
      div.textContent = `[${i + 1}] ${label}`;
      wrap.appendChild(div);
    });
    body.appendChild(wrap);
  }

  det.appendChild(body);
  return det;
}

function badge(text, cls) {
  const span = document.createElement("span");
  span.className = `case-badge ${cls || ""}`;
  span.textContent = text;
  return span;
}

function sectionLabel(text) {
  const div = document.createElement("div");
  div.className = "case-section-label";
  div.textContent = text;
  return div;
}

function judgeItem(label, value) {
  const div = document.createElement("div");
  div.className = "judge-item";
  const strong = document.createElement("strong");
  strong.textContent = label;
  div.appendChild(strong);
  div.appendChild(document.createTextNode(value));
  return div;
}

document.addEventListener("input", (e) => {
  if (e.target.id === "cases-search") filterCases();
});
document.addEventListener("change", (e) => {
  if (e.target.name === "case-filter") filterCases();
});

bootstrap();
