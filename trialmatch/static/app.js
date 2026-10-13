"use strict";

// All text from the note, the trial and the model is inserted with textContent,
// never innerHTML.

const $ = (id) => document.getElementById(id);
const state = { run: null, reviews: {}, selected: null, filter: "all", sort: "attention", editing: null, showAll: {} };

const VERDICT_OPTIONS = {
  inclusion: [["met", "Met"], ["not_met", "Not met (blocks enrollment)"], ["unknown", "Insufficient information"]],
  exclusion: [["met", "Applies (blocks enrollment)"], ["not_met", "Does not apply"], ["unknown", "Insufficient information"]],
};
const OVERALL_LABELS = {
  possibly_eligible: "Possibly eligible",
  not_eligible: "Likely not eligible",
  needs_more_info: "More information needed",
};
const FILTERS = [
  ["all", "All"], ["blocks", "Blocks"], ["unresolved", "Unresolved"], ["supports", "Supports"], ["low", "Needs attention"],
];

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value);
  }
  for (const child of children) if (child) node.append(child);
  return node;
}

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = response.statusText;
    try { detail = (await response.json()).detail || detail; } catch (_) { /* not JSON */ }
    throw new Error(detail);
  }
  return response.json();
}

function setStatus(message, isError = false) {
  const node = $("status");
  node.hidden = !message;
  node.textContent = message || "";
  node.classList.toggle("error", isError);
}

// ---- verdict logic (mirrors trialmatch.models.effect_of and summarize) ----

function effectOf(type, verdict) {
  if (verdict === "unknown") return "unresolved";
  if (type === "inclusion") return verdict === "met" ? "supports" : "blocks";
  return verdict === "met" ? "blocks" : "supports";
}

function verdictLabel(type, verdict) {
  if (verdict === "unknown") return "? Insufficient information";
  if (type === "inclusion") return verdict === "met" ? "✓ Met" : "✕ Not met — blocks enrollment";
  return verdict === "met" ? "✕ Exclusion applies — blocks enrollment" : "✓ Exclusion does not apply";
}

// The clinician's saved verdict, when there is one, replaces the model's.
function current(c) {
  const review = state.reviews[c.criterion_id];
  const verdict = review ? review.reviewer_verdict : c.verdict;
  return { verdict, effect: effectOf(c.type, verdict), review };
}

function needsAttention(c) {
  return !state.reviews[c.criterion_id] && (c.confidence === "low" || c.downgraded || c.flags.length > 0);
}

function overallOf(effects) {
  if (effects.includes("blocks")) return "not_eligible";
  if (effects.includes("unresolved")) return "needs_more_info";
  return "possibly_eligible";
}

// ---- note pane ----

function renderNote() {
  const run = state.run;
  const note = $("note");
  const hint = $("note-hint");
  // Offsets from the backend count Unicode code points, not UTF-16 units.
  const chars = Array.from(run.patient.text);
  const selected = run.criteria.find((c) => c.criterion_id === state.selected);
  const evidence = selected ? selected.evidence : [];
  const effect = selected ? current(selected).effect : null;

  note.replaceChildren();
  let cursor = 0;
  const marks = [];
  for (const item of evidence) {
    if (item.start < cursor) continue; // overlapping quotes: keep the first
    note.append(chars.slice(cursor, item.start).join(""));
    const mark = el("mark", { "data-effect": effect, text: chars.slice(item.start, item.end).join("") });
    marks.push(mark);
    note.append(mark);
    cursor = item.end;
  }
  note.append(chars.slice(cursor).join(""));

  hint.hidden = !selected || evidence.length > 0;
  if (selected && evidence.length === 0) {
    hint.textContent = `${selected.criterion_id}: no passage in the note was cited for this criterion.`;
  }
  return marks;
}

function focusMark(marks, index = 0) {
  marks.forEach((m) => m.classList.remove("focus"));
  if (!marks[index]) return;
  marks[index].classList.add("focus");
  marks[index].scrollIntoView({ block: "center", behavior: "smooth" });
}

function selectCriterion(id, evidenceIndex = 0, scrollCard = false) {
  state.selected = id;
  state.editing = state.editing === id ? id : null;
  renderCriteria();
  focusMark(renderNote(), evidenceIndex);
  if (scrollCard) {
    const card = document.querySelector(`.card[data-id="${CSS.escape(id)}"]`);
    if (card) card.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }
}

// ---- summary ----

function linkList(target, items, emptyText) {
  const limit = state.showAll[target.id] ? 0 : 3;
  target.replaceChildren();
  if (!items.length) {
    target.append(el("li", { class: "none", text: emptyText }));
    return;
  }
  const shown = limit ? items.slice(0, limit) : items;
  for (const item of shown) {
    target.append(el("li", {},
      el("button", { type: "button", onclick: () => jumpTo(item.criterion_id) },
        el("span", { class: "cid", text: item.criterion_id }), document.createTextNode(item.text))));
  }
  if (limit && items.length > limit) {
    target.append(el("li", {}, el("button", {
      type: "button", class: "show-all", text: `Show all ${items.length}`,
      onclick: () => { state.showAll[target.id] = true; renderSummary(); },
    })));
  }
}

function jumpTo(id) {
  // Make sure the card is visible even if a filter is hiding it.
  const visible = visibleCriteria().some((c) => c.criterion_id === id);
  if (!visible) state.filter = "all";
  renderFilters();
  selectCriterion(id, 0, true);
}

function renderSummary() {
  const { run } = state;
  const s = run.summary;

  const context = $("summary-context");
  context.replaceChildren(
    document.createTextNode(`${run.patient.id}  ·  `),
    el("a", { href: run.trial.url, target: "_blank", rel: "noopener", text: run.trial.nct_id }),
    document.createTextNode(`  ${run.trial.title}`));

  const overall = $("overall");
  overall.textContent = s.overall_label;
  overall.dataset.overall = s.overall;

  const effects = run.criteria.map((c) => current(c).effect);
  const reviewedOverall = overallOf(effects);
  const reviewed = Object.keys(state.reviews).length;
  const reviewedNode = $("reviewed-overall");
  reviewedNode.hidden = reviewed === 0;
  reviewedNode.textContent = reviewedOverall === s.overall
    ? `${reviewed} of ${run.criteria.length} criteria reviewed`
    : `After clinician review (${reviewed} of ${run.criteria.length} criteria): ${OVERALL_LABELS[reviewedOverall]}`;

  const count = (effect) => effects.filter((e) => e === effect).length;
  const stats = run.citation_stats;
  const tiles = [
    ["supports", count("supports"), "support enrollment"],
    ["blocks", count("blocks"), "block enrollment"],
    ["unresolved", count("unresolved"), "unresolved"],
    ["low", run.criteria.filter(needsAttention).length, "need attention"],
    ["", `${stats.quotes_verified}/${stats.quotes_total}`, "quotes verified"],
  ];
  $("counts").replaceChildren(...tiles.map(([cls, value, label]) =>
    el("div", { class: `count ${cls}` }, el("b", { text: String(value) }), el("span", { text: label }))));

  $("disclaimer").textContent = s.disclaimer;
  linkList($("key-reasons"), s.key_reasons, "No determinations support or block enrollment yet.");
  $("follow-up-heading").textContent = `To check (${s.follow_up.length})`;
  linkList($("follow-up"), s.follow_up, "Nothing outstanding.");
}

// ---- criteria pane ----

function visibleCriteria() {
  const list = state.run.criteria.filter((c) => {
    if (state.filter === "all") return true;
    if (state.filter === "low") return needsAttention(c);
    return current(c).effect === state.filter;
  });
  if (state.sort === "trial") return list;
  const rank = (c) => (needsAttention(c) ? 0 : { blocks: 1, unresolved: 2, supports: 3 }[current(c).effect]);
  return list.map((c, i) => [c, i]).sort((a, b) => rank(a[0]) - rank(b[0]) || a[1] - b[1]).map(([c]) => c);
}

function renderFilters() {
  const criteria = state.run.criteria;
  const countFor = (key) => {
    if (key === "all") return criteria.length;
    if (key === "low") return criteria.filter(needsAttention).length;
    return criteria.filter((c) => current(c).effect === key).length;
  };
  $("filters").replaceChildren(...FILTERS.map(([key, label]) => el("button", {
    type: "button", "aria-pressed": String(state.filter === key), text: `${label} ${countFor(key)}`,
    onclick: () => { state.filter = key; renderFilters(); renderCriteria(); },
  })));
}

function detailRow(dl, term, value) {
  if (value === null || value === undefined || value === "") return;
  dl.append(el("dt", { text: term }), value instanceof Node ? el("dd", {}, value) : el("dd", { text: value }));
}

function renderDetails(c) {
  const box = el("div", { class: "details" });
  if (c.downgraded) {
    box.append(el("p", { class: "warning", text:
      `The model answered "${c.model_verdict.replace("_", " ")}", but its quote could not be found in the note. ` +
      "The determination was discarded and needs a manual check." }));
  } else if (c.flags.includes("quote_not_found_in_note")) {
    box.append(el("p", { class: "warning", text: "One of the model's quotes could not be found in the note and was removed." }));
  } else if (c.flags.length) {
    box.append(el("p", { class: "warning", text: c.rationale }));
  }
  if (!c.source_verified) {
    box.append(el("p", { class: "warning", text: "The original wording below could not be matched exactly against the registry text." }));
  }

  const dl = el("dl");
  if (c.evidence.length) {
    const quotes = el("div");
    c.evidence.forEach((e, i) => quotes.append(el("button", {
      type: "button", class: "quote", text: e.quote, title: `Characters ${e.start}–${e.end}`,
      onclick: (event) => { event.stopPropagation(); focusMark(renderNote(), i); },
    })));
    detailRow(dl, "Evidence", quotes);
  }
  if (c.rejected_quotes.length) {
    const rejected = el("div");
    c.rejected_quotes.forEach((q) => rejected.append(el("div", { class: "rejected", text: q })));
    detailRow(dl, "Not in note", rejected);
  }
  if (c.threshold) {
    const t = c.threshold;
    detailRow(dl, "Threshold", `${t.parameter} ${t.comparator} ${t.value}${t.unit ? " " + t.unit : ""}`);
  }
  detailRow(dl, "Patient value", c.patient_value);
  detailRow(dl, "Time window", c.time_window);
  if (!c.downgraded) detailRow(dl, "To check", c.missing_info);
  if (c.source_text && c.source_text !== c.criterion_text) {
    detailRow(dl, c.origin === "structured_field" ? "Registry field" : "Original wording", c.source_text);
  }
  const concepts = c.concepts.map((k) => (k.normalized && k.normalized !== k.original ? `${k.original} → ${k.normalized}` : k.original));
  detailRow(dl, "Concepts", concepts.join("; "));
  box.append(dl);
  return box;
}

async function saveReview(c, action, verdict, comment) {
  try {
    const review = await api(`/api/runs/${encodeURIComponent(state.run.run_id)}/reviews`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        criterion_id: c.criterion_id, action, reviewer_verdict: verdict, comment: comment || "", reviewer: $("reviewer-input").value,
      }),
    });
    state.reviews[c.criterion_id] = review;
    state.editing = null;
    setStatus("");
    renderAll();
  } catch (error) {
    setStatus(`Could not save the review: ${error.message}`, true);
  }
}

function renderReview(c) {
  const { review, verdict } = current(c);
  const row = el("div", { class: "review" });
  row.append(
    el("button", { type: "button", text: "Agree", "aria-pressed": String(review?.action === "agree"),
      onclick: () => saveReview(c, "agree", null, "") }),
    el("button", { type: "button", text: "Modify", "aria-pressed": String(review?.action === "modify"),
      onclick: () => { state.editing = state.editing === c.criterion_id ? null : c.criterion_id; renderCriteria(); } }));

  if (review) {
    const who = review.reviewer ? ` by ${review.reviewer}` : "";
    const what = review.action === "agree" ? "Agreed" : `Changed from "${review.model_verdict.replace("_", " ")}"`;
    row.append(el("span", { class: "state", text: `${what}${who}${review.comment ? ` — ${review.comment}` : ""}` }));
  }

  if (state.editing === c.criterion_id) {
    const select = el("select", { "aria-label": "Corrected verdict" });
    for (const [value, label] of VERDICT_OPTIONS[c.type]) {
      select.append(el("option", { value, text: label, selected: value === verdict ? "" : null }));
    }
    const comment = el("input", { placeholder: "Reason for the change (optional)", "aria-label": "Comment", value: review?.comment || null });
    row.append(el("form", { class: "modify-form", onsubmit: (event) => { event.preventDefault(); saveReview(c, "modify", select.value, comment.value); } },
      select, comment, el("button", { type: "submit", class: "primary", text: "Save" })));
  }
  return row;
}

function renderCard(c) {
  const { verdict, effect, review } = current(c);
  const selected = c.criterion_id === state.selected;
  const top = el("div", { class: "card-top" },
    el("span", { class: "cid", text: c.criterion_id }),
    el("span", { class: `chip ${c.type}`, text: c.type === "inclusion" ? "Inclusion" : "Exclusion" }),
    el("span", { class: "verdict", "data-effect": effect, text: verdictLabel(c.type, verdict) }),
    c.downgraded ? el("span", { class: "badge warn", text: "Evidence not verified" }) : null,
    c.confidence === "low"
      ? el("span", { class: "badge low", text: "Low confidence — review first" })
      : el("span", { class: "badge conf", text: `${c.confidence} confidence` }),
    review ? el("span", { class: "badge reviewed", text: review.action === "agree" ? "Reviewed" : "Modified by clinician" }) : null);

  const main = el("button", { type: "button", class: "card-main", "aria-expanded": String(selected),
    onclick: () => selectCriterion(c.criterion_id) },
    top, el("div", { class: "card-text", text: c.criterion_text }), el("div", { class: "card-rationale", text: c.rationale }));

  return el("li", { class: `card${selected ? " selected" : ""}`, "data-effect": effect, "data-id": c.criterion_id },
    main, selected ? renderDetails(c) : null, renderReview(c));
}

function renderCriteria() {
  const list = visibleCriteria();
  $("criteria").replaceChildren(...list.map(renderCard));
  if (!list.length) $("criteria").append(el("li", { class: "muted", text: "No criteria match this filter." }));
}

function renderAll() {
  renderSummary();
  renderFilters();
  renderCriteria();
  renderNote();
}

// ---- loading ----

async function loadRun(runId) {
  setStatus("");
  try {
    const data = await api(`/api/runs/${encodeURIComponent(runId)}`);
    Object.assign(state, { run: data.run, reviews: data.reviews, selected: null, filter: "all", editing: null, showAll: {} });
  } catch (error) {
    setStatus(`Could not load ${runId}: ${error.message}`, true);
    return;
  }
  $("empty").hidden = true;
  $("summary").hidden = false;
  $("workspace").hidden = false;
  $("note-source").textContent = state.run.patient.source + (state.run.patient.note_date ? `  ·  note dated ${state.run.patient.note_date}` : "");
  $("run-select").value = runId;
  history.replaceState(null, "", `?run=${encodeURIComponent(runId)}`);
  renderAll();
  $("note").scrollTop = 0;
}

async function refreshRuns() {
  const runs = await api("/api/runs");
  const select = $("run-select");
  select.replaceChildren(el("option", { value: "", text: runs.length ? "Select a run" : "No saved runs yet" }));
  for (const run of runs) {
    select.append(el("option", { value: run.run_id, text: `${run.patient_id} × ${run.nct_id} — ${run.overall_label}` }));
  }
  return runs;
}

async function init() {
  $("reviewer-input").value = localStorage.getItem("reviewer") || "";
  $("reviewer-input").addEventListener("change", (e) => localStorage.setItem("reviewer", e.target.value));
  $("run-select").addEventListener("change", (e) => e.target.value && loadRun(e.target.value));
  $("sort-select").addEventListener("change", (e) => { state.sort = e.target.value; renderCriteria(); });

  $("new-run").addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = $("run-button");
    button.disabled = true;
    setStatus("Running the match: fetching the trial, splitting its criteria and judging each one. This usually takes one to three minutes.");
    try {
      const { run_id } = await api("/api/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ patient_id: $("patient-select").value, nct_id: $("nct-input").value }),
      });
      await refreshRuns();
      await loadRun(run_id);
    } catch (error) {
      setStatus(`The match did not run: ${error.message}`, true);
    } finally {
      button.disabled = false;
    }
  });

  try {
    const [patients, runs] = await Promise.all([api("/api/patients"), refreshRuns()]);
    $("patient-select").replaceChildren(...patients.map((p) => el("option", { value: p.id, text: p.label })));
    // Deep links: ?run=<id> opens a run, &criterion=<id> also selects one criterion.
    const params = new URLSearchParams(location.search);
    const criterion = params.get("criterion");
    if (params.get("run")) await loadRun(params.get("run"));
    else if (runs.length) await loadRun(runs[0].run_id);
    if (criterion && state.run?.criteria.some((c) => c.criterion_id === criterion)) jumpTo(criterion);
  } catch (error) {
    setStatus(`Could not reach the server: ${error.message}`, true);
  }
}

init();
