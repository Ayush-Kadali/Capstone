"use strict";
// Rendering and interaction only. Selection, measurements, correctness and
// comparisons are produced by the C++ engine and scripts/study.py.
const $ = (id) => document.getElementById(id);
const state = {
  id: null,
  bundle: null,
  node: null,
  plan: null,
  baseline: null,
  diff: null,
  jobs: [],
  busy: false,
  request: 0,
  modelAvailable: false,
  modelSubmitting: false,
  profiles: [],
  modelDetail: null,
};
const esc = (value) =>
  String(value ?? "unavailable").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const num = (value, digits = 4) =>
  typeof value === "number" && Number.isFinite(value)
    ? value.toLocaleString(undefined, { maximumFractionDigits: digits })
    : "unavailable";
const tag = (label, kind) => `<span class="tag ${kind}">${esc(label)}</span>`;
const byte = (n) => (n == null ? "unavailable" : `${num(n, 0)} B`);
const shape = (x) => (Array.isArray(x) ? x.join(" × ") : "unavailable");
const kv = (rows) =>
  `<dl class="kv">${rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join("")}</dl>`;
function table(headers, rows) {
  return `<table><thead><tr>${headers.map((h) => `<th scope="col">${esc(h)}</th>`).join("")}</tr></thead><tbody>${rows.join("")}</tbody></table>`;
}
function error(message) {
  $("error").hidden = !message;
  $("error").textContent = message || "";
}
async function api(path, body) {
  const response = await fetch(
    path,
    body === undefined
      ? {}
      : {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-Capstone-Request": "1",
          },
          body: JSON.stringify(body),
        },
  );
  const value = await response.json();
  if (!response.ok)
    throw new Error(value.error || `Request failed (${response.status})`);
  return value;
}
function exportLink(id, path, filename) {
  const link = $(id);
  link.setAttribute("aria-disabled", path ? "false" : "true");
  link.tabIndex = path ? 0 : -1;
  link.download = filename || "";
  if (path) link.setAttribute("href", path);
  else link.removeAttribute("href");
}
async function requestExport(event) {
  const source = event.currentTarget;
  if (source.getAttribute("aria-disabled") === "true") {
    event.preventDefault();
    return;
  }
  const path = source.getAttribute("href");
  error("");
  try {
    // Leave the anchor's default download untouched. A parallel HEAD request
    // surfaces read/validation failures in the existing page; GET independently
    // validates the same evidence. No blob, synthetic click or new window.
    const response = await fetch(path, { method: "HEAD" });
    if (!response.ok) {
      await api(path); // GET the server's bounded JSON diagnostic.
      throw new Error(`Export failed (${response.status})`);
    }
    if (!response.headers.get("Content-Disposition")?.startsWith("attachment;"))
      throw new Error("Export response did not contain a saved attachment");
    $("status").textContent =
      "Download requested. The browser controls where the saved file is stored.";
  } catch (failure) {
    error(`Export unavailable: ${failure.message}`);
  }
}
function predictionStore(id, value) {
  try {
    localStorage.setItem(`capstone-prediction-${id}`, JSON.stringify(value));
  } catch {
    /* optional browser storage */
  }
}
function predictionRead(id) {
  try {
    return JSON.parse(localStorage.getItem(`capstone-prediction-${id}`));
  } catch {
    return null;
  }
}
function isBounded(bundle = state.bundle) {
  return bundle?.artifact_type === "bounded-placement-study";
}
function syncPolicyScope() {
  const scope = isBounded() ? "bounded" : "exhaustive";
  if (state.policyScope === scope) return;
  const previous = $("policy").value;
  const choices = [["robust", "Boundary and uncertainty aware"], ["cpu", "CPU only"],
    ["maximal", "Strict maximal supported"], ["additive", "Profile additive"],
    ...(scope === "bounded" ? [["best_observed_candidate", "Best observed candidate"]] :
      [["oracle", "Measured oracle"], ["maximal_feasible", "Maximal feasible node count"]])];
  $("policy").innerHTML = choices.map(([value, label]) => `<option value="${value}">${label}</option>`).join("");
  $("policy").value = choices.some(([value]) => value === previous) ? previous :
    scope === "bounded" && previous === "oracle" ? "best_observed_candidate" :
    scope === "exhaustive" && previous === "best_observed_candidate" ? "oracle" : "robust";
  state.policyScope = scope;
}
function selectedPolicy() {
  return state.bundle?.policies[$("policy").value];
}
function related(command) {
  const node = state.bundle?.graph.nodes.find((n) => n.id === state.node);
  return (
    node &&
    (command.node === node.id ||
      (command.origins || []).some((id) => (node.origins || []).includes(id)) ||
      command.inputs.some((h) =>
        [node.output, ...node.inputs].includes(h.slice(2)),
      ) ||
      [node.output, ...node.inputs].includes(command.output?.slice(2)))
  );
}
function passView(records) {
  return records.length
    ? records
        .map(
          (p) =>
            `<details class="pass"><summary>${esc(p.name)}</summary><p>${(p.reasons || []).map(esc).join("; ")}</p><pre>${esc((p.changes || []).join("\n"))}</pre><p>${(p.consequences || []).map(esc).join("; ")}</p><details><summary>Input / output IR</summary><pre>${esc(p.input_ir)}</pre><pre>${esc(p.output_ir)}</pre></details></details>`,
        )
        .join("")
    : "<p>Unavailable: this artifact has no pass records.</p>";
}
function renderGraph() {
  const b = state.bundle,
    p = state.plan,
    changed = new Set(
      state.diff?.policies?.[$("policy").value]?.changed_node_ids || [],
    );
  $("graph").innerHTML = b.graph.nodes
    .map(
      (n, i) =>
        `<button class="node ${p?.placement[i] === "A" ? "device" : ""} ${state.node === n.id ? "selected" : ""} ${changed.has(n.id) ? "changed" : ""}" data-node="${esc(n.id)}" aria-pressed="${state.node === n.id}"><span class="identifier">${esc(n.id)}<span>${p ? esc(p.placement[i]) : "?"}</span></span><span class="operator">${esc(n.op)}</span><span class="tensor">${esc(n.inputs.join(" + "))} → ${esc(n.output)}<br>${esc(shape(n.shape))} · ${esc(n.dtype || "f32")}</span></button>`,
    )
    .join("");
  $("source-graph").innerHTML = table(
    ["Node", "Operation", "Inputs → output", "Shape"],
    b.source_graph.nodes.map(
      (n) =>
        `<tr><td>${esc(n.id)}</td><td>${esc(n.op)}</td><td>${esc(n.inputs.join(", "))} → ${esc(n.output)}</td><td>${esc(shape(n.shape))}</td></tr>`,
    ),
  );
  $("frontend-passes").innerHTML = passView(b.frontend_passes || []);
}
function renderInspector() {
  const b = state.bundle,
    p = state.plan,
    n = b.graph.nodes.find((x) => x.id === state.node);
  if (!n) {
    $("inspector").textContent = "Select a graph operation.";
    $("buffers").textContent = "Select a graph operation.";
    return;
  }
  const cap = b.capabilities?.nodes.find((c) => c.node_id === n.id);
  const index = b.graph.nodes.indexOf(n);
  $("inspector").innerHTML = kv([
    ["Operation", `<strong>${esc(n.id)} / ${esc(n.op)}</strong>`],
    [
      "Schedule segment",
      esc(
        p?.partitions?.find((part) => part.node_ids.includes(n.id))?.id ??
          "unavailable",
      ),
    ],
    [
      "Selected lane",
      p
        ? p.placement[index] === "A"
          ? "Logical device"
          : "CPU"
        : "Unavailable: no feasible plan for this policy",
    ],
    [
      "Capability",
      cap
        ? `${cap.device_supported ? "Accepted" : "Rejected"}: ${esc(cap.reason)}`
        : "Unavailable: not exported",
    ],
    ["Input tensors", n.inputs.map(esc).join(", ")],
    ["Output", esc(n.output)],
    [
      "Storage",
      `${esc(shape(n.shape))}; ${esc(n.dtype || "f32")}; ${esc(n.layout || "unavailable")}`,
    ],
    ["Source lineage", (n.origins || [n.id]).map(esc).join(", ")],
    ["Original ONNX operation", (state.modelDetail?.import_manifest?.normalized_nodes || [])
      .filter(source => (n.origins || [n.id]).includes(source.generated_node))
      .map(source => esc(`#${source.source_index} ${source.source_name || "unnamed"} → ${source.operator}`)).join("<br>") || "Not an ONNX import"],
    ["Kernel parameters", n.parameters?.length ? esc(JSON.stringify(n.parameters)) : "None"],
  ]);
  const buffers =
    p?.buffers.filter((r) => [n.output, ...n.inputs].includes(r.tensor)) || [];
  $("buffers").innerHTML = buffers.length
    ? table(
        ["Handle / shape", "Bytes / slot", "Lifetime (command IDs)"],
        buffers.map(
          (r) =>
            `<tr><td class="handle">${esc(r.handle)}<br>${esc(shape(r.shape))}</td><td>${byte(r.bytes)}<br>slot ${r.slot == null ? "unavailable" : esc(r.slot)}</td><td>${r.initial_buffer_id ? "Prepared before executor" : r.preexisting_input ? "Preexisting input" : `ALLOC #${esc(r.allocation_id)}`}<br>last use ${r.last_use_id == null ? "unavailable" : "#" + esc(r.last_use_id)}<br>${r.free_id == null ? "Retained through run end" : "FREE #" + esc(r.free_id)}</td></tr>`,
        ),
      )
    : "<p>No buffer records for the selected policy.</p>";
}
function renderMetrics() {
  const b = state.bundle,
    p = state.plan,
    policy = selectedPolicy();
  if (!p) {
    $("metrics").innerHTML = "";
    $("policy-reason").textContent =
      `${policy?.display_name || $("policy").selectedOptions[0].text}: ${policy?.status || "unavailable"}. Requested placement ${policy?.requested_placement ?? policy?.placement ?? "unavailable"}. ${policy?.reason_code || "No valid policy result"}. No substitute plan is displayed.`;
    return;
  }
  const m = p.metrics,
    objective =
      $("policy").value === "additive"
        ? ["Additive objective", p.prediction?.additive_ms]
        : $("policy").value === "robust"
          ? ["Robust objective", p.prediction?.robust_score_ms]
          : ["Boundary estimate", p.prediction?.boundary_ms],
    rows = [
      ["Executor p50", `${num(m.p50_ms)} ms`, "Measured", "measured"],
      ["Executor p95", `${num(m.p95_ms)} ms`, "Measured", "measured"],
      [objective[0], `${num(objective[1])} ms`, "Modeled", "modeled"],
      [
        isBounded(b) ? "Gap to best observed candidate" : "Oracle regret",
        b.conclusions_valid ? `${num(isBounded(b) ? m.regret_vs_best_observed_ms : m.regret_ms)} ms` : "Suppressed",
        b.conclusions_valid ? "Derived" : "Unavailable",
        b.conclusions_valid ? "derived" : "unavailable",
      ],
      [
        "Executor H→D / D→H bytes",
        `${num(m.h2d_bytes, 0)} / ${num(m.d2h_bytes, 0)}`,
        "Exact structural",
        "exact",
      ],
      [
        "Reserved device pool",
        byte(m.peak_allocated_device_bytes),
        "Exact structural",
        "exact",
      ],
      [
        "Device runs / syncs",
        `${num(m.islands, 0)} / ${num(m.synchronizations, 0)}`,
        "Exact structural",
        "exact",
      ],
      [
        "Correctness",
        m.correct ? "Reference match" : "Failed",
        "Measured",
        "measured",
      ],
      [
        "Input setup H→D bytes",
        byte(m.preparation_h2d_bytes),
        m.preparation_h2d_bytes == null ? "Unavailable" : "Exact structural",
        m.preparation_h2d_bytes == null ? "unavailable" : "exact",
      ],
      [
        "Input setup p50",
        m.input_preparation
          ? `${num(m.input_preparation.p50_ms)} ms`
          : "unavailable",
        m.input_preparation ? "Measured" : "Unavailable",
        m.input_preparation ? "measured" : "unavailable",
      ],
      [
        "Setup + executor H→D",
        m.preparation_h2d_bytes == null
          ? "unavailable"
          : byte(m.preparation_h2d_bytes + m.h2d_bytes),
        m.preparation_h2d_bytes == null ? "Unavailable" : "Derived",
        m.preparation_h2d_bytes == null ? "unavailable" : "derived",
      ],
      [
        "Requested host waits",
        m.injected_boundary_ms == null
          ? "unavailable"
          : `${num(m.injected_boundary_ms)} ms`,
        m.injected_boundary_ms == null ? "Unavailable" : "Derived",
        m.injected_boundary_ms == null ? "unavailable" : "derived",
      ],
    ];
  $("metrics").innerHTML = rows
    .map(
      ([label, value, provenance, kind]) =>
        `<div class="metric"><span class="label">${esc(label)}</span><span class="value">${esc(value)}</span>${tag(provenance, kind)}</div>`,
    )
    .join("");
  $("policy-reason").textContent =
    `Placement ${p.placement}. ${policy?.reason_code || "Selected artifact plan"}. Empirical envelope ${num(p.prediction?.lower_ms)}–${num(p.prediction?.upper_ms)} ms; not a confidence interval. Illustrative core estimate ${num(p.illustrative_modeled_ms)} ms is separate from calibration and measurement. ${b.experiment.config.emulate_boundaries ? "Host boundary simulation is enabled: executor observations include injected software waits, not hardware transfer latency." : "Boundary costs are model-only: no host delay was injected."} Input setup time/traffic is outside the executor interval and shown separately.`;
}
function renderCommands() {
  const p = state.plan;
  if (!p) {
    $("commands").textContent =
      "Unavailable: this policy has no executable plan.";
    $("passes").textContent = "No plan selected.";
    return;
  }
  const events = new Map(p.events.map((e) => [e.command_id, e]));
  $("commands").innerHTML = table(
    [
      "Command",
      "Lane / operation",
      "Operands → output",
      "Slot",
      "Host µs",
      "Device µs",
    ],
    p.commands.map((c) => {
      const e = events.get(c.id);
      return `<tr data-command="${c.id}" class="${related(c) ? "linked" : ""}"><td><button data-command="${c.id}" aria-label="Inspect command ${c.id}">#${c.id} ${esc(c.kind)}</button></td><td>${esc(c.backend)}<br>${esc(c.op || "")}</td><td class="handle">${esc(c.inputs.join(", "))} ${c.output ? "→ " + esc(c.output) : ""}</td><td>${c.buffer_slot == null ? "unavailable" : esc(c.buffer_slot)}</td><td>${num(e?.duration_us)}</td><td>${e?.device_us == null || e.device_us < 0 ? "unavailable" : num(e.device_us)}</td></tr>`;
    }),
  );
  $("passes").innerHTML = passView(p.passes || []);
}
function renderTimeline() {
  const p = state.plan;
  exportLink(
    "trace-export",
    p ? `/api/export/${state.id}/trace?plan=${encodeURIComponent(p.id)}` : null,
    p ? `capstone-trace-${state.id}-${p.id}.json` : null,
  );
  if (!p) {
    $("timeline").textContent = "Unavailable: select an available policy.";
    $("hardware").textContent = "External backend overlay unavailable.";
    return;
  }
  const byId = new Map(p.commands.map((c) => [c.id, c]));
  const end = Math.max(1, ...p.events.map((e) => e.start_us + e.duration_us));
  const lanes = [
    "CPU compute",
    "Device compute",
    "Transfers",
    "Alloc / free",
    "Synchronization",
  ];
  const lane = (e) => {
    const c = byId.get(e.command_id);
    return c?.kind === "COMPUTE"
      ? c.backend === "CPU"
        ? 0
        : 1
      : c?.kind === "COPY"
        ? 2
        : c?.kind === "SYNC"
          ? 4
          : 3;
  };
  $("timeline").innerHTML =
    lanes
      .map(
        (label, i) =>
          `<div class="timeline-lane"><span>${esc(label)}</span><div class="timeline-track">${p.events
            .filter((e) => lane(e) === i)
            .map((e) => {
              const c = byId.get(e.command_id);
              const title = `#${e.command_id} ${c?.kind} ${c?.node || c?.output || ""}: host ${num(e.duration_us)} µs`;
              return `<button class="event ${i === 1 ? "device" : i > 1 ? "other" : ""} ${c && related(c) ? "linked" : ""}" data-command="${e.command_id}" aria-label="${esc(title)}" title="${esc(title)}" style="left:${(100 * e.start_us) / end}%;width:${Math.min(100 - (100 * e.start_us) / end, (100 * e.duration_us) / end)}%"></button>`;
            })
            .join("")}</div></div>`,
      )
      .join("") +
    `<div class="timeline-axis"><span>0 µs</span><span>${num(end / 2)} µs</span><span>${num(end)} µs</span></div><p class="hint">Input setup p50: ${p.metrics.input_preparation ? num(p.metrics.input_preparation.p50_ms) + " ms" : "unavailable"} (outside this executor timeline). Queue-wait p50: ${p.metrics.queue_wait ? num(p.metrics.queue_wait.p50_ms) + " ms" : "unavailable"}. End-to-end p50: ${p.metrics.end_to_end ? num(p.metrics.end_to_end.p50_ms) + " ms" : "unavailable"}.</p>`;
  const real = p.events.filter((e) => e.device_us != null && e.device_us >= 0);
  $("hardware").innerHTML = real.length
    ? `${tag("Measured device events", "measured")} ${real.length} exported device event durations from ${esc(p.execution_backend)}. Host timeline remains separate. Diagnostic readback: ${byte(p.metrics.diagnostic_readback_bytes)}. Vendor internals unavailable unless exported.`
    : `${tag("External overlay unavailable", "unavailable")} No measured hardware events in this artifact. Host software timing does not validate BeagleY-AI or accelerator performance.`;
}
function renderProvenance() {
  const b = state.bundle,
    p = state.plan;
  $("provenance").innerHTML =
    kv([
      ["Schema", esc(`${b.schema_version} / ${b.artifact_type}`)],
      [
        "Graph hash",
        `<span class="hash">${esc(b.experiment.graph_sha256)}</span>`,
      ],
      [
        "Binary hash",
        `<span class="hash">${esc(b.experiment.binary_sha256)}</span>`,
      ],
      [
        "Structure hash",
        `<span class="hash">${esc(b.structure_sha256)}</span>`,
      ],
      ["Host", esc(b.experiment.platform)],
      ["Backend", esc(p?.execution_backend)],
      ["Seed", esc(b.experiment.seed)],
      [
        "Observations",
        esc(
          `${b.experiment.calibration_runs} calibration; ${b.experiment.evaluation_runs} held-out evaluation`,
        ),
      ],
      [isBounded(b) ? "Candidate comparison scope" : "Oracle scope", esc(isBounded(b) ? b.best_observed_definition : b.experiment.oracle_definition)],
      ["Warm-up scope", esc(b.experiment.warmup_scope)],
      ["Software buffer reuses", num(p?.metrics.software_buffer_reuses, 0)],
      [
        "Runtime peak reserved bytes",
        byte(p?.metrics.runtime_peak_device_reserved_bytes),
      ],
      ["Input setup H→D bytes", byte(p?.metrics.preparation_h2d_bytes)],
      [
        "Input setup p50",
        p?.metrics.input_preparation
          ? num(p.metrics.input_preparation.p50_ms) + " ms"
          : "unavailable",
      ],
      [
        "Requested injected host waits",
        p?.metrics.injected_boundary_ms == null
          ? "unavailable"
          : num(p.metrics.injected_boundary_ms) + " ms",
      ],
      ["Diagnostic readback bytes", byte(p?.metrics.diagnostic_readback_bytes)],
      [
        "Tile count / load bytes",
        `${num(p?.metrics.tile_count, 0)} / ${byte(p?.metrics.tile_load_bytes)}`,
      ],
      ["Peak local scratch", byte(p?.metrics.peak_local_bytes)],
      [
        isBounded(b) ? "Rejected attempted candidates" : "Rejected placements",
        isBounded(b) ? `${num(b.search.rejected, 0)} of ${num(b.search.evaluated, 0)} attempted; unvisited feasibility unknown` : `${num(b.rejections?.count, 0)}: ${esc(b.rejections?.reason_code)}`,
      ],
    ]) +
    Object.entries(b.provenance)
      .map(([k, v]) => `<p><strong>${esc(k)}:</strong> ${esc(v)}</p>`)
      .join("") +
    `<details><summary>Exact experiment configuration</summary><pre>${esc(JSON.stringify(b.experiment, null, 2))}</pre></details>`;
}
function renderComparison() {
  const el = $("comparison");
  if (!state.baseline) {
    el.textContent = "Pin a saved study, change one control, then run again.";
    return;
  }
  if (state.baseline === state.id) {
    el.textContent =
      "This study is the baseline. Change one control and run again to inspect a comparison.";
    return;
  }
  if (!state.diff) {
    el.textContent = state.compareError || "Loading baseline comparison…";
    return;
  }
  const d = state.diff,
    policy = d.policies?.[$("policy").value];
  el.innerHTML =
    `<p class="hint">Baseline ${esc(state.baseline.slice(0, 8))} → candidate ${esc(state.id.slice(0, 8))}. Structural differences are exact; timing deltas are observed, not a causal attribution.</p>` +
    (d.comparison_scope ? `<p class="notice">${esc(d.comparison_scope)} Candidate sets ${d.candidate_sets_equal ? "match" : "differ"}.</p>` : "") +
    d.changes
      .map(
        (c) =>
          `<div class="diff-change"><strong>${esc(c.entity_id)} · ${esc(c.reason_code)}</strong><code>${esc(JSON.stringify(c.before))} → ${esc(JSON.stringify(c.after))}</code></div>`,
      )
      .join("");
  if (!policy) {
    el.innerHTML += "<p>Comparison unavailable for this policy.</p>";
    return;
  }
  el.innerHTML += `<p><strong>${esc(policy.before_placement)} → ${esc(policy.after_placement)}</strong></p><p>${(policy.reason_codes || []).map(esc).join("; ")}</p>`;
  const structural = [
    ...(policy.added_commands || []).map(
      (c) => `Added #${c.id} ${c.kind}: ${c.inputs.join(", ")} → ${c.output}`,
    ),
    ...(policy.removed_commands || []).map(
      (c) => `Removed #${c.id} ${c.kind}: ${c.inputs.join(", ")} → ${c.output}`,
    ),
    ...(policy.changed_commands || []).map(
      (c) => `Changed ${c.stable_id}: ${c.before.kind} → ${c.after.kind}`,
    ),
  ];
  el.innerHTML += `<pre>${esc(structural.join("\n") || "No command structure changed.")}</pre>`;
  if (d.conclusions_valid && policy.metric_deltas) {
    el.innerHTML += table(
      ["Metric delta (candidate − baseline)", "Value", "Provenance"],
      Object.entries(policy.metric_deltas).map(
        ([k, v]) =>
          `<tr><td>${esc(k)}</td><td>${num(v)}</td><td>${k.includes("ms") ? "Measured observations; not causal" : "Exact structural"}</td></tr>`,
      ),
    );
  } else
    el.innerHTML +=
      '<p class="failure">Performance conclusions suppressed: correctness failed or a selected policy is unavailable.</p>';
}
function render() {
  const b = state.bundle;
  if (!b) return;
  syncPolicyScope();
  state.plan = b.plans.find((p) => p.id === selectedPolicy()?.plan_id) || null;
  if (!b.graph.nodes.some((n) => n.id === state.node))
    state.node = b.graph.nodes[0]?.id;
  $("empty").hidden = true;
  $("loaded").hidden = false;
  $("correctness").className = b.conclusions_valid ? "notice" : "failure";
  const hardwarePlan = b.plans.find((plan) =>
    plan.events.some(
      (event) => event.device_us != null && event.device_us >= 0,
    ),
  );
  $("correctness").textContent = b.conclusions_valid
    ? hardwarePlan
      ? `All recorded outputs matched the reference. This study includes measured device events from ${hardwarePlan.execution_backend}. This does not validate BeagleY-AI or other hardware.`
      : "All recorded calibration and evaluation outputs matched the reference. These host software observations do not validate accelerator hardware."
    : "Correctness failed. Performance rankings, regret and timing conclusions are suppressed. Raw measurements remain available for diagnosis.";
  $("study-title").textContent =
    b.experiment.workbench_label || "Saved execution study";
  $("identity-summary").textContent =
    `${state.id.slice(0, 8)} · ${b.plans.length} ${isBounded(b) ? "retained candidates" : "feasible placements"} · ${b.experiment.evaluation_runs} held-out runs · ${b.experiment.machine || "host unspecified"}`;
  $("search-scope").textContent = isBounded(b)
    ? `Bounded candidate search: ${b.search.evaluated}/${b.search.max_evaluations} whole-plan attempts, beam width ${b.search.beam_width}, ${b.plans.length} retained candidates. ${b.search.budget_exhausted ? "Search budget exhausted." : "Search stopped within budget."} Best observed compares only measured retained candidates. Unvisited placements may be faster; no global oracle or global regret is available.`
    : "Exhaustive small-graph study: every feasible placement is measured. The measured oracle is scoped to this graph, configuration and observation set.";
  $("raw").href = `/api/bundles/${state.id}`;
  $("pin").disabled = false;
  exportLink(
    "download",
    `/api/export/${state.id}/bundle`,
    `capstone-study-${state.id}.json`,
  );
  $("clear-baseline").disabled = !state.baseline;
  renderMetrics();
  renderGraph();
  renderInspector();
  renderCommands();
  renderTimeline();
  renderComparison();
  renderProvenance();
  renderEstimate();
  const prediction = predictionRead(state.id);
  $("prediction-result").hidden = !prediction;
  if (prediction) {
    const pp = b.plans.find(
      (p) => p.id === b.policies[prediction.policy]?.plan_id,
    );
    $("prediction-result").textContent = pp
      ? `Your prediction for ${prediction.policy}: ${prediction.count} COPY commands. Exact plan result: ${pp.commands.filter((c) => c.kind === "COPY").length}. Review transfers below; this is a single prediction, not a proficiency assessment.`
      : `Your prediction was saved. The ${prediction.policy} policy is unavailable, so no result is scored.`;
  }
}
async function loadStudy(id) {
  const request = ++state.request;
  error("");
  try {
    const b = await api(`/api/bundles/${id}`);
    if (request !== state.request) return;
    if (!((b.schema_version === 2 && b.artifact_type === "placement-study") || (b.schema_version === 1 && b.artifact_type === "bounded-placement-study")))
      throw new Error(
        "Unsupported artifact schema; expected exhaustive placement-study v2 or bounded-placement-study v1",
      );
    state.id = id;
    state.bundle = b;
    state.modelDetail = null;
    state.diff = null;
    $("study").value = id;
    render();
    await showModelEvidence(b.experiment.workbench_model_id || null);
    await loadComparison();
  } catch (e) {
    error(e.message);
  }
}
async function loadComparison() {
  state.compareError = null;
  if (!state.baseline || !state.id || state.baseline === state.id) {
    state.diff = null;
    renderComparison();
    return;
  }
  const before = state.baseline,
    after = state.id;
  try {
    const d = await api(`/api/compare?before=${before}&after=${after}`);
    if (before !== state.baseline || after !== state.id) return;
    state.diff = d;
    renderComparison();
    renderGraph();
  } catch (e) {
    error(e.message);
    state.compareError = `Comparison unavailable: ${e.message}. The saved studies remain inspectable.`;
    $("comparison").textContent = state.compareError;
  }
}
async function refresh() {
  try {
    const jobs = await api("/api/studies");
    state.jobs = jobs;
    syncEstimateModels();
    const running = jobs.find((j) => j.state === "running");
    state.busy = Boolean(running);
    $("run").disabled = state.busy || state.modelSubmitting;
    $("run-model").disabled = state.busy || state.modelSubmitting || !state.modelAvailable;
    $("run").textContent = state.busy
      ? "Experiment running…"
      : "Run experiment";
    const completed = jobs.filter((j) => j.state === "complete");
    $("study").innerHTML = completed.length
      ? completed
          .map(
            (j) =>
              `<option value="${esc(j.id)}">${esc(j.label)} / ${j.id.slice(0, 8)}</option>`,
          )
          .join("")
      : '<option value="">No saved studies</option>';
    if (state.id) $("study").value = state.id;
    $("status").textContent = running
      ? `Running ${running.label}. Raw reports are saved as each observation completes.`
      : `${completed.length} completed ${completed.length === 1 ? "study" : "studies"}. Ready to run.`;
    if (state.pending) {
      const job = jobs.find((j) => j.id === state.pending);
      if (job?.state === "complete") {
        state.pending = null;
        await loadStudy(job.id);
        if (job.kind === "onnx") $("model-availability").textContent = "ONNX exploration complete. Original-model validation passed; source, target and search scope are shown in the evidence.";
      } else if (job?.state === "failed") {
        state.pending = null;
        error(job.error);
        if (job.kind === "onnx") {
          $("model-availability").textContent = "ONNX exploration did not produce an accepted study. Review the diagnostic evidence.";
          await showModelEvidence(job.id);
        }
      }
    } else if (!state.id && completed.length)
      await loadStudy(completed[completed.length - 1].id);
  } catch (e) {
    error(e.message);
  } finally {
    setTimeout(refresh, 1800);
  }
}
$("experiment-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (state.busy) return;
  error("");
  const form = new FormData(event.currentTarget),
    payload = {};
  for (const [key, value] of form) {
    payload[key] = key === "topology" ? value : Number(value);
  }
  for (const key of [
    "relu_supported",
    "fusion",
    "reuse_buffers",
    "asynchronous",
    "mirrored_inputs",
    "emulate_boundaries",
  ])
    payload[key] = event.currentTarget.elements[key].checked;
  $("run").disabled = true;
  try {
    const job = await api("/api/run", payload);
    state.pending = job.id;
    state.busy = true;
    $("status").textContent =
      "Experiment submitted. Waiting for the core study runner…";
    if ($("prediction").value !== "")
      predictionStore(job.id, {
        count: Number($("prediction").value),
        policy: $("policy").value,
      });
  } catch (e) {
    error(e.message);
    $("run").disabled = false;
  }
});
$("study").addEventListener("change", (e) => {
  if (e.target.value) loadStudy(e.target.value);
});
$("policy").addEventListener("change", render);
$("pin").addEventListener("click", () => {
  state.baseline = state.id;
  state.diff = null;
  render();
});
$("clear-baseline").addEventListener("click", () => {
  state.baseline = null;
  state.diff = null;
  render();
});
document.addEventListener("click", (e) => {
  const node = e.target.closest("[data-node]");
  if (node) {
    state.node = node.dataset.node;
    render();
    if (node.closest("#target-estimate")) $("graph").scrollIntoView({ block: "nearest" });
    return;
  }
  const command = e.target.closest("[data-command]");
  if (command && state.plan) {
    const c = state.plan.commands.find(
      (x) => String(x.id) === command.dataset.command,
    );
    const n =
      state.bundle.graph.nodes.find((n) => n.id === c?.node) ||
      state.bundle.graph.nodes.find((n) =>
        [n.output, ...n.inputs].some(
          (t) =>
            [c?.output, ...(c?.inputs || [])].includes("H:" + t) ||
            [c?.output, ...(c?.inputs || [])].includes("D:" + t),
        ),
      );
    if (n) {
      state.node = n.id;
      render();
    }
  }
});
$("download").addEventListener("click", requestExport);
$("trace-export").addEventListener("click", requestExport);
async function showModelEvidence(id) {
  const request = (state.modelRequest || 0) + 1;
  state.modelRequest = request;
  $("evidence").removeAttribute("data-model-diagnostic");
  $("model-evidence").hidden = !id;
  state.modelDetail = null;
  if (!id) { if (state.bundle) renderInspector(); return; }
  try {
    const detail = await api(`/api/model-results/${id}`), r = detail.result;
    if (request !== state.modelRequest) return;
    if (r.status !== "verified") $("evidence").setAttribute("data-model-diagnostic", "true");
    state.modelDetail = detail;
    if (state.bundle && r.status === "verified") renderInspector();
    const source = detail.upload;
    const issues = detail.compatibility?.issues || r.issues || [];
    $("model-provenance").innerHTML = kv([
      ["Uploaded model", esc(source.model_name)],
      ["Named inputs", esc(source.inputs_name)],
      ["Import / execution status", esc(r.status)],
      ["Target", esc(r.profile.id)],
      ["Evidence level", esc(r.profile.evidence.level)],
      ["Execution scope", esc(r.execution_scope)],
      ["Search scope", esc(r.search_scope || "Unavailable")],
      ["Original ONNX reference", r.numerical_execution_validated === true ? `${esc(r.placements_checked)} placements matched ONNX Runtime CPU` : "Not validated; no performance conclusion"],
      ["Hardware validation", "Not performed"],
      ["Task accuracy", "Not evaluated; numerical agreement is not task accuracy"],
      ["Model SHA-256", `<code>${esc(source.model_sha256)}</code>`],
    ]) + (issues.length ? `<h3>Import diagnostics</h3><ul>${issues.map(i => `<li>${esc(i.code)}: ${esc(i.message)}</li>`).join("")}</ul>` : "") +
    (r.error ? `<p>${esc(r.error)}</p>` : "") +
    (detail.target_analysis ? `<details><summary>Target capability and source mapping</summary><pre>${esc(JSON.stringify(detail.target_analysis, null, 2))}</pre></details>` : "");
  } catch (failure) {
    if (request !== state.modelRequest) return;
    $("evidence").setAttribute("data-model-diagnostic", "true");
    $("model-provenance").textContent = `Model evidence unavailable: ${failure.message}`;
  }
}
function targetDescription() {
  const profile = state.profiles.find(p => p.id === $("target-profile").value);
  $("target-description").textContent = profile ? `${profile.description} Evidence: ${profile.evidence.level}. ${profile.evidence.source}` : "";
}
async function loadModelCapabilities() {
  try {
    const c = await api("/api/model-capabilities");
    state.modelAvailable = c.available;
    state.profiles = c.profiles;
    $("target-profile").innerHTML = c.profiles.map(p => `<option value="${esc(p.id)}">${esc(p.id)} · ${esc(p.evidence.level)}</option>`).join("");
    targetDescription();
    $("model-availability").textContent = c.available ? c.scope : `ONNX unavailable in the server's Python environment. Missing: ${c.missing_dependencies.join(", ")}. Generated graphs still work.`;
    $("run-model").disabled = !c.available || state.busy;
  } catch (failure) {
    $("model-availability").textContent = `ONNX unavailable: ${failure.message}`;
  }
}
function modelError(message) {
  $("model-error").hidden = !message;
  $("model-error").textContent = message || "";
}
function checkFile(file) {
  if (!file || file.size < 1 || file.size > 16 * 1024 * 1024)
    throw new Error("Choose a nonempty file of at most 16 MiB for both model and inputs.");
}
async function fileBase64(file) {
  checkFile(file);
  const bytes = new Uint8Array(await file.arrayBuffer());
  let text = "";
  for (let start = 0; start < bytes.length; start += 8192)
    text += String.fromCharCode(...bytes.subarray(start, start + 8192));
  return btoa(text);
}
for (const id of ["model-file", "inputs-file"])
  $(id).addEventListener("change", event => {
    try { checkFile(event.target.files[0]); modelError(""); }
    catch (failure) { modelError(failure.message); }
  });
$("target-profile").addEventListener("change", targetDescription);
$("model-form").addEventListener("submit", async event => {
  event.preventDefault();
  if (state.busy || state.modelSubmitting || !state.modelAvailable) return;
  modelError(""); error("");
  state.modelSubmitting = true;
  $("run-model").disabled = true;
  $("run").disabled = true;
  try {
    const model = $("model-file").files[0], inputs = $("inputs-file").files[0];
    checkFile(model); checkFile(inputs);
    $("model-availability").textContent = "Uploading local files; validating model semantics and original ONNX output before publishing measurements…";
    const job = await api("/api/models/run", {
      model_name: model.name, model_base64: await fileBase64(model),
      inputs_name: inputs.name, inputs_base64: await fileBase64(inputs),
      profile_id: $("target-profile").value,
    });
    state.pending = job.id;
    state.busy = true;
    $("model-availability").textContent = "ONNX exploration queued. Inspect the source and target evidence after validation.";
  } catch (failure) {
    modelError(failure.message);
  } finally {
    state.modelSubmitting = false;
    $("run-model").disabled = state.busy || !state.modelAvailable;
    $("run").disabled = state.busy;
  }
});
// Architecture target estimate: scripts/architecture_model.py runs on the
// server; this code only renders its JSON and computes no cost or placement.
const ms = (value) => (value == null ? "unavailable" : `${num(value, 6)} ms`);
function estimateError(message) {
  $("estimate-error").hidden = !message;
  $("estimate-error").textContent = message || "";
}
function estimateTargets() {
  return [...document.querySelectorAll("#estimate-targets input:checked")].map((i) => i.value);
}
function syncEstimateButton() {
  const count = estimateTargets().length, max = state.architectureMax || 4;
  $("run-estimate").disabled = Boolean(state.estimateBusy) || count < 1 || count > max;
  $("run-estimate").textContent = state.estimateBusy ? "Estimating…" : count < 1 || count > max ? `Choose 1–${max} targets` : count > 1 ? `Compare ${count} targets` : "Estimate on 1 target";
}
function syncEstimateModels() {
  const models = state.jobs.filter((j) => j.kind === "onnx"),
    key = models.map((j) => j.id + j.state).join();
  if (key === state.estimateModelsKey) return;
  state.estimateModelsKey = key;
  const select = $("estimate-model"), previous = select.value;
  select.innerHTML = models.map((j) =>
    `<option value="${esc(j.id)}">Saved: ${esc(j.label.replace(/^ONNX: /, ""))} / ${esc(j.id.slice(0, 8))}${j.state === "complete" ? "" : ` (${esc(j.state)})`}</option>`).join("") +
    '<option value="file">File chosen under “Bring an ONNX model”</option>';
  const current = state.bundle?.experiment.workbench_model_id;
  select.value = [previous, current, models[0]?.id].find((v) => v && v !== "file" && models.some((j) => j.id === v)) || previous || "file";
}
async function loadArchitectures() {
  try {
    const c = await api("/api/architectures");
    state.architectureMax = c.compare_max;
    $("estimate-targets").innerHTML = c.profiles.map((p, i) =>
      `<label class="check"><input type="checkbox" value="${esc(p.id)}"${i === 0 ? " checked" : ""} /><span><strong>${esc(p.id)}</strong> · r${esc(p.revision)} · ${esc(p.evidence.level)}<small>${esc(p.description)}</small></span></label>`).join("");
    syncEstimateButton();
  } catch (failure) {
    $("estimate-targets").textContent = `Architecture profiles unavailable: ${failure.message}`;
  }
}
function graphNodeFor(generated) {
  // Link only when the open study is the same saved ONNX upload.
  const b = state.bundle;
  if (!b || !state.estimateModel || b.experiment.workbench_model_id !== state.estimateModel) return null;
  return b.graph.nodes.find((n) => (n.origins || [n.id]).includes(generated))?.id ?? null;
}
const scroll = (html) => `<div class="table-scroll">${html}</div>`;
function estimateReport(r, inComparison) {
  const head = `<p class="estimate-meta">${tag("Not hardware-validated", "unavailable")} <strong>${esc(r.target)}</strong> r${esc(r.target_revision)} · evidence ${esc(r.evidence.level)} · estimator ${esc(r.estimator)}${r.resident_weights ? " · weights resident" : " · weights start on host"}</p>`;
  if (!r.analysis)
    return head + `<p class="failure">${tag("Unavailable", "unavailable")} No latency: the importer rejected these semantics, so no placement or time is estimated.</p>` +
      (r.issues.length ? `<ul class="estimate-issues">${r.issues.map((i) => `<li>${esc(i.code)}: ${esc(i.message)}</li>`).join("")}</ul>` : "");
  const a = r.analysis, s = a.selected, c = s.components_ms, range = s.scenario_range_ms;
  const metric = ([label, value, provenance, kind]) =>
    `<div class="metric"><span class="label">${esc(label)}</span><span class="value">${esc(value)}</span>${tag(provenance, kind)}</div>`;
  const feasible = s.estimated_ms != null;
  const rows = [
    ["Selected estimate", feasible ? ms(s.estimated_ms) : "Infeasible", feasible ? "Modeled" : "Unavailable", feasible ? "modeled" : "unavailable"],
    ["Assumption range, not a confidence interval", range ? `${num(range[0], 6)}–${num(range[1], 6)} ms` : "unavailable", range ? "Modeled" : "Unavailable", range ? "modeled" : "unavailable"],
    ["Kernels / transfers / syncs", `${num(c.compute_ms, 6)} / ${num(c.transfer_ms, 6)} / ${num(c.sync_ms, 6)} ms`, "Modeled", "modeled"],
    ["Dominant cost", `${s.dominant_cost.component}: ${ms(s.dominant_cost.ms)}`, "Modeled", "modeled"],
    ["Placement (C = CPU, A = accelerator)", s.placement, "Exact", "exact"],
    ["Moved H→D / D→H", `${byte(s.h2d_bytes)} / ${byte(s.d2h_bytes)}`, "Exact", "exact"],
    ["Synchronizations", num(s.syncs, 0), "Exact", "exact"],
    ["Peak device / declared memory", `${byte(s.device_peak_bytes)} / ${byte(r.profile.accelerator.memory_bytes)}`, "Exact", "exact"],
  ];
  const baseline = (label, b) =>
    `${label} ${esc(b.placement)}: ${b.estimated_ms != null ? esc(ms(b.estimated_ms)) : "infeasible (" + b.failures.flatMap((f) => f.reasons).map(esc).join("; ") + ")"}`;
  const linked = s.nodes.some((n) => graphNodeFor(n.node));
  return head +
    `<div class="metrics estimate-metrics">${rows.map(metric).join("")}</div>` +
    (s.failures.length ? `<p class="failure">${s.failures.flatMap((f) => f.reasons).map(esc).join("; ")}</p>` : "") +
    `<p class="reason">Search: ${esc(a.search.mode)}; ${num(a.search.evaluated, 0)} placements scored; ${a.search.optimum_of_this_model ? "optimum of this model" : "a candidate, not an optimum"}; ${num(a.infeasible_candidates, 0)} infeasible; ${num(a.ties_with_selected.length, 0)} tied. ${baseline("CPU only", a.cpu_only)}. ${baseline("Maximal supported", a.maximal_supported)}.</p>` +
    `<h3>Placement per normalized node</h3><p class="hint">${linked ? "Select a node to highlight it in the graph above." : "Rows link to the graph when the open saved study is this same ONNX upload."} Operations and traffic are exact counts from shapes; kernel time is modeled.</p>` +
    scroll(table(["Node", "Source node", "Operator", "Placement", "Ops / traffic", "Bound", "Kernel time", "Fallback reason"],
      s.nodes.map((n) => {
        const g = graphNodeFor(n.node);
        return `<tr class="${g && g === state.node ? "linked" : ""}"><td>${g ? `<button type="button" data-node="${esc(g)}">${esc(n.node)}</button>` : esc(n.node)}</td><td>#${esc(n.source_index)} ${esc(n.source_name || "unnamed")}</td><td>${esc(n.operator)}</td><td>${n.backend === "A" ? "Accelerator" : "CPU"}</td><td>${num(n.operations, 0)} ops<br>${byte(n.traffic_bytes)}</td><td>${esc(n.bottleneck)}</td><td>${esc(ms(n.total_ms))}</td><td>${n.device_rejection_reasons.length ? n.device_rejection_reasons.map(esc).join("; ") : n.backend === "C" ? "None: supported, CPU chosen" : "—"}</td></tr>`;
      }))) +
    (r.sensitivity.length
      ? `<h3>Sensitivity ${tag("Modeled", "modeled")}</h3><p class="hint">One declared parameter at a time (×0.5, ×2); the placement is reselected each time.</p>` +
        scroll(table(["Parameter", "Factor", "Estimate", "Change", "Placement"], r.sensitivity.map((x) =>
          `<tr><td>${esc(x.parameter)}</td><td>×${esc(x.factor)}</td><td>${x.estimated_ms == null ? "Infeasible" : esc(ms(x.estimated_ms))}</td><td>${x.change_ms == null ? "unavailable" : (x.change_ms > 0 ? "+" : "") + esc(ms(x.change_ms))}</td><td>${esc(x.placement)}${x.changed_placement ? " <strong>changed</strong>" : ""}</td></tr>`)))
      : inComparison ? "" : "<p>Sensitivity unavailable.</p>") +
    `<details><summary>Value kinds and model limits</summary>${kv(Object.entries(r.value_kinds).map(([k, v]) => [k, esc(v)]))}<ul>${r.limitations.map((x) => `<li>${esc(x)}</li>`).join("")}</ul></details>`;
}
function renderEstimate() {
  const c = state.estimate;
  if (!c) { $("estimate-result").innerHTML = '<p class="hint">No estimate yet. Choose a model and at least one target, then estimate.</p>'; return; }
  if (c.kind !== "target comparison") { $("estimate-result").innerHTML = estimateReport(c, false); return; }
  const flips = c.winner_flips;
  $("estimate-result").innerHTML =
    `<p class="${c.robust_winner ? "notice" : "failure"} estimate-conclusion">${tag("Modeled", "modeled")} ${esc(c.conclusion)}</p>` +
    `<p class="hint">${esc(c.scope)} Objective: ${esc(c.objective)}. Ranges are assumption ranges, not confidence intervals. Hardware recommendation validated: no.</p>` +
    scroll(table(["Target", "Estimate", "Assumption range", "Placement", "Moved", "Syncs", "Dominant cost"], c.reports.map((r) => {
      const s = r.analysis?.selected;
      if (!s) return `<tr><td>${esc(r.target)}</td><td colspan="6">Importer rejected the model; no latency</td></tr>`;
      const range = s.scenario_range_ms;
      return `<tr><td><strong>${esc(r.target)}</strong></td><td>${s.estimated_ms == null ? "Infeasible under declared limits" : esc(ms(s.estimated_ms))}</td><td>${range ? `${num(range[0], 6)}–${num(range[1], 6)} ms` : "unavailable"}</td><td class="handle">${esc(s.placement)}</td><td>${byte(s.transfer_bytes)}</td><td>${num(s.syncs, 0)}</td><td>${esc(s.dominant_cost.component)}</td></tr>`;
    }))) +
    `<p class="reason">Nominal order: ${c.nominal_order_text ? esc(c.nominal_order_text) : "none"}. Efficiency scenario ranges ${c.separated_by_efficiency_scenarios ? "separate the leader" : "do not separate the leader"}. ${num(flips.length, 0)} one-parameter perturbation(s) change the winner.</p>` +
    (flips.length ? `<details><summary>Winner flips (${num(flips.length, 0)})</summary>${scroll(table(["Changed target", "Parameter", "Factor", "New winner"], flips.map((f) =>
      `<tr><td>${esc(f.target_changed)}</td><td>${esc(f.parameter)}</td><td>×${esc(f.factor)}</td><td>${esc(f.new_winner)}</td></tr>`)))}</details>` : "") +
    c.reports.map((r) => `<details><summary>${esc(r.target)}: placement per node and cost components</summary>${estimateReport(r, true)}</details>`).join("");
}
$("estimate-targets").addEventListener("change", syncEstimateButton);
$("estimate-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const ids = estimateTargets(), source = $("estimate-model").value;
  if (state.estimateBusy || !ids.length) return;
  estimateError("");
  state.estimateBusy = true;
  $("estimate-result").setAttribute("aria-busy", "true");
  syncEstimateButton();
  try {
    const body = { resident_weights: $("estimate-resident").checked };
    if (source === "file") {
      const file = $("model-file").files[0];
      if (!file) throw new Error("Choose an ONNX file under “Bring an ONNX model”, or pick a saved model.");
      body.model_name = file.name;
      body.model_base64 = await fileBase64(file);
    } else body.model_id = source;
    if (ids.length > 1) body.profile_ids = ids;
    else body.profile_id = ids[0];
    $("estimate-status").textContent = "Running the analytical model on the server…";
    const result = await api(ids.length > 1 ? "/api/architectures/compare" : "/api/architectures/estimate", body);
    state.estimate = result;
    state.estimateModel = source === "file" ? null : source;
    renderEstimate();
    $("estimate-status").textContent = `${ids.length > 1 ? "Comparison" : "Estimate"} for ${result.reports?.[0]?.model ?? result.model} complete. Values are the server's architecture-model output; this page recomputes nothing.`;
  } catch (failure) {
    estimateError(failure.message);
    $("estimate-status").textContent = "";
  } finally {
    state.estimateBusy = false;
    $("estimate-result").removeAttribute("aria-busy");
    syncEstimateButton();
  }
});
loadArchitectures();
loadModelCapabilities();
refresh();
