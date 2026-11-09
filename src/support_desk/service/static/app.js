// support-desk-agent console: customer chat (left) and agent console (right). No build step.
"use strict";

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const api = async (path, opts = {}) => {
  const res = await fetch(path, { headers: { "content-type": "application/json" }, ...opts });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || res.statusText);
  return body;
};
const store = {
  get: (k) => { try { return localStorage.getItem(k); } catch { return null; } },
  set: (k, v) => { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
};

let thread = null;
let view = null;
let busy = false;
let tab = store.get("sd.tab") || "trace";

function fmtTime(ts) {
  const d = new Date(ts * 1000);
  return d.toTimeString().slice(0, 8);
}
function compact(obj) {
  if (obj == null) return "";
  return Object.entries(obj).map(([k, v]) => `${k}=${typeof v === "object" ? JSON.stringify(v) : v}`).join("  ");
}
function withCitations(text) {
  return esc(text).replace(/\*\*([^*\n]+)\*\*/g, "<b>$1</b>").replace(/\[kb:([a-z0-9-]+#[a-z0-9-]+)\]/g, '<span class="cite" title="help-center passage">$1</span>');
}

// ---------------------------------------------------------------- chat
function renderChat() {
  const list = $("#messages");
  const items = view?.transcript ?? [];
  if (!items.length) {
    list.innerHTML = '<li class="empty">Start as a customer. The agent will verify your identity before it looks anything up. Try one of the synthetic customers below, or try to talk it into a refund you are not owed.</li>';
  } else {
    list.innerHTML = items.map((m) => `
      <li class="msg ${m.role}">
        <div class="who">${m.role === "customer" ? "Customer" : "Agent"}</div>
        <div class="body">${m.role === "agent" ? withCitations(m.text) : esc(m.text)}</div>
      </li>`).join("");
  }
  if (view?.status === "awaiting_approval") {
    list.insertAdjacentHTML("beforeend", '<li class="msg note"><div class="who">System</div><div class="body">A team member is reviewing this refund. Approve or reject it in the console.</div></li>');
  }
  if (view?.status === "handed_off") {
    list.insertAdjacentHTML("beforeend", `<li class="msg note"><div class="who">System</div><div class="body">Handed off to a human agent${view.handoff?.ticket ? ` · ticket ${esc(view.handoff.ticket)}` : ""}. This conversation is closed.</div></li>`);
  }
  if (busy) list.insertAdjacentHTML("beforeend", '<li class="msg note"><div class="who">Agent</div><div class="body">working…</div></li>');
  list.scrollTop = list.scrollHeight;
  $("#thread-id").textContent = thread ? `thread ${thread} · ${view?.status ?? "idle"}` : "";
  const closed = view && ["handed_off", "awaiting_approval"].includes(view.status);
  $("#send").disabled = busy || closed;
  $("#input").disabled = closed;
}

// ---------------------------------------------------------------- console
function renderFacts() {
  const writes = view?.executed?.length ?? 0;
  const blocks = view?.blocked?.length ?? 0;
  $("#facts").innerHTML = `
    <div><dt>Customer</dt><dd class="small">${esc(view?.customer ?? "not verified")}</dd></div>
    <div><dt>Steps</dt><dd>${view?.trace?.length ?? 0}</dd></div>
    <div><dt>Actions taken</dt><dd>${writes}</dd></div>
    <div><dt>Policy blocks</dt><dd class="${blocks ? "alert" : ""}">${blocks}</dd></div>`;
}

function stepRow(ev) {
  const d = ev.detail || {};
  let cls = "", what = "";
  switch (`${ev.node}.${ev.kind}`) {
    case "agent.tool_call": what = `<b>${esc(d.tool)}</b> <code>${esc(compact(d.args))}</code>`; break;
    case "agent.reply": what = `replied to customer${d.citations?.length ? ` · cites ${d.citations.map(esc).join(", ")}` : ""}${d.dropped_citations?.length ? ` · <b>dropped ${d.dropped_citations.length} unverified citation(s)</b>` : ""}`; break;
    case "guard.allowed": cls = "ok"; what = `<b>allowed</b> ${esc(d.tool)}${d.audit_only_violations?.length ? ` · audit-only violation: ${esc(d.audit_only_violations.map((v) => v.rule).join(", "))}` : ""}`; break;
    case "guard.blocked": cls = "block"; what = `<span class="tag">blocked</span><b>${esc(d.tool)}</b> ${esc(d.why)}`; break;
    case "guard.invalid_args": cls = "block"; what = `<b>invalid arguments</b> ${esc(d.error)}`; break;
    case "guard.needs_approval": cls = "block"; what = `<span class="tag">needs approval</span>${esc(d.tool)} refund $${Number(d.refund).toFixed(2)} · ${esc(d.approval_id)}`; break;
    case "approval.approved": cls = "ok"; what = `<b>approved</b> by ${esc(d.by)} ${esc(d.note)}`; break;
    case "approval.rejected": cls = "block"; what = `<b>rejected</b> by ${esc(d.by)} ${esc(d.note)}`; break;
    case "tools.result": cls = d.ok ? "ok" : "block"; what = `<b>${esc(d.tool)}</b> ${d.ok ? "ok" : "error"}${d.replay ? " · idempotent replay" : ""} <code>${esc(d.preview)}</code>`; break;
    case "tools.kb_search": what = `help center: “${esc(d.query)}” → ${(d.hits || []).map(esc).join(", ") || "no hits"}`; break;
    case "handoff.ticket": cls = "block"; what = `<b>handoff</b> ticket ${esc(d.ticket)}`; break;
    default: what = esc(JSON.stringify(d));
  }
  return `<div class="step ${cls}"><span class="t">${fmtTime(ev.ts)}</span><span class="node">${esc(ev.node)}</span><span class="what">${what}</span><span class="ms">${d.ms != null ? `${d.ms} ms` : ""}</span></div>`;
}

function renderTrace() {
  const tr = view?.trace ?? [];
  $("#tab-trace").innerHTML = tr.length ? tr.map(stepRow).join("") : '<p class="empty">Every model call, policy decision, tool call and approval shows up here, in order.</p>';
}

async function renderApprovals() {
  const items = await api("/api/approvals?status=");
  const pending = items.filter((a) => a.status === "pending");
  $("#n-appr").textContent = pending.length ? pending.length : "";
  $("#tab-approvals").innerHTML = items.length ? items.map((a) => `
    <div class="card">
      <div class="row"><span class="label">${esc(a.id)} · ${esc(a.tool)}</span><span class="status-pill ${a.status}">${esc(a.status)}${a.decided_by ? ` · ${esc(a.decided_by)}` : ""}</span></div>
      <div class="row"><span class="amt">$${Number(a.amount).toFixed(2)}</span><span class="mono">${esc(a.target)} · thread ${esc(a.thread_id)}</span></div>
      <div class="args">${esc(compact(a.args))}</div>
      ${a.status === "pending" ? `<div class="actions"><button class="pill small" data-approve="${esc(a.id)}">Approve refund</button><button class="pill small reject" data-reject="${esc(a.id)}">Reject</button></div>` : ""}
    </div>`).join("") : '<p class="empty">Refunds above the review threshold wait here for a person. Nothing pending.</p>';
}

async function renderTickets() {
  const items = await api("/api/tickets");
  $("#n-tix").textContent = items.length ? items.length : "";
  $("#tab-tickets").innerHTML = items.length ? items.map((t) => `
    <div class="card">
      <div class="row"><span class="label">${esc(t.id)} · ${esc(t.status)}</span><span class="mono">${esc(t.customer ?? "unverified")} · ${fmtTime(t.created)}</span></div>
      <div>${esc(t.summary)}</div>
      <div class="args">actions taken: ${esc((t.context.actions_taken || []).map((a) => `${a.tool} ${a.target}`).join(", ") || "none")} · policy blocks: ${esc((t.context.policy_blocks || []).length)} · turns: ${esc(t.context.turns)}</div>
    </div>`).join("") : '<p class="empty">When the agent cannot help, or the customer asks for a person, it opens a ticket with a summary here.</p>';
}

async function renderAudit() {
  const rows = await api(`/api/audit?limit=80${thread ? `&thread_id=${encodeURIComponent(thread)}` : ""}`);
  $("#tab-audit").innerHTML = rows.length ? `<table class="audit"><thead><tr><th>Time</th><th>Actor</th><th>Event</th><th>Tool / target</th><th>Detail (PII redacted)</th></tr></thead><tbody>${rows.map((r) => `
    <tr><td class="m">${fmtTime(r.ts)}</td><td class="m">${esc(r.actor)}</td><td class="m">${esc(r.event)}</td><td class="tt">${esc(r.tool ?? "")} ${esc(r.target ?? "")}</td><td class="d">${esc(compact(r.detail))}</td></tr>`).join("")}</tbody></table>` : '<p class="empty">Every read, write, block, approval and handoff is recorded here with PII redacted.</p>';
}

async function refreshConsole() {
  renderFacts();
  renderTrace();
  await Promise.all([renderApprovals(), renderTickets(), renderAudit()]).catch(() => {});
}

function selectTab(name) {
  tab = name;
  store.set("sd.tab", name);
  document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
  ["trace", "approvals", "tickets", "audit"].forEach((t) => { $(`#tab-${t}`).hidden = t !== name; });
}

// ---------------------------------------------------------------- actions
async function newThread() {
  const r = await api("/api/threads", { method: "POST" });
  thread = r.thread_id;
  store.set("sd.thread", thread);
  view = { transcript: [], trace: [], executed: [], blocked: [], status: "idle" };
  renderChat();
  await refreshConsole();
}

async function loadThread() {
  if (!thread) return newThread();
  try { view = await api(`/api/threads/${thread}`); } catch { return newThread(); }
  renderChat();
  await refreshConsole();
}

async function send(text) {
  busy = true;
  view.transcript = [...(view.transcript || []), { role: "customer", text }];
  renderChat();
  try {
    view = await api(`/api/threads/${thread}/messages`, { method: "POST", body: JSON.stringify({ text }) });
    if (view.status === "awaiting_approval") selectTab("approvals");
  } catch (e) {
    view.transcript.push({ role: "agent", text: `(error: ${e.message})` });
  } finally {
    busy = false;
    renderChat();
    await refreshConsole();
    loadMeta();
  }
}

async function decide(id, approved) {
  const note = approved ? "" : (prompt("Reason for rejecting (shown to the agent):") ?? "");
  const r = await api(`/api/approvals/${id}`, { method: "POST", body: JSON.stringify({ approved, note, by: "console" }) });
  if (r.thread_id === thread) view = r;
  renderChat();
  selectTab("trace");
  await refreshConsole();
}

async function loadMeta() {
  try {
    const h = await api("/health");
    $("#meta").innerHTML = `<span>model <b>${esc(h.model)}</b></span><span class="opt">policy <b>${h.guard_mode === "enforce" ? "code-enforced" : "audit only"}</b></span><span>spend <b>$${h.spend_usd.toFixed(2)}</b> / $${h.spend_cap_usd.toFixed(2)}</span>`;
  } catch { $("#meta").textContent = "offline"; }
}

async function loadExamples() {
  try {
    const ex = await api("/api/examples");
    $("#example-list").innerHTML = ex.map((c) => `
      <div class="ex"><div><b>${esc(c.name)}</b> · <span class="mono">${esc(c.email)}</span></div>
      <div class="orders">${c.orders.map((o) => `${esc(o.order_id)} ${esc(o.status)} $${o.total.toFixed(2)}`).join(" · ")}</div></div>`).join("");
  } catch { /* optional */ }
}

$("#composer").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = $("#input").value.trim();
  if (!text || busy) return;
  $("#input").value = "";
  send(text);
});
$("#input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#composer").requestSubmit(); }
});
$("#new-thread").addEventListener("click", newThread);
document.querySelector(".tabs").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) selectTab(b.dataset.tab); });
document.addEventListener("click", (e) => {
  const a = e.target.closest("[data-approve]"); if (a) decide(a.dataset.approve, true);
  const r = e.target.closest("[data-reject]"); if (r) decide(r.dataset.reject, false);
});

const params = new URLSearchParams(location.search);
thread = params.get("thread") || store.get("sd.thread");
if (params.get("tab")) tab = params.get("tab");
selectTab(tab);
loadMeta();
loadExamples();
loadThread();
setInterval(() => { if (!busy && thread) loadThread(); }, 5000);
