// support-desk-agent inbox: conversation list, thread, customer/order panel, approvals and the
// agent's activity timeline. Talks only to the existing JSON API. No build step.
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
const icon = (name) => `<svg class="ic" aria-hidden="true"><use href="#i-${name}"/></svg>`;
const money = (n) => `$${Number(n).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

let thread = null;
let view = null;
let busy = false;
let tab = store.get("sd.tab") || "trace";
let approvalsCache = [];
const previews = new Map(); // thread_id -> { updated, view }

function fmtTime(ts) {
  const d = new Date(ts * 1000);
  return d.toTimeString().slice(0, 8);
}
function fmtWhen(ts) {
  const d = new Date(ts * 1000);
  const today = new Date();
  return d.toDateString() === today.toDateString()
    ? d.toTimeString().slice(0, 5)
    : d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}
function compact(obj) {
  if (obj == null) return "";
  return Object.entries(obj).map(([k, v]) => `${k}=${typeof v === "object" ? JSON.stringify(v) : v}`).join("  ");
}
function withCitations(text) {
  return esc(text).replace(/\*\*([^*\n]+)\*\*/g, "<b>$1</b>").replace(/\[kb:([a-z0-9-]+#[a-z0-9-]+)\]/g, '<span class="cite" title="help-center passage">$1</span>');
}
function statusBadge(status) {
  if (status === "awaiting_approval") return `<span class="badge waiting">${icon("clock")}Waiting approval</span>`;
  if (status === "handed_off") return `<span class="badge handed">${icon("handoff")}Handed off</span>`;
  return `<span class="badge open">${icon("chat")}Open</span>`;
}
const TOOL_LABEL = {
  cancel_pending_order: "Cancel order",
  return_delivered_order_items: "Return items",
  exchange_delivered_order_items: "Exchange items",
  modify_pending_order_items: "Modify items",
  modify_pending_order_address: "Change address",
  modify_pending_order_payment: "Change payment",
  modify_user_address: "Change account address",
};
const toolLabel = (t) => TOOL_LABEL[t] || t;

// ---------------------------------------------------------------- conversation list
async function loadThreads() {
  let rows;
  try { rows = await api("/api/threads"); } catch { return; }
  rows = rows.slice(0, 25);
  await Promise.all(rows.map(async (r) => {
    const cached = previews.get(r.thread_id);
    if (cached && cached.updated === r.updated) return;
    try { previews.set(r.thread_id, { updated: r.updated, view: await api(`/api/threads/${r.thread_id}`) }); } catch { /* skip */ }
  }));
  renderThreads(rows);
}

function renderThreads(rows) {
  const counts = { open: 0, waiting: 0, handed: 0 };
  rows.forEach((r) => { counts[r.status === "awaiting_approval" ? "waiting" : r.status === "handed_off" ? "handed" : "open"] += 1; });
  $("#list-sub").textContent = `${counts.open} open · ${counts.waiting} waiting approval · ${counts.handed} handed off`;
  $("#threads").innerHTML = rows.map((r) => {
    const v = r.thread_id === thread && view ? view : previews.get(r.thread_id)?.view;
    const tr = v?.transcript ?? [];
    const first = tr.find((m) => m.role === "customer");
    const last = tr[tr.length - 1];
    const who = v?.customer ? `<span class="mono">${esc(v.customer)}</span>` : "New visitor";
    return `<li><button class="thread" type="button" data-thread="${esc(r.thread_id)}" aria-current="${r.thread_id === thread}">
      <span class="top"><span class="who">${who}</span><span class="when">${fmtWhen(r.updated)}</span></span>
      <span class="subj">${first ? esc(first.text) : "No messages yet"}</span>
      ${last && last !== first ? `<span class="snip">${last.role === "agent" ? "Agent: " : ""}${esc(last.text)}</span>` : ""}
      <span class="badges">${statusBadge(r.status)}</span>
    </button></li>`;
  }).join("") || '<li class="empty">No conversations yet.</li>';
}

// ---------------------------------------------------------------- thread
function renderChat() {
  const list = $("#messages");
  const items = view?.transcript ?? [];
  if (!items.length) {
    list.innerHTML = '<li class="empty">Start as a customer. The agent will verify your identity before it looks anything up. Try one of the synthetic customers below, or try to talk it into a refund you are not owed.</li>';
  } else {
    list.innerHTML = items.map((m) => m.role === "customer" ? `
      <li class="msg customer">
        <div class="av" aria-hidden="true">${icon("chat")}</div>
        <div class="col"><div class="who"><b>Customer</b>${view?.customer ? ` · <span class="mono">${esc(view.customer)}</span>` : ""}</div>
        <div class="body">${esc(m.text)}</div></div>
      </li>` : `
      <li class="msg agent">
        <div class="av" aria-hidden="true">AI</div>
        <div class="col"><div class="who"><b>Support agent</b> · AI</div>
        <div class="body">${withCitations(m.text)}</div></div>
      </li>`).join("");
  }
  if (view?.status === "awaiting_approval") {
    list.insertAdjacentHTML("beforeend", `<li class="msg note"><div class="body">${icon("clock")}<span><b>System note</b> · A team member is reviewing this refund. Approve or reject it in the approval card.</span></div></li>`);
  }
  if (view?.status === "handed_off") {
    list.insertAdjacentHTML("beforeend", `<li class="msg note handoff"><div class="body">${icon("handoff")}<span><b>System note</b> · Handed off to a human agent${view.handoff?.ticket ? ` · ticket <span class="mono">${esc(view.handoff.ticket)}</span>` : ""}. This conversation is closed.</span></div></li>`);
  }
  if (busy) list.insertAdjacentHTML("beforeend", `<li class="msg note working"><div class="body">${icon("clock")}<span>Agent is working…</span></div></li>`);
  list.scrollTop = list.scrollHeight;
  $("#conv-h").textContent = view?.customer ? `Conversation with ${view.customer}` : "New conversation";
  $("#thread-id").innerHTML = thread ? `<span class="mono">${esc(thread)}</span> · ${items.length} messages` : "";
  $("#conv-status").innerHTML = thread ? statusBadge(view?.status) : "";
  const closed = view && ["handed_off", "awaiting_approval"].includes(view.status);
  $("#send").disabled = busy || closed;
  $("#input").disabled = closed;
  document.querySelectorAll(".thread").forEach((b) => b.setAttribute("aria-current", String(b.dataset.thread === thread)));
}

// ---------------------------------------------------------------- right panel
function renderFacts() {
  const writes = view?.executed?.length ?? 0;
  const blocks = view?.blocked?.length ?? 0;
  const c = view?.customer;
  $("#customer-card").innerHTML = `
    <h3>${icon("user")}Customer</h3>
    <div class="cust">
      <div class="av" aria-hidden="true">${icon("user")}</div>
      <div><div class="id">${esc(c ?? "Not verified yet")}</div>
      <div class="v">${c ? `<span class="badge ok">${icon("check")}Identity verified</span>` : `<span class="badge handed">${icon("clock")}Not verified</span>`}</div></div>
    </div>
    <dl class="facts">
      <div><dt>Steps</dt><dd>${view?.trace?.length ?? 0}</dd></div>
      <div><dt>Actions taken</dt><dd>${writes}</dd></div>
      <div><dt>Policy blocks</dt><dd class="${blocks ? "alert" : ""}">${blocks}</dd></div>
    </dl>`;
  renderOrders();
}

function renderOrders() {
  // Orders this conversation touched, from the agent's own tool calls, writes and approvals.
  const orders = new Map();
  const touch = (id) => { if (!orders.has(id)) orders.set(id, { id, notes: [], acts: new Map(), amount: null }); return orders.get(id); };
  (view?.trace ?? []).forEach((ev) => {
    const oid = ev.detail?.args?.order_id;
    if (ev.node === "agent" && ev.kind === "tool_call" && oid) {
      const o = touch(oid);
      if (ev.detail.tool === "get_order_details" && !o.notes.includes("Looked up")) o.notes.push("Looked up");
    }
  });
  approvalsCache.filter((a) => a.thread_id === thread).forEach((a) => {
    const o = touch(a.target);
    o.amount = a.amount;
    o.acts.set(a.tool, a.status === "pending" ? "waiting approval" : a.status);
  });
  (view?.executed ?? []).forEach((x) => {
    const o = touch(x.target);
    if (x.refund != null) o.amount = x.refund;
    o.acts.set(x.tool, o.acts.get(x.tool) === "approved" ? "approved, done" : "done");
  });
  const items = [...orders.values()];
  $("#orders-card").innerHTML = `<h3>${icon("box")}Orders in this conversation</h3>` + (items.length
    ? `<ul class="orders">${items.map((o) => `<li class="order">${icon("box")}<div class="oid">${esc(o.id)}${o.amount != null ? `<span>${money(o.amount)}</span>` : ""}</div><div class="what">${esc([...o.notes, ...[...o.acts].map(([t, s]) => `${toolLabel(t)}: ${s}`)].join(" · ") || "Mentioned")}</div></li>`).join("")}</ul>`
    : '<p class="empty">The agent hasn\'t opened an order yet.</p>');
}

function renderPending() {
  const mine = approvalsCache.filter((a) => a.thread_id === thread);
  const pending = mine.find((a) => a.status === "pending");
  const done = !pending && mine[0];
  const el = $("#pending-card");
  if (pending) {
    el.innerHTML = `<section class="approval" aria-labelledby="pend-h">
      <div class="head" id="pend-h">${icon("shield")}Refund waiting for your approval</div>
      <div class="amt">${money(pending.amount)}</div>
      <div class="desc">${esc(toolLabel(pending.tool))} <code class="mono">${esc(pending.target)}</code> · <span class="mono">${esc(pending.id)}</span></div>
      <div class="why">Refunds over the review threshold pause the agent until a person decides. The customer can't send messages meanwhile.</div>
      <div class="actions"><button class="btn approve" data-approve="${esc(pending.id)}">${icon("check")}Approve</button><button class="btn reject" data-reject="${esc(pending.id)}">${icon("x")}Reject</button></div>
    </section>`;
  } else if (done) {
    el.innerHTML = `<section class="approval done"><div class="head">${icon("shield")}Refund review</div>
      <div class="desc">${money(done.amount)} · ${esc(toolLabel(done.tool))} <code class="mono">${esc(done.target)}</code></div>
      <div>${done.status === "approved" ? `<span class="badge ok">${icon("check")}Approved</span>` : `<span class="badge bad">${icon("x")}Rejected</span>`} <span class="sub">by ${esc(done.decided_by ?? "")}</span></div></section>`;
  } else {
    el.innerHTML = "";
  }
}

function stepRow(ev) {
  const d = ev.detail || {};
  let cls = "", ic = "tool", what = "";
  switch (`${ev.node}.${ev.kind}`) {
    case "agent.tool_call": what = `Called <b>${esc(d.tool)}</b><code>${esc(compact(d.args))}</code>`; break;
    case "agent.reply": cls = "reply"; ic = "chat"; what = `Replied to customer${d.citations?.length ? ` · cites ${d.citations.map((c) => `<span class="cite">${esc(c)}</span>`).join(" ")}` : ""}${d.dropped_citations?.length ? ` · <b>dropped ${d.dropped_citations.length} unverified citation(s)</b>` : ""}`; break;
    case "guard.allowed": cls = "ok"; ic = "check"; what = `<b class="k">Allowed</b> ${esc(d.tool)}${d.audit_only_violations?.length ? ` · audit-only violation: ${esc(d.audit_only_violations.map((v) => v.rule).join(", "))}` : ""}`; break;
    case "guard.blocked": cls = "block"; ic = "x"; what = `<b class="k">Blocked</b> ${esc(d.tool)} · ${esc(d.why)}`; break;
    case "guard.invalid_args": cls = "block"; ic = "x"; what = `<b class="k">Invalid arguments</b> ${esc(d.error)}`; break;
    case "guard.needs_approval": cls = "wait"; ic = "clock"; what = `<b class="k">Needs approval</b> ${esc(d.tool)} · refund ${money(d.refund)} · <span class="mono">${esc(d.approval_id)}</span>`; break;
    case "approval.approved": cls = "ok"; ic = "check"; what = `<b class="k">Approved</b> by ${esc(d.by)} ${esc(d.note)}`; break;
    case "approval.rejected": cls = "block"; ic = "x"; what = `<b class="k">Rejected</b> by ${esc(d.by)} ${esc(d.note)}`; break;
    case "tools.result": cls = d.ok ? "ok" : "block"; ic = "db"; what = `<b>${esc(d.tool)}</b> <span class="k">${d.ok ? "ok" : "error"}</span>${d.replay ? " · idempotent replay" : ""}<code>${esc(d.preview)}</code>`; break;
    case "tools.kb_search": ic = "book"; what = `Help center: “${esc(d.query)}” → ${(d.hits || []).map(esc).join(", ") || "no hits"}`; break;
    case "handoff.ticket": cls = "wait"; ic = "handoff"; what = `<b class="k">Handed off</b> · ticket <span class="mono">${esc(d.ticket)}</span>`; break;
    default: what = esc(JSON.stringify(d));
  }
  return `<li class="step ${cls}"><span class="dot">${icon(ic)}</span><div class="what">${what}<div class="meta">${esc(ev.node)} · ${fmtTime(ev.ts)}${d.ms != null ? ` · ${d.ms} ms` : ""}</div></div></li>`;
}

function renderTrace() {
  const tr = view?.trace ?? [];
  $("#n-steps").textContent = tr.length ? `${tr.length} steps` : "";
  $("#tab-trace").innerHTML = tr.length ? tr.map(stepRow).join("") : '<li class="empty">Every model call, policy decision, tool call and approval shows up here, in order.</li>';
}

// ---------------------------------------------------------------- center views
async function renderApprovals() {
  const items = await api("/api/approvals?status=");
  approvalsCache = items;
  const pending = items.filter((a) => a.status === "pending");
  $("#n-appr").textContent = pending.length ? pending.length : "";
  renderPending();
  renderOrders();
  $("#tab-approvals").innerHTML = `<header class="view-head"><div><h2>Approvals</h2><p>Refunds above the review threshold wait here for a person · ${pending.length} pending</p></div></header>
    <div class="view-body">${items.length ? items.map((a) => `
    <article class="card">
      <div class="row"><span class="ttl">${esc(toolLabel(a.tool))} <span class="mono">${esc(a.target)}</span></span>${a.status === "pending" ? `<span class="badge waiting">${icon("clock")}Pending</span>` : a.status === "approved" ? `<span class="badge ok">${icon("check")}Approved${a.decided_by ? ` · ${esc(a.decided_by)}` : ""}</span>` : `<span class="badge bad">${icon("x")}Rejected${a.decided_by ? ` · ${esc(a.decided_by)}` : ""}</span>`}</div>
      <div class="row"><span class="amt">${money(a.amount)}</span><span class="sub"><span class="mono">${esc(a.id)}</span> · <button class="link" type="button" data-open="${esc(a.thread_id)}">open conversation</button></span></div>
      <div class="args">${esc(compact(a.args))}</div>
      ${a.status === "pending" ? `<div class="actions"><button class="btn approve sm" data-approve="${esc(a.id)}">${icon("check")}Approve refund</button><button class="btn reject sm" data-reject="${esc(a.id)}">${icon("x")}Reject</button></div>` : ""}
    </article>`).join("") : '<p class="empty">Nothing pending.</p>'}</div>`;
}

async function renderTickets() {
  const items = await api("/api/tickets");
  $("#n-tix").textContent = items.length ? items.length : "";
  $("#tab-tickets").innerHTML = `<header class="view-head"><div><h2>Handoffs</h2><p>When the agent can't help, or the customer asks for a person, it opens a ticket with a summary</p></div></header>
    <div class="view-body">${items.length ? items.map((t) => `
    <article class="card">
      <div class="row"><span class="ttl"><span class="mono">${esc(t.id)}</span> · ${esc(t.customer ?? "unverified")}</span><span class="badge handed">${icon("handoff")}${esc(t.status)}</span></div>
      <div>${esc(t.summary)}</div>
      <div class="sub">Actions taken: ${esc((t.context.actions_taken || []).map((a) => `${a.tool} ${a.target}`).join(", ") || "none")} · policy blocks: ${esc((t.context.policy_blocks || []).length)} · turns: ${esc(t.context.turns)} · ${fmtTime(t.created)} · <button class="link" type="button" data-open="${esc(t.thread_id)}">open conversation</button></div>
    </article>`).join("") : '<p class="empty">No handoffs yet.</p>'}</div>`;
}

function eventClass(e) {
  if (/blocked|denied|rejected|error|fail/.test(e)) return "bad";
  if (/approval_requested|handoff|replay/.test(e)) return "warn";
  if (/write_ok|granted|approved|verified/.test(e)) return "ok";
  return "";
}

async function renderAudit() {
  const rows = await api(`/api/audit?limit=80${thread ? `&thread_id=${encodeURIComponent(thread)}` : ""}`);
  $("#tab-audit").innerHTML = `<header class="view-head"><div><h2>Audit log</h2><p>${thread ? `Conversation <span class="mono">${esc(thread)}</span> · ` : ""}every read, write, block, approval and handoff · personal data redacted</p></div></header>
    <div class="view-body">${rows.length ? `<div class="tablewrap"><table class="audit"><thead><tr><th scope="col">Time</th><th scope="col">Actor</th><th scope="col">Event</th><th scope="col">Tool / target</th><th scope="col">Detail (PII redacted)</th></tr></thead><tbody>${rows.map((r) => `
    <tr><td class="m">${fmtTime(r.ts)}</td><td class="m">${esc(r.actor)}</td><td><span class="ev ${eventClass(r.event)}">${esc(r.event)}</span></td><td class="tt"><code>${esc(r.tool ?? "")}</code> ${r.target ? `<code>${esc(r.target)}</code>` : ""}</td><td class="d">${esc(compact(r.detail))}</td></tr>`).join("")}</tbody></table></div>` : '<p class="empty">Nothing recorded for this conversation yet.</p>'}</div>`;
}

async function refreshConsole() {
  renderFacts();
  renderTrace();
  await Promise.all([renderApprovals(), renderTickets(), renderAudit(), loadThreads()]).catch(() => {});
}

function selectTab(name) {
  tab = name;
  store.set("sd.tab", name);
  document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
  $("#tab-inbox").hidden = name !== "trace";
  ["approvals", "tickets", "audit"].forEach((t) => { $(`#tab-${t}`).hidden = t !== name; });
}

// ---------------------------------------------------------------- actions
async function newThread() {
  const r = await api("/api/threads", { method: "POST" });
  thread = r.thread_id;
  store.set("sd.thread", thread);
  view = { transcript: [], trace: [], executed: [], blocked: [], status: "idle" };
  selectTab("trace");
  renderChat();
  await refreshConsole();
  $("#input").focus();
}

async function loadThread() {
  if (!thread) return newThread();
  try { view = await api(`/api/threads/${thread}`); } catch { return newThread(); }
  renderChat();
  await refreshConsole();
}

async function openThread(id) {
  thread = id;
  store.set("sd.thread", id);
  if (tab !== "audit") selectTab("trace");
  await loadThread();
}

async function send(text) {
  busy = true;
  view.transcript = [...(view.transcript || []), { role: "customer", text }];
  renderChat();
  try {
    view = await api(`/api/threads/${thread}/messages`, { method: "POST", body: JSON.stringify({ text }) });
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
  document.querySelectorAll(`[data-approve="${id}"], [data-reject="${id}"]`).forEach((b) => { b.disabled = true; });
  const r = await api(`/api/approvals/${id}`, { method: "POST", body: JSON.stringify({ approved, note, by: "console" }) });
  if (r.thread_id === thread) view = r;
  renderChat();
  selectTab("trace");
  await refreshConsole();
}

async function loadMeta() {
  try {
    const h = await api("/health");
    const pct = Math.min(100, (h.spend_usd / h.spend_cap_usd) * 100);
    $("#meta").innerHTML = `
      <div class="row"><span>Model <b>${esc(h.model)}</b></span><span>Policy <b>${h.guard_mode === "enforce" ? "code-enforced" : "audit only"}</b></span></div>
      <div class="row"><span>API spend</span><span><b>$${h.spend_usd.toFixed(2)}</b> of $${h.spend_cap_usd.toFixed(2)} cap</span></div>
      <div class="meter" role="meter" aria-label="API spend against cap" aria-valuemin="0" aria-valuemax="${h.spend_cap_usd}" aria-valuenow="${h.spend_usd.toFixed(2)}"><i style="width:${pct.toFixed(1)}%"></i></div>`;
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
  const t = e.target.closest("[data-thread]"); if (t && t.dataset.thread !== thread) openThread(t.dataset.thread);
  const o = e.target.closest("[data-open]"); if (o) openThread(o.dataset.open).then(() => selectTab("trace"));
});

const params = new URLSearchParams(location.search);
thread = params.get("thread") || store.get("sd.thread");
if (params.get("tab")) tab = params.get("tab");
selectTab(tab);
loadMeta();
loadExamples();
loadThread();
setInterval(() => { if (!busy && thread) loadThread(); }, 5000);
