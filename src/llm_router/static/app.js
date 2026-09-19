"use strict";

// llm-router session dashboard.
//
// Polls /sessions every POLL_MS while the tab is visible, and nothing while it is
// hidden. Between polls, durations tick forward locally so waits read as live
// without asking the router for anything more.
//
// Every value shown comes from client-supplied headers, so everything goes into
// the page as text (textContent / setAttribute), never as markup.

(() => {
  const POLL_MS = 2000;
  const RETRY_MS = 5000;
  const TICK_MS = 1000;
  const SVG = "http://www.w3.org/2000/svg";

  const ui = {
    data: null,
    fetchedAt: 0,
    lastOk: 0,
    paused: false,
    timer: null,
    user: null,          // user key being filtered on
    userName: null,
    expanded: new Map(), // session n -> detail (or null while loading)
    search: "",
    sort: "urgent",
    errorsOnly: false,
  };

  const $ = (id) => document.getElementById(id);

  // ---------------------------------------------------------------- building

  function el(tag, props, ...kids) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(props || {})) {
      if (value == null || value === false) continue;
      if (key === "class") node.className = value;
      else if (key === "on") for (const [ev, fn] of Object.entries(value)) node.addEventListener(ev, fn);
      else if (key === "data") Object.assign(node.dataset, value);
      else node.setAttribute(key, value === true ? "" : String(value));
    }
    for (const kid of kids.flat()) {
      if (kid == null || kid === false) continue;
      node.append(kid instanceof Node ? kid : String(kid));
    }
    return node;
  }

  const ICONS = {
    slow: ["M8 1.75a6.25 6.25 0 1 0 0 12.5 6.25 6.25 0 0 0 0-12.5z", "M8 4.75V8l2.25 1.5"],
    stuck: ["M8 1.75a6.25 6.25 0 1 0 0 12.5 6.25 6.25 0 0 0 0-12.5z", "M8 4.75v3.75", "M8 11.1v.15"],
    error: ["M8 1.75a6.25 6.25 0 1 0 0 12.5 6.25 6.25 0 0 0 0-12.5z", "M5.9 5.9l4.2 4.2", "M10.1 5.9l-4.2 4.2"],
    ok: ["M3.25 8.4l3 3 6.5-6.8"],
    cancelled: ["M8 1.75a6.25 6.25 0 1 0 0 12.5 6.25 6.25 0 0 0 0-12.5z", "M3.6 3.6l8.8 8.8"],
    chevron: ["M6 3.5 10.5 8 6 12.5"],
  };

  function icon(name) {
    const svg = document.createElementNS(SVG, "svg");
    svg.setAttribute("viewBox", "0 0 16 16");
    svg.setAttribute("aria-hidden", "true");
    svg.setAttribute("class", `icon ${name}`);
    for (const d of ICONS[name]) {
      const path = document.createElementNS(SVG, "path");
      path.setAttribute("d", d);
      svg.append(path);
    }
    return svg;
  }

  // -------------------------------------------------------------- formatting

  function dur(s) {
    if (s == null || !Number.isFinite(s)) return "—";
    if (s < 10) return `${Math.max(0, s).toFixed(1)}s`;
    s = Math.floor(s);
    if (s < 60) return `${s}s`;
    const m = Math.floor(s / 60);
    if (m < 60) return `${m}m ${String(s % 60).padStart(2, "0")}s`;
    const h = Math.floor(m / 60);
    if (h < 24) return `${h}h ${String(m % 60).padStart(2, "0")}m`;
    return `${Math.floor(h / 24)}d ${h % 24}h`;
  }

  // A configured threshold, in the units it was probably written in.
  function limit(s) {
    if (s == null) return "—";
    if (s < 60 || s % 60) return s < 60 ? `${s}s` : dur(s);
    return s % 3600 ? `${s / 60}m` : `${s / 3600}h`;
  }

  function compact(n) {
    if (n == null) return "—";
    if (n < 10000) return n.toLocaleString();
    if (n < 1e6) return `${(n / 1e3).toFixed(n < 1e5 ? 1 : 0)}K`;
    return `${(n / 1e6).toFixed(n < 1e7 ? 1 : 0)}M`;
  }

  function size(n) {
    if (!n) return "—";
    if (n < 1024) return `${n} B`;
    if (n < 1048576) return `${(n / 1024).toFixed(1)} KB`;
    return `${(n / 1048576).toFixed(1)} MB`;
  }

  function pct(r) {
    return r == null ? "—" : `${Math.round(r * 100)}%`;
  }

  function tokens(inTok, outTok) {
    if (!inTok && !outTok) return "—";
    return `${compact(inTok || 0)} / ${compact(outTok || 0)}`;
  }

  // A duration that keeps counting between polls.
  function tick(seconds, suffix = "", cls = "") {
    if (seconds == null) return el("span", { class: "dim" }, "—");
    return el("span", { class: `tick ${cls}`.trim(), data: { base: seconds, suffix } }, dur(seconds) + suffix);
  }

  // What a request asked the model for. "default" means it said nothing, so the
  // backend's own settings apply.
  function asked(r) {
    const bits = [];
    if (r.effort) bits.push(`effort ${r.effort}`);
    if (r.thinking === true) bits.push(r.budget ? `thinking ${compact(r.budget)} budget` : "thinking");
    else if (r.thinking === false) bits.push("no thinking");
    if (r.max_tokens) bits.push(`${compact(r.max_tokens)} out`);
    if (!bits.length) return el("span", { class: "dim" }, "default");
    const text = bits.join(" · ");
    return el("span", { title: `asked for: ${text}` }, text);
  }

  // Time holding a backend, and how much of it others spent queued behind.
  function slot(r, live) {
    const held = live ? tick(r.slot_s) : dur(r.slot_s);
    if (!r.contended_s) return held;
    return el("span", { class: "stack" }, held,
      el("span", { class: "sub", title: "of that, time other requests were queued behind it" },
        `${dur(r.contended_s)} with a queue`));
  }

  function severity(silence) {
    const t = ui.data && ui.data.thresholds;
    if (!t || silence == null) return null;
    if (silence >= t.stuck_s) return "stuck";
    if (silence >= t.slow_s) return "slow";
    return null;
  }

  function severityBadge(sev) {
    if (!sev) return null;
    return el("span", { class: `badge ${sev}` }, icon(sev), sev);
  }

  const STATE_LABEL = {
    queued: "queued", holding: "holding", processing: "processing",
    streaming: "streaming", idle: "idle",
  };
  const STATE_DOT = {
    queued: "waiting", holding: "waiting", processing: "processing",
    streaming: "streaming", idle: "idle",
  };

  function stateTip(r) {
    switch (r.state) {
      case "holding":
        return `Waiting for its pinned backend${r.pinned ? ` ${r.pinned}` : ""} to free a slot, to reuse the prompt cache there`;
      case "queued":
        return "Waiting for a free slot on any backend";
      case "processing":
        return r.stream
          ? `Sent to ${r.backend}; no tokens back yet (prefill)`
          : `Sent to ${r.backend}; waiting for the whole reply (not streamed)`;
      case "streaming":
        return `Receiving tokens from ${r.backend}`;
      default:
        return "Nothing in flight: waiting on the client";
    }
  }

  function stateChip(r) {
    const label = STATE_LABEL[r.state] || r.state;
    return el("span", { class: "state", title: stateTip(r) },
      el("span", { class: `dot ${STATE_DOT[r.state] || "idle"}` }), label);
  }

  const RESULT_ICON = { ok: "ok", error: "error", cancelled: "cancelled" };

  function result(r) {
    if (!r.state || !(r.state in RESULT_ICON)) return el("span", { class: "dim" }, "—");
    const why = r.state === "cancelled"
      ? r.note || "client left"
      : r.state === "error"
        ? [r.status, r.note].filter(Boolean).join(" ")
        : r.status && r.status !== 200 ? String(r.status) : "";
    return el("span", { class: "result" }, icon(RESULT_ICON[r.state]), r.state,
      why ? el("span", { class: "why" }, why) : null);
  }

  const VIA_TIP = {
    "claude-code": "Claude Code's own session id (x-claude-code-session-id)",
    "open-webui": "Open WebUI chat id (X-OpenWebUI-Chat-Id)",
    header: "Session id sent by the client",
    inferred: "No session id was sent, so this is inferred from the start of the prompt. A conversation whose opening is edited, or compacted, shows up as a new session.",
    none: "The request carried no messages to identify it by",
  };

  function sessionCell(s, n, via, agents) {
    const shown = s ? s.slice(0, 8) : "unidentified";
    const tip = `${s || "no id"}\n${VIA_TIP[via] || ""}`;
    const button = el("button", {
      type: "button", class: "link", title: tip,
      on: { click: () => openSession(n) },
    }, el("span", { class: "mono primary" }, shown));
    const extras = [];
    if (via === "inferred") extras.push(el("span", { class: "badge soft", title: VIA_TIP.inferred }, "inferred"));
    if (agents) extras.push(el("span", { class: "sub" }, `${agents} subagent${agents === 1 ? "" : "s"}`));
    return el("span", { class: "state" }, button, ...extras);
  }

  function userButton(key, name, sub) {
    return el("button", {
      type: "button", class: "link", title: `Show only ${name}`,
      on: { click: () => setUser(key, name) },
    }, el("span", { class: "stack" }, el("span", { class: "primary" }, name),
      sub ? el("span", { class: "sub" }, sub) : null));
  }

  function empty(cols, text) {
    return el("tr", {}, el("td", { class: "empty", colspan: cols }, text));
  }

  function fill(table, rows) {
    const body = $(table).tBodies[0];
    body.replaceChildren(...rows);
  }

  // --------------------------------------------------------------- rendering

  function render() {
    const d = ui.data;
    if (!d) return;
    const focusKey = document.activeElement && document.activeElement.dataset
      ? document.activeElement.dataset.focus : null;

    renderMeta(d);
    renderBanner(d);
    renderTiles(d);
    renderBackends(d);
    renderFilter();
    renderInflight(d);
    renderUsers(d);
    renderSessions(d);
    renderRecent(d);

    if (focusKey) {
      const again = document.querySelector(`[data-focus="${CSS.escape(focusKey)}"]`);
      if (again) again.focus();
    }
  }

  function renderMeta(d) {
    const r = d.router || {};
    const t = d.thresholds || {};
    $("meta").textContent = [
      r.uptime_s != null ? `up ${dur(r.uptime_s)}` : null,
      r.config_generation ? `config generation ${r.config_generation}` : null,
      t.slow_s ? `slow ≥ ${limit(t.slow_s)} · stuck ≥ ${limit(t.stuck_s)} silent` : null,
    ].filter(Boolean).join(" · ");
  }

  function renderBanner(d) {
    const banner = $("banner");
    const error = d.router && d.router.config_error;
    if (!d.enabled) {
      banner.className = "banner info";
      banner.replaceChildren("Session tracking is switched off (tracking.enabled: false). Nothing is being recorded.");
      banner.hidden = false;
    } else if (error) {
      banner.className = "banner";
      banner.replaceChildren(icon("error"),
        el("span", {}, `Config reload rejected; still running generation ${d.router.config_generation}. `, el("code", {}, error)));
      banner.hidden = false;
    } else {
      banner.hidden = true;
    }
  }

  function tile(label, value, sub, cls, iconName) {
    return el("div", { class: `tile ${cls || ""}`.trim() },
      el("div", { class: "label" }, iconName ? icon(iconName) : null, label),
      el("div", { class: "value" }, value),
      sub ? el("div", { class: "sub" }, sub) : null);
  }

  function renderTiles(d) {
    const s = d.summary || {};
    const b = s.by_state || {};
    const r = d.router || {};
    const waiting = (b.queued || 0) + (b.holding || 0);
    const breakdown = [
      b.streaming ? `${b.streaming} streaming` : null,
      b.processing ? `${b.processing} processing` : null,
      waiting ? `${waiting} waiting for a slot` : null,
    ].filter(Boolean).join(" · ") || "nothing running";

    const longest = (d.in_flight || [])[0];
    const users = d.users || [];
    const busyUsers = users.filter((u) => u.in_flight).length;

    $("tiles").replaceChildren(
      tile("In flight", String(s.in_flight || 0), breakdown),
      tile("Stuck", String(s.stuck || 0),
        s.slow ? `+${s.slow} slow` : `silent ≥ ${limit((d.thresholds || {}).stuck_s)}`,
        s.stuck ? "stuck" : s.slow ? "slow" : "", s.stuck ? "stuck" : s.slow ? "slow" : null),
      tile("Longest wait", longest ? tick(longest.waiting_s) : "—",
        longest ? `${longest.user_name} · ${STATE_LABEL[longest.state] || longest.state}` : "no requests waiting"),
      tile("Users", String(users.length), `${busyUsers} with requests in flight`),
      tile("Sessions", String(s.sessions || 0), `${s.busy_sessions || 0} busy`),
      tile("Queue", String(r.queue_depth || 0),
        r.waiting_on_affinity ? `${r.waiting_on_affinity} holding for a pinned host` : "requests waiting for a slot"),
      holdingUp(d),
    );
  }

  // Whoever is keeping the queue waiting longest right now.
  function holdingUp(d) {
    const blocking = (d.in_flight || []).filter((r) => r.blocking);
    if (!blocking.length) return tile("Holding up the queue", "—", "nothing is queued behind a busy backend");
    const worst = blocking.reduce((a, b) => (b.slot_s > a.slot_s ? b : a));
    const bits = [worst.effort ? `effort ${worst.effort}` : null,
      `held ${dur(worst.slot_s)}`, `${worst.blocking} waiting`].filter(Boolean);
    return tile("Holding up the queue", worst.user_name, bits.join(" · "));
  }

  function renderBackends(d) {
    const cards = (d.backends || []).map((b) => {
      const status = b.draining ? "draining" : !b.healthy ? "down" : b.cooling_down ? "cooling down" : "";
      const full = b.inflight >= b.capacity && b.capacity > 0;
      const width = b.capacity ? Math.min(100, (100 * b.inflight) / b.capacity) : 0;
      const bad = status === "down" || status === "cooling down";
      // Through the CSSOM: the page's CSP refuses inline style attributes.
      const bar = el("div", { class: "fill" });
      bar.style.width = `${width}%`;
      return el("div", { class: "backend" },
        el("div", { class: "row" },
          bad ? icon("error") : null,
          el("span", { class: "bname", title: b.name }, b.name),
          el("span", { class: `bstate ${bad ? "bad" : ""}`.trim() },
            status || (b.waiting ? `${b.waiting} queued` : full ? "full" : ""))),
        el("div", {
          class: `meter ${full ? "full" : ""} ${bad || b.draining ? "off" : ""}`.trim(),
          role: "meter", "aria-valuemin": 0, "aria-valuemax": b.capacity, "aria-valuenow": b.inflight,
          "aria-label": `${b.name}: ${b.inflight} of ${b.capacity} slots in use`,
        },
        el("div", { class: "track" }, bar),
        `${b.inflight} / ${b.capacity}`));
    });
    $("backends").replaceChildren(...cards);
  }

  function renderFilter() {
    $("filter").hidden = !ui.user;
    $("filter-name").textContent = ui.userName || "";
  }

  function mine(item) {
    return !ui.user || item.user === ui.user;
  }

  function renderInflight(d) {
    const rows = (d.in_flight || []).filter(mine).map((r) => {
      const sev = severity(r.silence_s);
      const backend = r.backend
        ? r.backend
        : r.pinned ? el("span", { class: "dim", title: "pinned backend it is holding for" }, `→ ${r.pinned}`) : el("span", { class: "dim" }, "—");
      return el("tr", { class: `row ${sev || ""}`.trim() },
        el("td", {}, stateChip(r), severityBadge(sev),
          r.attempts > 1 ? el("span", { class: "sub", title: "attempts, after failing over" }, ` try ${r.attempts}`) : null),
        el("td", { class: "num" }, tick(r.waiting_s)),
        el("td", { class: "num" }, tick(r.silence_s, "", `silence ${sev || ""}`)),
        el("td", {}, userButton(r.user, r.user_name, r.client)),
        el("td", {}, el("span", { class: "stack" },
          sessionCell(r.session_id, r.session, r.session_via, 0),
          r.agent ? el("span", { class: "sub mono", title: r.agent }, `agent ${r.agent.slice(0, 8)}`) : null)),
        el("td", {}, r.model),
        el("td", {}, asked(r)),
        el("td", {}, backend,
          r.blocking
            ? el("span", { class: "badge soft", title: `${r.blocking} request(s) queued for this backend` },
              `${r.blocking} waiting`)
            : null),
        el("td", { class: "num" }, slot(r, true)),
        el("td", { class: "num" }, size(r.bytes)));
    });
    fill("inflight", rows.length ? rows : [empty(10, ui.user ? "Nothing in flight for this user." : "Nothing in flight.")]);
  }

  function renderUsers(d) {
    const rows = (d.users || []).map((u) => {
      const sev = u.stuck ? "stuck" : u.slow ? "slow" : null;
      const how = u.via === "address" ? "by address"
        : u.via === "open-webui" ? `Open WebUI${u.addresses.length ? ` · ${u.addresses.join(", ")}` : ""}`
          : `named by header${u.addresses.length ? ` · ${u.addresses.join(", ")}` : ""}`;
      return el("tr", { class: `row ${sev || ""}`.trim() },
        el("td", {}, userButton(u.user, u.name, how)),
        el("td", {}, u.clients.join(", ") || "—"),
        el("td", { class: "num" }, u.busy_sessions ? `${u.busy_sessions} busy / ${u.sessions}` : String(u.sessions)),
        el("td", { class: "num" }, String(u.in_flight || 0),
          u.stuck ? severityBadge("stuck") : u.slow ? severityBadge("slow") : null),
        el("td", { class: "num" }, u.longest_wait_s != null ? tick(u.longest_wait_s) : el("span", { class: "dim" }, "—")),
        el("td", { class: "num" }, compact(u.requests)),
        el("td", { class: "num" }, u.errors ? String(u.errors) : el("span", { class: "dim" }, "0")),
        el("td", {}, asked({
          effort: u.efforts.join(" / ") || null, thinking: u.thinking, budget: u.budget,
        })),
        el("td", { class: "num" }, dur(u.slot_s)),
        el("td", { class: "num", title: "time this user's requests kept others queued" },
          u.contended_s ? dur(u.contended_s) : el("span", { class: "dim" }, "—")),
        el("td", { class: "num", title: "time this user's own requests spent queued" },
          u.queued_s ? dur(u.queued_s) : el("span", { class: "dim" }, "—")),
        el("td", { class: "num" }, tokens(u.prompt_tokens, u.completion_tokens)),
        el("td", { class: "num" }, u.in_flight ? "now" : tick(u.idle_s, " ago")));
    });
    fill("users", rows.length ? rows : [empty(13, "No users yet. They appear here with their first request.")]);
  }

  function matches(s) {
    if (!ui.search) return true;
    const hay = [s.user_name, s.id, s.client, s.backend, s.address, ...(s.models || [])]
      .join(" ").toLowerCase();
    return hay.includes(ui.search);
  }

  const SESSION_ORDER = { stuck: 0, slow: 1 };

  function renderSessions(d) {
    const list = (d.sessions || []).filter((s) => mine(s) && matches(s));
    const by = {
      contended: (s) => s.contended_s,
      slot: (s) => s.slot_s,
      tokens: (s) => s.prompt_tokens + s.completion_tokens,
    }[ui.sort];
    if (by) {
      list.sort((a, b) => by(b) - by(a));
    } else {
      // Worst first: stuck, slow, other busy by wait, then idle by recency.
      list.sort((a, b) => {
        const sa = SESSION_ORDER[severity(a.silence_s)] ?? (a.in_flight ? 2 : 3);
        const sb = SESSION_ORDER[severity(b.silence_s)] ?? (b.in_flight ? 2 : 3);
        if (sa !== sb) return sa - sb;
        if (a.in_flight) return (b.waiting_s || 0) - (a.waiting_s || 0);
        return a.idle_s - b.idle_s;
      });
    }

    const rows = [];
    for (const s of list) {
      const sev = s.in_flight ? severity(s.silence_s) : null;
      const open = ui.expanded.has(s.n);
      const lastRequest = { state: s.last_outcome, status: s.last_status };
      rows.push(el("tr", { class: `row ${sev || ""}`.trim() },
        el("td", {}, el("button", {
          type: "button", class: "expander", "aria-expanded": open ? "true" : "false",
          "aria-label": `${open ? "Hide" : "Show"} requests for session ${s.id.slice(0, 8) || s.n}`,
          data: { focus: `expand-${s.n}` },
          on: { click: () => toggle(s.n) },
        }, icon("chevron"))),
        el("td", {}, stateChip(s), severityBadge(sev),
          s.in_flight > 1 ? el("span", { class: "sub" }, ` ×${s.in_flight}`) : null),
        el("td", { class: "num" }, s.in_flight
          ? tick(s.waiting_s, "", `silence ${sev || ""}`)
          : tick(s.idle_s, " idle", "dim")),
        el("td", {}, userButton(s.user, s.user_name, null)),
        el("td", {}, sessionCell(s.id, s.n, s.via, s.agents)),
        el("td", {}, s.client || "—"),
        el("td", { title: s.models.join(", ") }, el("span", { class: "stack" },
          el("span", {}, s.models[s.models.length - 1] || "—",
            s.models.length > 1 ? el("span", { class: "sub" }, ` +${s.models.length - 1}`) : null),
          el("span", { class: "sub" }, asked(s)))),
        el("td", {}, s.backend || el("span", { class: "dim" }, "—")),
        el("td", { class: "num" }, compact(s.requests)),
        el("td", { class: "num" }, s.errors ? String(s.errors) : el("span", { class: "dim" }, "0")),
        el("td", { class: "num" }, slot(s, false)),
        el("td", { class: "num" }, tokens(s.prompt_tokens, s.completion_tokens)),
        el("td", { class: "num", title: "Share of prompt tokens served from the backend's prefix cache" }, pct(s.cache_rate)),
        el("td", {}, result(lastRequest))));
      if (open) rows.push(detailRow(s.n, ui.expanded.get(s.n)));
    }
    fill("sessions", rows.length ? rows : [empty(14,
      ui.search || ui.user ? "No sessions match." : "No sessions yet.")]);

    const total = (d.summary || {}).sessions || 0;
    const shown = (d.sessions || []).length;
    $("sessions-note").textContent = total > shown
      ? `Showing the ${shown} most recently active of ${total} sessions. Busy sessions are always shown.`
      : "";
  }

  function requestRows(requests) {
    return requests.map((r) => {
      const live = !(r.state in RESULT_ICON);
      const sev = live ? severity(r.silence_s) : null;
      return el("tr", { class: `row ${sev || ""}`.trim() },
        el("td", { class: "num" }, live ? "now" : tick(r.ended_s, " ago")),
        el("td", {}, live ? [stateChip(r), severityBadge(sev)] : result(r)),
        el("td", { class: "mono", title: r.agent || "main conversation" }, r.agent ? r.agent.slice(0, 8) : "main"),
        el("td", {}, r.model),
        el("td", {}, r.backend || el("span", { class: "dim" }, "—")),
        el("td", {}, asked(r)),
        el("td", { class: "num" }, dur(r.queue_s)),
        el("td", { class: "num" }, dur(r.ttft_s)),
        el("td", { class: "num" }, slot(r, live)),
        el("td", { class: "num" }, live ? tick(r.waiting_s) : dur(r.duration_s)),
        el("td", { class: "num" }, tokens(r.prompt_tokens, r.completion_tokens)),
        el("td", { class: "num" }, r.cached_tokens != null && r.prompt_tokens
          ? pct(r.cached_tokens / r.prompt_tokens) : el("span", { class: "dim" }, "—")));
    });
  }

  function detailRow(n, detail) {
    if (!detail) {
      return el("tr", { class: "detail" }, el("td", { colspan: 14 }, el("span", { class: "dim" }, "Loading…")));
    }
    const facts = [
      ["Session id", detail.id || "none", "mono"],
      ["Identified by", VIA_TIP[detail.via] || detail.via],
      ["User", detail.user_name],
      ["Address", detail.address || "—"],
      ["User agent", detail.user_agent || "—", "mono"],
      ["First seen", `${dur(detail.first_seen_s)} ago`],
    ];
    if (detail.agent_ids && detail.agent_ids.length) {
      facts.push(["Subagents", detail.agent_ids.map((a) => a.slice(0, 8)).join(", "), "mono"]);
    }
    const requests = [...detail.active, ...detail.recent];
    const head = ["Ended", "Result", "Agent", "Model", "Backend", "Asked for", "Queued",
      "First byte", "Holding slot", "Total", "Tokens in / out", "Cache"];
    const numeric = new Set(["Ended", "Queued", "First byte", "Holding slot", "Total",
      "Tokens in / out", "Cache"]);
    return el("tr", { class: "detail" }, el("td", { colspan: 14 },
      el("dl", { class: "facts" }, facts.map(([k, v, cls]) =>
        el("div", {}, el("dt", {}, k), el("dd", { class: cls || null }, v)))),
      el("div", { class: "scroll" }, el("table", {},
        el("thead", {}, el("tr", {}, head.map((h) => el("th", { scope: "col", class: numeric.has(h) ? "num" : null }, h)))),
        el("tbody", {}, requests.length ? requestRows(requests) : empty(12, "No requests remembered for this session."))))));
  }

  function renderRecent(d) {
    const rows = (d.recent || [])
      .filter((r) => mine(r) && (!ui.errorsOnly || r.state === "error"))
      .map((r) => el("tr", { class: "row" },
        el("td", { class: "num" }, tick(r.ended_s, " ago")),
        el("td", {}, userButton(r.user, r.user_name, null)),
        el("td", {}, sessionCell(r.session_id, r.session, r.session_via, 0)),
        el("td", {}, r.model),
        el("td", {}, r.backend || el("span", { class: "dim" }, "—")),
        el("td", {}, result(r)),
        el("td", {}, asked(r)),
        el("td", { class: "num" }, dur(r.queue_s)),
        el("td", { class: "num" }, dur(r.ttft_s)),
        el("td", { class: "num" }, dur(r.duration_s)),
        el("td", { class: "num" }, tokens(r.prompt_tokens, r.completion_tokens))));
    fill("recent", rows.length ? rows : [empty(11, ui.errorsOnly ? "No recent errors." : "No finished requests yet.")]);
  }

  // ------------------------------------------------------------ interaction

  function setUser(key, name) {
    ui.user = key;
    ui.userName = name;
    history.replaceState(null, "", key ? `#user=${encodeURIComponent(key)}` : location.pathname);
    render();
  }

  function toggle(n) {
    if (ui.expanded.has(n)) {
      ui.expanded.delete(n);
      render();
    } else {
      ui.expanded.set(n, null);
      render();
      loadDetail(n).then(render);
    }
  }

  function openSession(n) {
    if (!ui.expanded.has(n)) toggle(n);
    const button = document.querySelector(`[data-focus="expand-${n}"]`);
    if (button) {
      button.scrollIntoView({ block: "center", behavior: "smooth" });
      button.focus({ preventScroll: true });
    }
  }

  async function loadDetail(n) {
    try {
      const response = await fetch(`sessions/${n}`, { cache: "no-store" });
      if (response.status === 404) {
        ui.expanded.delete(n); // forgotten by the router
        return;
      }
      if (response.ok && ui.expanded.has(n)) ui.expanded.set(n, await response.json());
    } catch {
      // Keep what we had; the next poll retries.
    }
  }

  // ---------------------------------------------------------------- polling

  function setConn(kind, text) {
    $("pulse").className = `pulse ${kind}`;
    $("conn").textContent = text;
  }

  function schedule(ms) {
    clearTimeout(ui.timer);
    ui.timer = setTimeout(poll, ms);
  }

  async function poll() {
    clearTimeout(ui.timer);
    if (ui.paused || document.hidden) return;
    try {
      const response = await fetch("sessions", { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      await Promise.all([...ui.expanded.keys()].map(loadDetail));
      ui.data = data;
      ui.fetchedAt = performance.now();
      ui.lastOk = Date.now();
      setConn("live", "live");
      render();
      schedule(POLL_MS);
    } catch (err) {
      setConn("down", `can't reach the router (${err.message || err}); retrying`);
      schedule(RETRY_MS);
    }
  }

  function tickAll() {
    if (!ui.data || ui.paused || !ui.lastOk || Date.now() - ui.lastOk > POLL_MS * 3) return;
    const elapsed = (performance.now() - ui.fetchedAt) / 1000;
    for (const node of document.querySelectorAll(".tick")) {
      const base = Number(node.dataset.base);
      node.textContent = dur(base + elapsed) + (node.dataset.suffix || "");
    }
  }

  // ------------------------------------------------------------------ wiring

  $("pause").addEventListener("click", () => {
    ui.paused = !ui.paused;
    $("pause").setAttribute("aria-pressed", String(ui.paused));
    $("pause").textContent = ui.paused ? "Resume" : "Pause";
    if (ui.paused) {
      clearTimeout(ui.timer);
      setConn("", "paused");
    } else {
      poll();
    }
  });

  $("filter-clear").addEventListener("click", () => setUser(null, null));

  $("search").addEventListener("input", (event) => {
    ui.search = event.target.value.trim().toLowerCase();
    render();
  });

  $("sort").addEventListener("change", (event) => {
    ui.sort = event.target.value;
    render();
  });

  $("errors-only").addEventListener("change", (event) => {
    ui.errorsOnly = event.target.checked;
    render();
  });

  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && !ui.paused) poll();
  });

  const fromHash = /^#user=(.+)$/.exec(location.hash);
  if (fromHash) {
    ui.user = decodeURIComponent(fromHash[1]);
    ui.userName = ui.user.replace(/^(u|ip):/, "");
  }

  setInterval(tickAll, TICK_MS);
  poll();
})();
