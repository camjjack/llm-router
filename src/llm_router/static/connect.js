"use strict";

// llm-router connect page: shows the client configs /clients generates.
//
// Model names and ids come from the router's config and user names from this
// page's own input, so everything goes into the page as text, never as markup.

(() => {
  const state = { data: null, client: null, timer: null };
  const $ = (id) => document.getElementById(id);

  function el(tag, props, ...kids) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(props || {})) {
      if (value == null || value === false) continue;
      if (key === "class") node.className = value;
      else if (key === "on") for (const [ev, fn] of Object.entries(value)) node.addEventListener(ev, fn);
      else node.setAttribute(key, value === true ? "" : String(value));
    }
    for (const kid of kids.flat()) {
      if (kid == null || kid === false) continue;
      node.append(kid instanceof Node ? kid : String(kid));
    }
    return node;
  }

  function query() {
    const params = new URLSearchParams();
    const user = $("user").value.trim();
    const model = $("model").value;
    if (user) params.set("user", user);
    if (model && state.data && model !== state.data.default_model_configured) params.set("model", model);
    const text = params.toString();
    return text ? `?${text}` : "";
  }

  // The Clipboard API needs a secure context, which a router on plain http://
  // isn't, so fall back to a selection copy there.
  async function copy(text, button) {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      const area = el("textarea", { class: "offscreen", readonly: true });
      area.value = text;
      document.body.append(area);
      area.select();
      document.execCommand("copy");
      area.remove();
    }
    const label = button.textContent;
    button.textContent = "Copied";
    setTimeout(() => { button.textContent = label; }, 1400);
  }

  function codeBlock(text, label) {
    const button = el("button", { type: "button", class: "btn small copy", "aria-label": `Copy ${label}` }, "Copy");
    button.addEventListener("click", () => copy(text, button));
    return el("div", { class: "code" }, el("pre", {}, el("code", {}, text)), button);
  }

  function renderTabs() {
    const tabs = state.data.clients.map((c) => el("button", {
      type: "button", role: "tab", class: "tab", id: `tab-${c.id}`,
      "aria-selected": c.id === state.client ? "true" : "false",
      "aria-controls": "client",
      on: { click: () => select(c.id) },
    }, c.label));
    $("tabs").replaceChildren(...tabs);
  }

  function renderClient() {
    const c = state.data.clients.find((x) => x.id === state.client) || state.data.clients[0];
    $("client").setAttribute("aria-labelledby", `tab-${c.id}`);
    const steps = c.steps.map((step) => el("li", {},
      el("p", {}, step.text),
      step.command ? codeBlock(step.command, "command") : null));
    const download = el("a", { class: "btn small", href: c.url, download: c.download }, "Download");
    $("client").replaceChildren(
      el("ol", { class: "steps" }, steps),
      el("div", { class: "file-head" },
        el("span", {}, "Goes in ", el("code", {}, c.file)), download),
      codeBlock(c.content, `${c.label} config`),
      c.notes.length ? el("ul", { class: "notes" }, c.notes.map((n) => el("li", {}, n))) : null,
    );
  }

  function flag(value) {
    return value ? "yes" : el("span", { class: "dim" }, "no");
  }

  function renderModels() {
    const rows = state.data.models.map((m) => el("tr", { class: "row" },
      el("td", { class: "mono" }, m.id),
      el("td", {}, m.name),
      el("td", { class: "num", title: `context: ${m.context_from}` },
        m.context != null ? m.context.toLocaleString() : el("span", { class: "dim" }, "unknown"),
        m.context_from === "configured" || m.context_from === "capped"
          ? el("span", { class: "sub" }, ` ${m.context_from}`) : null),
      el("td", { class: "num" }, m.max_output_tokens != null
        ? m.max_output_tokens.toLocaleString() : el("span", { class: "dim" }, "—")),
      el("td", {}, flag(m.tool_call)),
      el("td", {}, flag(m.reasoning)),
      el("td", {}, flag(m.images)),
      el("td", { class: "mono" }, Object.keys(m.request_params).length
        ? JSON.stringify(m.request_params) : el("span", { class: "dim" }, "—"))));
    $("models").tBodies[0].replaceChildren(...rows);
  }

  function renderModelSelect() {
    const select = $("model");
    if (select.options.length) return;
    for (const m of state.data.models) {
      select.append(el("option", { value: m.id }, m.name === m.id ? m.id : `${m.name} (${m.id})`));
    }
    select.value = state.data.default_model;
  }

  function render() {
    const d = state.data;
    renderModelSelect();
    $("base").textContent = d.base_url_from === "config"
      ? `Clients will use ${d.base_url}, from the router's clients.base_url.`
      : `Clients will use ${d.base_url}, the address this page was opened at.`;
    const warnings = $("warnings");
    warnings.hidden = !d.warnings.length;
    warnings.replaceChildren(el("ul", {}, d.warnings.map((w) => el("li", {}, w))));
    renderTabs();
    renderClient();
    renderModels();
    $("raw").replaceChildren(...d.clients.flatMap((c, i) => [
      i ? " · " : null, el("a", { href: c.url }, c.label)]).filter(Boolean));
  }

  function select(id) {
    state.client = id;
    history.replaceState(null, "", `#${id}`);
    renderTabs();
    renderClient();
    const tab = document.getElementById(`tab-${id}`);
    if (tab) tab.focus();
  }

  async function load() {
    try {
      const response = await fetch(`clients${query()}`, { cache: "no-store" });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
      if (!state.data) data.default_model_configured = data.default_model;
      else data.default_model_configured = state.data.default_model_configured;
      state.data = data;
      if (!data.clients.some((c) => c.id === state.client)) state.client = data.clients[0].id;
      render();
    } catch (err) {
      const warnings = $("warnings");
      warnings.hidden = false;
      warnings.replaceChildren(`Couldn't load the client configs: ${err.message || err}`);
    }
  }

  $("user").addEventListener("input", () => {
    clearTimeout(state.timer);
    state.timer = setTimeout(load, 350);
  });
  $("model").addEventListener("change", load);
  $("tabs").addEventListener("keydown", (event) => {
    if (!state.data || !["ArrowLeft", "ArrowRight"].includes(event.key)) return;
    const ids = state.data.clients.map((c) => c.id);
    const step = event.key === "ArrowRight" ? 1 : -1;
    select(ids[(ids.indexOf(state.client) + step + ids.length) % ids.length]);
  });

  state.client = location.hash.slice(1) || null;
  load();
})();
