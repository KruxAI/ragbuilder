const EVENTS = new Set(["installation_started", "run_started", "run_completed", "run_failed", "error"]);
const MODULES = new Set(["ragbuilder", "data_ingest", "retriever", "generation", "eval_data_generation", "ui"]);
const FIELDS = new Set(["event", "module", "installation_id", "version"]);
const MAX_BODY = 1024;

export function validateEvent(value) {
  if (!value || Array.isArray(value) || typeof value !== "object") return false;
  if (Object.keys(value).some(key => !FIELDS.has(key))) return false;
  return EVENTS.has(value.event) && MODULES.has(value.module)
    && typeof value.installation_id === "string" && /^a-[a-f0-9]{32}$/.test(value.installation_id)
    && typeof value.version === "string" && /^[A-Za-z0-9.+_-]{1,80}$/.test(value.version);
}

async function readEvent(request) {
  if (!request.body) throw new Error("Missing body");
  const reader = request.body.getReader();
  const chunks = [];
  let size = 0;
  try {
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > MAX_BODY) throw new Error("Body too large");
      chunks.push(value);
    }
  } finally {
    await reader.cancel();
  }
  const bytes = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
  return JSON.parse(new TextDecoder("utf-8", {fatal: true}).decode(bytes));
}

export class DailyBudget {
  constructor(ctx, env) { this.ctx = ctx; this.env = env; }
  async fetch(request) {
    const {installation_id} = await request.json();
    const configuredLimit = Number(this.env.DAILY_EVENT_LIMIT);
    const limit = Number.isInteger(configuredLimit) && configuredLimit > 0 ? Math.min(configuredLimit, 1000) : 1000;
    const day = new Date().toISOString().slice(0, 10);
    const allowed = await this.ctx.storage.transaction(async storage => {
      let budget = await storage.get("budget");
      if (!budget || budget.day !== day) budget = {day, total: 0, installations: {}};
      const count = budget.installations[installation_id] || 0;
      if (budget.total >= limit || count >= 50) return false;
      budget.total += 1;
      budget.installations[installation_id] = count + 1;
      await storage.put("budget", budget);
      return true;
    });
    return new Response(null, {status: allowed ? 204 : 429});
  }
}

export default {
  async fetch(request, env) {
    const path = new URL(request.url).pathname;
    if (request.method === "GET" && path === "/health") return new Response(null, {status: 204});
    if (request.method !== "POST" || path !== "/events") return new Response(null, {status: 404});
    if (!env.HONEYCOMB_API_KEY) return new Response(null, {status: 503});
    if (!request.headers.get("content-type")?.toLowerCase().startsWith("application/json")) {
      return new Response(null, {status: 415});
    }
    if (Number(request.headers.get("content-length")) > MAX_BODY) return new Response(null, {status: 413});
    const ip = request.headers.get("CF-Connecting-IP") || "unknown";
    if (!(await env.CLIENT_LIMIT.limit({key: ip})).success) return new Response(null, {status: 429});
    let event;
    try { event = await readEvent(request); }
    catch { return new Response(null, {status: 400}); }
    if (!validateEvent(event)) return new Response(null, {status: 400});
    try {
      const budget = env.BUDGET.get(env.BUDGET.idFromName("global"));
      const allowance = await budget.fetch(new Request("https://budget/", {
        method: "POST", body: JSON.stringify({installation_id: event.installation_id}),
      }));
      if (allowance.status !== 204) return new Response(null, {status: 429});
      // Consume the budget before forwarding. Failures do not retry or refund it.
      const upstream = await fetch(`https://api.honeycomb.io/1/events/${encodeURIComponent(env.HONEYCOMB_DATASET)}`, {
        method: "POST", redirect: "error", signal: AbortSignal.timeout(3000),
        headers: {"Content-Type": "application/json", "X-Honeycomb-Team": env.HONEYCOMB_API_KEY},
        body: JSON.stringify({...event, "service.name": "ragbuilder"}),
      });
      const ok = upstream.ok;
      await upstream.body?.cancel();
      return new Response(null, {status: ok ? 204 : 502});
    } catch {
      return new Response(null, {status: 503});
    }
  },
};
