import assert from "node:assert/strict";
import {test} from "node:test";
import worker, {DailyBudget, validateEvent} from "../src/index.js";

const event = {event: "run_started", module: "ragbuilder", installation_id: "a-" + "1".repeat(32), version: "0.1.7"};
const request = (body = event, headers = {}) => new Request("https://collector.example/events", {
  method: "POST", headers: {"Content-Type": "application/json", ...headers}, body: JSON.stringify(body),
});

function storage() {
  const data = new Map();
  let previous = Promise.resolve();
  const api = {
    get: async key => structuredClone(data.get(key)),
    put: async (key, value) => data.set(key, structuredClone(value)),
    transaction: fn => {
      const current = previous.then(() => fn(api));
      previous = current.catch(() => {});
      return current;
    },
  };
  return api;
}

function environment(limit = "1000") {
  const env = {
    HONEYCOMB_API_KEY: "test-only-token", HONEYCOMB_DATASET: "ragbuilder-usage", DAILY_EVENT_LIMIT: limit,
    CLIENT_LIMIT: {limit: async () => ({success: true})},
  };
  const budget = new DailyBudget({storage: storage()}, env);
  env.BUDGET = {idFromName: () => "global", get: () => budget};
  return env;
}

test("only approved fields and enumerated events are accepted", () => {
  assert.ok(validateEvent(event));
  for (const input of [null, [], {}, {...event, source: "/private/document"}, {...event, error_message: "secret"},
    {...event, event: "custom-event"}, {...event, installation_id: "email@example.com"}, {...event, version: "x".repeat(81)}]) {
    assert.equal(validateEvent(input), false);
  }
});

test("secrets and client IP are never included in forwarded data or responses", async t => {
  let sent;
  t.mock.method(globalThis, "fetch", async (url, options) => {
    sent = {url, options};
    return new Response("private-provider-response", {status: 200});
  });
  const response = await worker.fetch(request(event, {"CF-Connecting-IP": "192.0.2.1"}), environment());
  assert.equal(response.status, 204);
  assert.equal(await response.text(), "");
  assert.equal(sent.url, "https://api.honeycomb.io/1/events/ragbuilder-usage");
  assert.equal(sent.options.headers["X-Honeycomb-Team"], "test-only-token");
  assert.deepEqual(JSON.parse(sent.options.body), {...event, "service.name": "ragbuilder"});
});

test("unknown routes, missing secrets, malformed and oversized payloads do not forward", async t => {
  t.mock.method(globalThis, "fetch", async () => assert.fail("unexpected forwarding"));
  const env = environment();
  assert.equal((await worker.fetch(new Request("https://collector.example/health"), env)).status, 204);
  assert.equal((await worker.fetch(new Request("https://collector.example/"), env)).status, 404);
  assert.equal((await worker.fetch(request(), {...env, HONEYCOMB_API_KEY: ""})).status, 503);
  assert.equal((await worker.fetch(request(event, {"Content-Type": "text/plain"}), env)).status, 415);
  assert.equal((await worker.fetch(request({...event, extra: "x".repeat(2000)}), env)).status, 400);
  assert.equal((await worker.fetch(request(event, {"Content-Length": "9999"}), env)).status, 413);
  assert.equal((await worker.fetch(request({...event, error_message: "private"}), env)).status, 400);
});

test("per-IP rate limit rejects before the durable budget and forwarding", async t => {
  t.mock.method(globalThis, "fetch", async () => assert.fail("unexpected forwarding"));
  const env = environment();
  env.CLIENT_LIMIT.limit = async () => ({success: false});
  env.BUDGET.get = () => assert.fail("budget accessed");
  assert.equal((await worker.fetch(request(), env)).status, 429);
});

test("daily budget is global and cannot be exceeded by concurrent requests", async t => {
  let forwarded = 0;
  t.mock.method(globalThis, "fetch", async () => { forwarded++; return new Response(null, {status: 200}); });
  const env = environment("3");
  const responses = await Promise.all(Array.from({length: 20}, () => worker.fetch(request(), env)));
  assert.equal(forwarded, 3);
  assert.equal(responses.filter(response => response.status === 204).length, 3);
  assert.equal(responses.filter(response => response.status === 429).length, 17);
});

test("one installation cannot consume more than 50 events per day", async t => {
  t.mock.method(globalThis, "fetch", async () => new Response(null, {status: 200}));
  const env = environment();
  for (let i = 0; i < 50; i++) assert.equal((await worker.fetch(request(), env)).status, 204);
  assert.equal((await worker.fetch(request(), env)).status, 429);
});

test("upstream failures consume budget and do not expose errors or retry", async t => {
  let forwarded = 0;
  t.mock.method(globalThis, "fetch", async () => { forwarded++; throw new Error("private-provider-error"); });
  const env = environment("1");
  const response = await worker.fetch(request(), env);
  assert.equal(response.status, 503);
  assert.equal(await response.text(), "");
  assert.equal((await worker.fetch(request(), env)).status, 429);
  assert.equal(forwarded, 1);
});

test("daily counters reset at the next UTC day", async () => {
  const db = storage();
  await db.put("budget", {day: "2000-01-01", total: 1000, installations: {[event.installation_id]: 50}});
  const budget = new DailyBudget({storage: db}, {DAILY_EVENT_LIMIT: "1000"});
  const result = await budget.fetch(new Request("https://budget/", {method: "POST", body: JSON.stringify(event)}));
  assert.equal(result.status, 204);
  assert.equal((await db.get("budget")).total, 1);
});
