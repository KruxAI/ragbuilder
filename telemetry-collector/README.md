# RAGBuilder usage collector

This Cloudflare Worker keeps the Honeycomb key outside the distributed Python package. It forwards only event type, component name, package version, and a random installation ID. Documents, prompts, answers, errors, filesystem paths, and client IPs are not forwarded. No request-body logging is enabled.

The endpoint is public. Events can be forged and are an indication of usage, not proof of genuine installations. Traffic is limited to approximately 10 requests per IP per minute, at most 50 forwarded events per installation ID per UTC day, and at most 1,000 forwarding attempts globally per UTC day. A single SQLite-backed Durable Object enforces the daily budget transactionally. The budget is consumed before calling Honeycomb, including failed attempts; there are no retries.

Workers Free and SQLite-backed Durable Objects both have free quotas. This deployment is intended for the Free plan. Exceeding a free quota stops operations rather than upgrading the account. Check your account's actual plan and other Workers' shared usage before deployment. See [Workers pricing](https://developers.cloudflare.com/workers/platform/pricing/) and [Durable Objects pricing](https://developers.cloudflare.com/durable-objects/platform/pricing/).

## Deploy

From this directory:

```sh
npm ci
npm test
npm exec wrangler login
npm run check
npm run deploy
npm exec wrangler secret put HONEYCOMB_API_KEY
```

The last command prompts for the key without putting it in shell history or source files. Use a new Honeycomb ingest-only key for the US region (`api.honeycomb.io`). Create the `ragbuilder-usage` dataset beforehand, or grant the key permission to create it. Never use a management or configuration key. Do not paste the key into chat or commit it. The deployed collector returns 503 for event submissions until its secret is configured.

Set `DEFAULT_TELEMETRY_ENDPOINT` in `src/ragbuilder/core/telemetry.py` to the deployed URL ending in `/events` before publishing the Python package. This URL is public and contains no credential. Leave `ENABLE_ANALYTICS` enabled by default; users can opt out with `ENABLE_ANALYTICS=false`. `RAGBUILDER_TELEMETRY_URL` overrides the URL for self-hosting.

`GET /health` checks reachability without generating a Honeycomb event. It does not validate the Honeycomb secret. After configuring the secret, send one synthetic test event and confirm it in Honeycomb, then revoke the old exposed key if it has not already been disabled.

To pause ingestion, remove the Worker secret or disable the Worker. To reduce the daily budget, lower `DAILY_EVENT_LIMIT` and redeploy. Values above 1,000 do not increase the hard cap. An attacker can exhaust the budget and hide legitimate usage until the next UTC day; this is an intentional tradeoff to bound ingestion.
