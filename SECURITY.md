# Security and maintenance

RAGBuilder is no longer actively maintained. Ongoing fixes and support are not guaranteed. Use it only with trusted operators and data. Do not expose the UI, an unauthenticated API, or a database directly to the internet.

## Security patch

This patch removes the shared Honeycomb and Mixpanel credentials, routes minimal usage events through a bounded collector, escapes generated Python arguments and displayed chat content, restricts document downloads, confines UI file access, and requires authentication for network access. It also upgrades vulnerable dependencies and moves legacy LangChain imports to `langchain-classic`.

Python 3.10 or newer is required. Use `requirements.lock` for the tested dependency set. Existing saved Python pipelines, databases, notebooks, vector stores, and caches remain trusted executable inputs. Do not load artifacts supplied by an untrusted party. These changes do not retrofit previously generated Python pipelines or already installed releases.

## Remaining upstream advisories

As of October 1, 2026, the dependency audit still reports the advisories below with no published fixed version. They are not claimed to be fixed by this patch. The audit script records these specific exceptions and fails on other findings.

| Dependency | Advisory | Exposure and mitigation |
| --- | --- | --- |
| Ragas 0.4.3 | [CVE-2026-6587](https://github.com/advisories/GHSA-95ww-475f-pr4f) | The affected multimodal faithfulness collection is not used by RAGBuilder's text evaluation paths. Do not add or invoke multimodal metrics on untrusted inputs. |
| DiskCache 5.6.3, a Ragas dependency | [CVE-2025-69872](https://github.com/advisories/GHSA-w8v5-vhqr-4h9v) | Pickle deserialization is unsafe if another user can write the cache. RAGBuilder does not configure a DiskCache backend. Keep all cache directories private and do not import untrusted caches. |
| ChromaDB | [CVE-2026-45829](https://github.com/advisories/GHSA-f4j7-r4q5-qw2c), [CVE-2026-45833](https://github.com/advisories/GHSA-36p7-vc44-83pf), [CVE-2026-45831](https://github.com/advisories/GHSA-xph7-9rjv-w5fr), [CVE-2026-45830](https://github.com/advisories/GHSA-2wm9-hf6c-p5cr) | These affect the Chroma server and authorization boundary. The default RAGBuilder integration uses local Chroma, not a hosted Chroma server. Do not expose a Chroma server, enable `trust_remote_code`, or rely on its affected authorization for tenant isolation. |

`langchain-community` is pinned to 0.4.1 because 0.4.2 removes a VertexAI compatibility module still imported by Ragas 0.4.3. The pinned version includes the XXE fix identified in the repository audit. Re-run the audit when changing this pin.

## Deployment boundaries

- Keep the default localhost binding. For remote access use a strong `RAGBUILDER_API_TOKEN` and HTTPS through a trusted reverse proxy. The token grants operator access, not isolated access for multiple tenants.
- Set `RAGBUILDER_DATA_ROOT` to a dedicated directory containing only documents intended for processing. Selected directories must not contain hidden files, hidden directories, or symlinks to hidden or out-of-root content. The SDK is a local programming interface and retains caller-directed filesystem access.
- Document and prompt URL downloads use public IP addresses only, pin the resolved address, validate each redirect, verify TLS, and cap decoded response size at 20 MiB. HTTP proxy environment settings are not used for these downloads.
- Set `NEO4J_PASSWORD` before deploying Compose. For an existing Neo4j volume, rotate the database password through Neo4j itself; changing `.env` alone does not rotate it.
- Disable the previously published Honeycomb key and revoke any matching historical OpenAI key in the provider consoles. Removing source text cannot revoke credentials or erase old clones, releases, and Git history.
- Keep secrets out of commits, images, notebooks, and logs. GitHub secret scanning and push protection should stay enabled.

## Verification

Run `python -m pytest -c tests/security/pytest.ini tests/security` and `npm test --prefix telemetry-collector`. The tests use synthetic data and do not call paid model APIs or send real telemetry. Run `python scripts/audit_dependencies.py` after installing `pip-audit`; its explicit exceptions are the unresolved advisories above.

The GitHub Actions workflow is prepared at `scripts/security-workflow.yml`. To activate it, move it to `.github/workflows/security.yml` and push using GitHub credentials with the `workflow` permission. The current automation login lacks that permission, so CI has not been activated by this patch.
