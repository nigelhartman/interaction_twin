# Private owner administration

The public site reads the isolated curated snapshot under `showcase/public/`. New uploads, processing runs, feedback and candidate revisions are private. The server enforces this boundary independently of navigation links.

Create the owner password locally with `.venv-sam/bin/python scripts/admin_access.py set-password`. The CLI uses a hidden password prompt and stores a salted scrypt hash. It never accepts a password argument or prints a password. Alternatively, `scripts/admin_access.py bootstrap` writes a random one-use setup code to the owner-only `.secrets/admin-setup-code`; open that file yourself and use the setup screen at `/admin/`. The code expires after 24 hours and is consumed when a password is set. Passwords require at least 12 characters. `reset-password` uses another hidden local prompt and invalidates existing sessions.

Auth HTTP contract:

- `GET /api/admin/auth` returns `{configured, authenticated, setup_available, csrf_token}`. It creates an anonymous session when needed. Responses are never cached.
- `POST /api/admin/setup` accepts exactly `{setup_code, password}`.
- `POST /api/admin/login` accepts exactly `{password}`.
- `POST /api/admin/logout` accepts `{}` and destroys the authenticated session.

Use same-origin credentials for fetch requests. Every POST requires an exact local `Origin` header, the server-issued session cookie and an `X-CSRF-Token` header matching the token from the auth response. Setup and login accept the anonymous session; all other mutations require authenticated Nigel ownership. Successful setup/login rotates both session ID and CSRF token. Cookies are `HttpOnly; SameSite=Strict; Path=/`; the current server binds to plain HTTP loopback only. Owner sessions expire after 12 hours and are kept server-side, so a server restart signs the owner out. Ten failed sign-in attempts block further attempts for 15 minutes. There is no default password or external identity provider.

Authenticated workflow routes are delegated to `admin_http.dispatch(handler, route, method, body)`. Authentication and mutation-CSRF checks run before dispatch. The handler supplies `admin_actor='nigel'` for authorized mutations. Workflow handlers must not infer ownership from body fields.

Existing upload, pipeline, cancellation, repair, human-review, package-export and model-job mutation routes require the same owner session and CSRF header. Raw uploaded videos and new pipeline outputs cannot be anonymously downloaded. `/workspace/` and private job/config/budget APIs require owner access; only explicitly allowed pure renderer/vendor modules remain public. The robot rehearsal is an owner workspace.

Anonymous `/files/data/...` access requires an exact matching record in `showcase/public/public-assets.json`:

```json
{"format":"interaction-twin-public-assets","version":1,"assets":[{"path":"data/example/asset.glb","sha256":"...64 hexadecimal characters...","bytes":123}]}
```

The server verifies the actual file length and SHA before serving it, including range requests. Broad directory prefixes, private source paths, secret files, hidden paths and traversal are never publication rules. Old unfiltered showcase catalogs/registries and workspace case payloads are owner-only. Public library search/facets use the curated snapshot, not the full internal catalog.

The default MCP server uses that same curated snapshot and exact artifact allowlist. A known private working-case ID cannot bypass the approval gate, and newer same-source candidates or companions cannot replace published references. The separate `--private-owner` configuration deliberately exposes the owner's full local diagnostics; public clients should use the default or `--curated`. Approved records expose `curation.human_guidance_used` as a boolean from the revision manifest. Annotation text and private memory remain excluded.

Private processing uses `Runner.execute('meta_sam3d_default', input_path=..., memory_context_path=..., publication_mode='admin_review')`. The memory argument is an immutable owner-only JSON snapshot under `.private/`, returned by scoped feedback retrieval. The runner verifies its internal context digest, exact file SHA, permissions and original upload identity; changed feedback requires a separate review run. Unknown or symlink paths fail before dispatch.

`scripts/run_admin_stage.py` accepts only the six existing reviewed Meta stage adapters. The child client preserves the fixed additional task allowance (at most $5 from the existing baseline), the older increment allowance and the global $100 estimated ledger. It adds bounded owner annotations only to structured reasoning requests, treating them as fallible preferences to evaluate against source evidence. SAM segmentation payloads are unchanged. Feedback does not train model weights, authorize tools or establish observed/physical truth. The original mechanism prompt is frozen; a later mechanism request can consume the newly reasoned semantic program, but receives no direct feedback injection.

The original stage `prompt.txt` remains its base prompt. Each augmented call saves its exact effective prompt privately in `.private/admin-prompts/`; `meta-request.json` records the base/effective prompt hashes and private proof reference without copying annotation text. These records are owner-only and excluded from the curated public packager. Admin runs pass `--private` to object-program export and `--no-replays` to workspace refresh, so processing does not advance the public library or scene publication registry. Only explicit approval releases a hash-current revision for curated packaging.

A failed hosted text review preserves its once-only attempt and may leave an unknown-charge hold. It is never resent automatically. For a separately reviewed, source/code-bound admin review plan, `scripts/audit_admin_text_uncertainty.py --request-id ... --proof output/...json` performs read-only validation. The distinct explicit `--apply --authorization-reference ...` operation records an immutable audit and changes an eligible transport failure from `uncertain` to `bounded_unknown`. It retains the entire full-context reservation, unknown usage and original failure reason. It does not assert a timeout, establish successful review, release funds, change spending caps or allow that role to resend. Other eligible requests can be considered only with the retained hold still counted in every budget. HTTP, image/SAM, changed code/payload, multiple attempts and unverified estimates remain ineligible. No server or queue path invokes this audit.
