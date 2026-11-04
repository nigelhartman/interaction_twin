# Estimated Meta API budget guard

The authorized working allowance is **100 USD**, estimated, across callers sharing
`.secrets/meta-budget.sqlite3`. The default maximum reservation for one request is
**2 USD**. This is an application guard, not a Meta account billing cap or a claim
about credit balance. Playground, other keys, other apps, exchange rates, taxes,
pricing changes and unreported charges can differ from this ledger. Reconcile
against the user's dashboard before adjusting the allowance.

`meta_budget.py` makes no network calls and reads no credentials. It uses exact
Decimal arithmetic rounded up to integer micro-USD, and SQLite `BEGIN IMMEDIATE`
transactions with full durability. Concurrent processes cannot admit work beyond
settled charges plus outstanding reservations. Reopening the same database with
changed caps is rejected; restarting the app cannot reset spending.

## Required transport sequence

1. Define dated, documented `TextPricing` or `SamPricing` and estimate conservatively.
   Text reservations must include image/video input and the provider's total
   output-token maximum. No cache discount is assumed. SAM video estimates charge
   at least 50 frames per request by default.
2. `reserve(stable_request_id, estimate, model=..., endpoint=..., kind=...)` must
   commit before transport. An existing ID is an idempotent lookup, not permission
   to repeat a request. Conflicting use of an ID is rejected.
3. `mark_started(request_id)` must succeed immediately before the single network
   attempt. It succeeds once. No automatic retry is built into the ledger.
4. Settle observed usage with `text_cost()` and `settle()`. Cache and reasoning
   counters avoid double counting. Settlement cannot be below the saved price
   applied to reported counts. Required usage fields cannot be omitted.
5. Timeouts, missing usage, server errors or ambiguous charges must call
   `mark_uncertain()`. The full reservation remains held and new work is blocked.
   A later documented settlement can resolve it. The narrow reviewed exception
   below can classify a proved maximum without releasing that reservation.
   A process crash preserves its in-flight reservation; never rerun that request
   ID blindly.

A known pre-transport cancellation can release its hold. A definite provider
validation rejection can release with a recorded allowlisted HTTP status; 5xx,
redirects, timeouts and successful responses with missing usage cannot.

Actual charges above a reservation are recorded in full and block new work until
an explicit overage review. Acknowledgement preserves all spending and the cap;
there is no automatic credit refill or daily reset. Settlements are idempotent
and cannot be edited to erase historical spend.

## Explicitly reviewed bounded timeout exception

The user authorized an **estimated** 100 USD allowance, rather than requiring an
exact invoice before every new request. A single timed-out Spark request can have
an unknown actual charge but a verified maximum. The optional, explicit
`acknowledge_bounded_text_uncertainty(request_id, proof=..., authorization_reference=...)`
records that review. It is never invoked automatically by transport or the app.

This method is pinned to the dated standard Muse Spark 1.3 rates in the original
reservation: 1.25 USD per million input tokens, 4.25 USD per million output tokens,
and 0.15 USD cached input. It requires a full **1,048,576-token input reservation**,
zero assumed cached tokens, a total output cap of at most 24,000 tokens including
reasoning, exactly one attempt/output, and no tools, background processing or
other billable modalities. For the 16,000-output-token request, the maximum is
**1.378720 USD**. Image inputs must already be covered by that model context bound.

The proof must bind the existing uncertain request to its original, hashed
`meta-request.json`, a sibling hashed `metadata.json` containing original source
and frame-index hashes, its payload hash, and the independently reviewed current
`meta_api.py` source hash. The receipt must identify a transport timeout with no
HTTP status, no retry and no Contributor mode. An unknown pricing/model, missing
source binding, multiple attempts, SAM request or HTTP error remains blocked.
Reviewers must verify these facts, including the transport actually used, before
calling the method; hashes alone do not establish them.

The new `bounded_unknown` state retains **every micro-dollar of the original
reservation** in all admission checks. `actual` and `usage` remain null. It is not
reported as known spending, a refund, free usage, or a fresh allowance. Status
exposes `unknown_usage`, `unknown_usage_count`, `bounded_unknown_count` and
`bounded_unknown_usd`. The invariant remains:

`settled estimated costs + bounded unknown holds + other reservations <= 100 USD`

The acknowledgement creates a hash-bound immutable audit record. Repeating the
same acknowledgement is idempotent; changing its proof is rejected. The request
cannot be retried or have its hold released. Genuine late usage can settle it
normally while preserving the audit. No cap/config/history is reset or increased.

Restart all app/model workers when deploying this policy. A database admission
trigger also fences older loaded code: its unreviewed SQLite connections cannot
insert reservations and therefore cannot omit the new state's retained costs.
Read-only dashboard checks still work, but an old app may display stale totals
until it restarts.

Proof fields are deliberately exact (no arbitrary comments, prompts or keys):
`request_receipt_path`, `request_receipt_sha256`, `source_record_path`,
`source_record_sha256`, `source_sha256`, `frame_index_sha256`,
`request_payload_sha256`, `transport_source_sha256`, `context_input_tokens`,
`max_output_tokens`, `attempt_count`, `n`, `tools`, `background`,
`other_billable_modalities`, `total_output_limit_includes_reasoning`,
`pricing_tier`, and `timeout_confirmed`. Paths must be absolute non-secret evidence
files; their bytes are bounded and hash-checked before an acknowledgement commits.

The ledger stores request identity, model/endpoint, dated pricing, numeric usage,
charge/state and sanitized reason codes. It never stores prompts, images, API keys
or arbitrary provider error bodies. `read_budget_status(path)` is read-only and
never initializes a missing database; its balance is labelled local estimate.

Offline verification: `python3 -m unittest tests.test_meta_budget -v` covers exact
rounding, concurrency, transaction rollback, caps, restarts, idempotency, uncertain
charges, overages, actual-usage validation, the SAM video minimum, bounded-timeout
proof validation, immutable audits, stale-code admission fencing, and concurrency
at the cap with a retained unknown hold. No real API requests are made by these
tests.
