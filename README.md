---
title: FinVault - Enterprise Multi-Agent AI
emoji: 🛡️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
suggested_hardware: cpu-upgrade
---

# FinVault

🚀 **Live Demo on Google Cloud:** [https://finavault-335067811983.europe-west1.run.app/app](https://finavault-335067811983.europe-west1.run.app/app)  
📚 **Interactive API Docs (Swagger):** [https://finavault-335067811983.europe-west1.run.app/docs](https://finavault-335067811983.europe-west1.run.app/docs)

A secure, self-hostable multi-agent RAG platform for finance. Built to be usable as a solo/small-business tool and as something a bank, hedge fund, or corporate finance/compliance team could adopt — security is the product's core differentiator, not a bolt-on.

**The pitch:** ask questions of your most sensitive financial documents without your most sensitive financial documents ever leaving your control unencrypted or unclassified.

> This is Phase 1 of a larger roadmap. The agent scaffolding (`agents/base.py`, the orchestrator/delegation pattern, the guardrail layer) is deliberately generic rather than finance-hardcoded, so a later phase can build a broader **agent operating system** on top of it.

## Why nothing leaks by default

1. **Ingest** — document → chunk → classify (`public` / `internal` / `confidential` / `restricted`) → embed **locally** (no third-party embedding API — `sentence-transformers` runs on-device) → envelope-encrypt each chunk (AES-256-GCM, per-chunk data key wrapped by a master key) → store ciphertext + routing metadata in the vector DB.
2. **Retrieve** — query embedded locally → vector search, org-scoped → ACL check per hit (does this user's role clear this chunk's classification?) → decrypt only what's authorized, in memory. Chunks a user lacks clearance for are silently dropped, not surfaced as a denial — so a lower-clearance user can't even infer that a matching restricted document exists.
3. **Reason** — retrieved content is wrapped in explicit untrusted-data delimiters with an instruction never to treat it as commands (prompt-injection defense), plus a heuristic scanner that flags known injection patterns as a second, independent signal.
4. **Send to the LLM** — the one point data leaves the system boundary. A chunk whose classification isn't in `FINVAULT_ALLOWED_EXTERNAL_CLASSIFICATIONS` (`restricted` is excluded by default) never has its text assembled into a tool result in the first place — enforced in the Retriever agent, at the point content would otherwise enter a prompt, not just checked after the fact on the final answer. ACL clearance (can this user *retrieve* the chunk) and externalization policy (can this chunk's content ever reach the LLM) are independent checks: a compliance officer can be fully authorized to retrieve a `restricted` memo without that memo's text ever leaving the system boundary. LLM calls are routed through [OpenRouter](https://openrouter.ai) (an OpenAI-compatible gateway) rather than calling Anthropic directly — see "Known limitations" below for what that means for this boundary.
5. **Respond** — every draft answer passes through the Compliance agent before the user sees it: a PII/secret pattern scan + redaction, then a lightweight semantic review. This runs as plain Python control flow, not as a tool the orchestrating LLM could choose to skip — that's what gives it real veto power.
6. **Audit** — every query, retrieval, and compliance decision is written to a hash-chained, append-only log. Any out-of-band tampering with a past entry breaks the chain from that point forward (`AuditLog.verify_chain()`).

## Architecture

```
User query ──▶ Orchestrator Agent (routes, assembles final answer)
                 │
                 ├──▶ Retriever Agent  (RAG search tool, ACL + decrypt)
                 └──▶ Analyst Agent    (financial reasoning, sandboxed calculator)
                 │
                 ▼
          Compliance / Guardrail Agent
     (input: injection scan · output: PII scan, classification ceiling, veto)
                 │
                 ▼
            Final answer
```

Four agent identities (per `agents/`):

| Agent | Role |
|---|---|
| **Orchestrator** | Delegates to Retriever/Analyst via tool calls, assembles the draft answer. |
| **Retriever** | Only tool: `search_documents`, wrapping `Retriever.retrieve` (ACL-filtered, decrypted). |
| **Analyst** | Grounded financial reasoning; arithmetic goes through a restricted AST-based calculator, not `eval()`. |
| **Compliance** | Runs deterministically after every draft answer — externalization policy, PII/secret scan, semantic review. Not exposed as a callable tool. |

## Project layout

```
src/finvault/
├── errors.py             # ← every failure mode, one hierarchy. Start here when debugging.
├── observability.py      # structured logging + the request id that correlates it
├── metrics.py            # Prometheus instrumentation (agent latency, token spend)
├── cache.py              # Cache interface + Redis + in-memory; tenant-scoped keys
├── config.py            # deployment-agnostic settings (env-driven)
├── db.py                 # Postgres schema: documents, audit_log
├── migrate.py            # `finvault-migrate` — applies revisions, then installs RLS policies
├── migrations/           # Alembic environment + reviewed revisions (ships with the package)
├── models.py              # Classification, Role, User, Document, Chunk, ...
├── security/
│   ├── encryption.py     # KeyProvider interface + LocalKeyProvider, envelope AES-256-GCM
│   ├── access_control.py # role clearance + org isolation checks
│   ├── audit.py           # hash-chained append-only audit log (in-memory + Postgres)
│   ├── rls.py             # PostgreSQL Row Level Security — isolation the database enforces
│   └── guardrails.py      # injection-defense wrapping/detection, PII redaction, externalization policy
├── ingestion/
│   ├── loaders.py         # PDF/DOCX/TXT/MD/CSV → plain text
│   ├── chunking.py        # paragraph-aware chunking with overlap
│   ├── embeddings.py      # EmbeddingProvider interface + local sentence-transformers
│   └── pipeline.py        # load -> chunk -> classify -> embed -> encrypt -> store
├── retrieval/
│   ├── vector_store.py    # VectorStore interface + Qdrant + in-memory reference impl
│   └── retriever.py       # search -> ACL filter -> decrypt
├── agents/                # base.py (generic tool-use loop) + the four agents
└── api/
    ├── error_handlers.py  # the only place an exception becomes an HTTP response
    ├── rate_limit.py      # per-identity request limits protecting /query
    ├── uploads.py         # bounded, self-cleaning upload handling
    ├── auth.py            # JWT verification -> User
    └── routes.py          # /documents, /query, /compare, /compliance; mounts frontend/ at /app
frontend/
└── index.html             # single-page UI: login, document upload, chat — talks to the API above

deploy/
├── postgres/01-app-role.sql        # the non-superuser role the app connects as (required for RLS)
├── observability/                  # Prometheus scrape config, alert rules, Grafana dashboard
└── helm/finvault/                  # production chart: Deployment, HPA, PDB, migration hook, NetworkPolicy
.github/workflows/ci.yml            # lint + tests + Postgres/Redis integration + dependency audit
Dockerfile                          # multi-stage, non-root, read-only root filesystem
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# set OPENROUTER_API_KEY (get one at https://openrouter.ai/keys)
# generate a real JWT secret for anything beyond throwaway local use:
#   python -c "import secrets; print(secrets.token_urlsafe(48))"
```

### Start Qdrant + Postgres

`docker-compose.yml` runs both on non-default ports (Qdrant `6350`, Postgres `5433`) so FinVault doesn't collide with another project's DB stack that might already be running on the standard ports:

```bash
docker compose up -d   # or: docker-compose up -d, depending on your Docker install
```

If neither `docker compose` nor `docker-compose` is available, start the two containers directly (equivalent to the compose file):

```bash
docker run -d --name finvault-qdrant -p 6350:6333 \
  -v finvault_qdrant_data:/qdrant/storage --restart unless-stopped qdrant/qdrant:v1.12.4

docker run -d --name finvault-postgres -p 5433:5432 \
  -e POSTGRES_USER=finvault -e POSTGRES_PASSWORD=finvault_dev_only -e POSTGRES_DB=finvault \
  -v finvault_postgres_data:/var/lib/postgresql/data --restart unless-stopped postgres:16-alpine
```

### Run the tests

```bash
pytest
```

### Run the end-to-end demo (CLI)

```bash
python scripts/ingest_sample.py
```

Ingests three sample finance documents at different classification tiers (`internal`, `confidential`, `restricted` — the restricted one deliberately contains an embedded prompt-injection attempt) and runs several queries as users with different roles, printing the ACL/guardrail/audit decisions made along the way. Falls back to in-memory vector store / audit log automatically if Qdrant/Postgres aren't reachable, so it also runs with zero infra beyond an API key.

### Run the full app (API + frontend)

```bash
uvicorn finvault.api.main:app --reload
```

Open **http://localhost:8000/app/** for the web UI (sign in with any username/role/org — see the dev-login note in `api/routes.py`, upload a document, ask a question) or drive the API directly:

```bash
TOKEN=$(curl -s -X POST localhost:8000/auth/token -H 'content-type: application/json' \
  -d '{"username": "dana", "role": "analyst", "org_id": "acme-capital"}' | python3 -c "import sys,json;print(json.load(sys.stdin)['access_token'])")

curl -X POST "localhost:8000/documents?classification=internal" \
  -H "authorization: Bearer $TOKEN" -F "file=@sample_docs/quarterly_report.txt"

curl -X POST localhost:8000/query -H "authorization: Bearer $TOKEN" -H 'content-type: application/json' \
  -d '{"question": "What was Q3 revenue growth?"}'
```

## Hugging Face Spaces Deployment (Docker)

FinVault is ready to deploy directly to **Hugging Face Spaces** using the **Docker SDK**.

### Option A: Automatic CLI Push

Use the helper deployment script:

```bash
python scripts/deploy_hf.py --space-id <your-username>/<your-space-name> --hf-token <your-hf-write-token>
```

### Option B: Manual Git Push

1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces) with **SDK: Docker** and **Blank** template.
2. Clone your HF Space repository locally and push this codebase:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/<your-space-name>
   git push space main
   ```

### Option C: Hugging Face Secrets Configuration

In your Space's **Settings -> Variables and secrets**, set the following environment secrets:

| Secret / Variable | Description | Example / Default |
|---|---|---|
| `OPENROUTER_API_KEY` | OpenRouter API Key for multi-agent LLM reasoning | `sk-or-v1-...` |
| `HF_TOKEN` | Hugging Face Access Token (optional if using HF router) | `hf_...` |
| `FINVAULT_MODEL` | Default LLM model slug | `minimax/minimax-m2.7:free` |
| `FINVAULT_JWT_SECRET` | Secret key for JWT signature validation | `<generate-random-token-urlsafe>` |
| `POSTGRES_DSN` | (Optional) Remote PostgreSQL DSN (Neon/Supabase) | `postgresql://...` (or auto-embedded SQLite) |
| `QDRANT_URL` | (Optional) Qdrant Cloud URL | `https://...` (or auto-embedded `:memory:` / path) |

---

## Error handling

Two files carry the whole story, and both are written to be read:

| File | What it answers |
| --- | --- |
| [`src/finvault/errors.py`](src/finvault/errors.py) | *What can go wrong, and what should happen when it does.* Every exception the system raises on purpose, in one hierarchy, each documenting who raises it, who catches it, and whether a retry helps. |
| [`src/finvault/observability.py`](src/finvault/observability.py) | *How you find out that it did.* Structured JSON logging, and the request id that ties an error a user saw to the traceback that caused it. |

The rules the rest of the codebase follows:

- **One hierarchy.** Everything descends from `FinVaultError`, in four branches: `ClientError` (the caller got it wrong → 4xx), `PolicyError` (refused on purpose → 403), `DependencyError` (an upstream failed → 503), `InternalError` (our bug → 500). No module defines its own exception class.
- **Routes raise; they do not translate.** No route picks a status code or writes an error string — `api/error_handlers.py` does that once, from the exception's own `code`, `http_status`, and `user_message`. Adding a failure mode means adding a class and raising it, not editing the API layer.
- **Every error response has the same shape**, whether it came from us, from FastAPI, or from an unhandled bug:

  ```json
  {"error": {"code": "access_denied", "message": "You do not have clearance for this resource.",
             "retryable": false, "request_id": "9f2c…"}}
  ```

- **Internal detail never reaches the caller.** The operator-facing message and its `context` go to the log; the client gets the class's vetted `user_message`. An unhandled exception is answered with an opaque 500 — its message could contain anything.
- **Fail closed.** A dependency failure never degrades into a guess. A failed retrieval is not "nothing found", and an unreachable compliance reviewer is a block, not an approval.
- **One request id, everywhere.** Returned in the `X-Request-ID` header and in every error body, stamped on every log record for that request — including the ones from the background thread serving `/query/stream`. A bug report that quotes it is enough to find the logs: `grep <id>`.
- **Log level follows the branch, not the call site.** Expected refusals log at WARNING with no traceback; incidents log at ERROR with one. `log_exception()` decides, so the levels stay consistent.

Tests for this layer: `tests/test_errors.py` (taxonomy invariants), `tests/test_error_handlers.py` (the HTTP contract, end to end), `tests/test_observability.py`, `tests/test_uploads.py`.

## Production operations

### Caching (`src/finvault/cache.py`)

Redis when `FINVAULT_REDIS_URL` is set, an in-memory equivalent otherwise — so
the system runs without it, and an unreachable Redis degrades performance,
never availability. Two caches:

- **Embeddings**, keyed by `(model, text)`. A pure function, so a hit is
  exactly what the model would have produced; no tenant scoping needed.
- **Query answers**, keyed by `(org, role, question, corpus generation)`.

That second key is the security-relevant one. **The role is in the key because
retrieval is clearance-filtered**: two users in the same org asking the
identical question are entitled to different answers, and a key without the
role would serve the first asker's answer to the second — a clearance bypass
that leaves no trace. `scoped_key()` is the only supported way to build one, so
no call site can omit it.

Keys are HMAC'd rather than plain digests, so reading the cache does not let
someone confirm a guessed document by hashing it themselves. An ingest bumps a
per-org **corpus generation** counter that is part of every answer key, which
retires that org's cached answers instantly — TTL alone cannot do that, and a
user who uploads a policy then asks about it must not be told what was true
before the upload. Conversation turns, blocked results, and uncited answers are
never cached; the reasons are in `_cacheable()`.

### Rate limiting (`src/finvault/api/rate_limit.py`)

Two windows per identity (burst + sustained), enforced as ASGI middleware so a
client already over its limit never gets its body parsed. Authenticated callers
are limited by **account**, not IP — NAT puts hundreds of legitimate users
behind one address — with IP as the fallback for anonymous traffic only, on a
much tighter allowance. A 429 carries `Retry-After`.

This is the one layer in FinVault that **fails open**: if the counter backend is
unreachable, requests are allowed and a `failed_open` metric fires. Refusing
everything because a counter is down would turn a cache outage into the very
denial of service the limiter exists to prevent.

### Schema migrations (`src/finvault/migrate.py`)

Alembic, with the revisions living **inside the package** (`finvault/migrations/`)
rather than in a top-level directory — the Helm pre-install Job runs out of the
application image, which has the installed package and not the repo layout.

```bash
alembic upgrade head                 # apply (reads POSTGRES_DSN via config.settings)
alembic revision --autogenerate -m "add retention_until"
alembic downgrade -1                 # step back one
alembic -x url=postgresql://... upgrade head   # override the target database
```

`alembic.ini` deliberately carries no `sqlalchemy.url`: `env.py` reads the DSN
from `config.settings`, so the connection string lives in one place and a
checked-in file never holds a credential.

Two paths build the schema, and the split is the point:

| | Path | Can it alter an existing table? |
|---|---|---|
| Dev / tests | `FINVAULT_AUTO_CREATE_SCHEMA=true` → `metadata.create_all` | **No** — creates missing tables only |
| Everywhere else | `finvault-migrate` → `alembic upgrade head`, then RLS policies | Yes |

`create_all` is kept because a throwaway database should not need Alembic
config, but it can only ever create — which is exactly how a schema change
ends up working on a laptop and nowhere else. CI closes that gap: the
integration job applies the migrations to a scratch database and runs
`alembic check`, so editing a `Table` in `db.py` without writing a revision
fails the build. `tests/test_migrations.py` asserts the stronger property
directly — that the migrated schema and `metadata` reflect identically.

RLS policies are **not** in a migration. They are re-derived idempotently from
`RLS_TABLES` on every deploy, which keeps a policy change from needing a
revision and keeps `verify_isolation()` the single authority on whether
isolation is actually in force. `finvault-migrate` runs both steps in order.

### Row Level Security (`src/finvault/security/rls.py`)

Tenant isolation the *database* enforces. With policies installed, a query that
forgets its `WHERE org_id = ...` returns **no rows** rather than another
tenant's — application filters are a convention, a policy is a mechanism.

Enable with `FINVAULT_ENABLE_RLS=true`. Two conditions must hold, and **both
fail silently if they do not**, which is why the app verifies them at startup
and refuses to serve when either fails:

1. **Connect as a non-superuser role without `BYPASSRLS`.** A superuser ignores
   row security entirely — policies installed, `pg_policies` listing them,
   nothing filtered. The `finvault` account created by `POSTGRES_USER` is a
   superuser, so use `finvault_app` (`deploy/postgres/01-app-role.sql`).
2. **That role must not own the tables** — plain `ENABLE` exempts an owner, so
   `FORCE ROW LEVEL SECURITY` is set too, and schema creation moves to a
   separate migration run as the owner (`FINVAULT_AUTO_CREATE_SCHEMA=false`;
   the Helm chart ships a pre-install Job that runs `finvault-migrate` — see
   [Schema migrations](#schema-migrations-srcfinvaultmigratepy)).

Any code path without a request behind it must declare its scope with
`org_scope(org_id)` — scripts, jobs, evaluation harnesses. Forgetting shows
nothing rather than everything, which is the correct direction.

Covered: `documents`, `review_queue`, `graph_nodes`, `graph_edges`. Not covered,
deliberately: `audit_log` (one hash chain across the deployment — a policy would
hide rows from `verify_chain()`) and `sessions` (scoped by user UUID, not tenant
data; bringing it under RLS needs a schema change and is a genuine follow-up).

### Metrics and dashboards (`src/finvault/metrics.py`)

`GET /metrics` serves Prometheus exposition — unauthenticated by design, since
it exposes counters and latencies but never document content or identity;
restrict it at the network layer (the chart ships a `NetworkPolicy`).

Labels are drawn from small closed sets only. Route labels are path
*templates*, so `/documents/{document_id}` is one series rather than one per
document — the classic way to melt a metrics backend, and a test pins it.

```bash
docker compose -f docker-compose.yml -f docker-compose.observability.yml up -d
# Grafana at http://localhost:3300 — dashboard provisioned, no login needed
```

The dashboard covers agent latency (p95 per agent, query percentiles), token
spend (by direction and by agent, plus estimated cost), reliability (errors
split into *expected refusals* vs *incidents*, LLM retries, limiter decisions),
and the cache. Alert rules are in `deploy/observability/alerts.yml`; the one
worth reading first is `FinVaultComplianceReviewerDown` — when the reviewer is
unreachable it fails closed and blocks every answer, so the service looks
perfectly healthy from outside while reviewing nothing.

Set `FINVAULT_PRICE_*_PER_MILLION` to turn token counters into a spend figure.
Unset, the cost panel reads a flat zero — the honest signal for "not
configured", which beats a guessed rate on a cost dashboard.

### Deployment

```bash
docker build -t finvault:0.1.0 .

helm install finvault deploy/helm/finvault \
  --set image.repository=your-registry/finvault \
  --set secrets.existingSecret=finvault-secrets
```

The chart refuses to render without a secret source — a shipped default JWT
secret is a shipped vulnerability. Pods run non-root with a read-only root
filesystem; the master key is written by an init container to a `tmpfs` volume
so it never touches a disk that outlives the pod. The HPA scales on CPU only:
memory here is dominated by the resident embedding model, so scaling on it would
add replicas that each allocate another copy of the same model.

**Back up the master key before the first ingest.** It wraps every per-chunk
data key; without it, every stored document is permanently unreadable.

### CI (`.github/workflows/ci.yml`)

Four jobs: `ruff check` + `ruff format --check`; pytest on 3.11 and 3.12; an
integration job with Postgres and Redis service containers (the only place the
RLS tests actually execute — a policy never run against a real database is a
policy nobody has verified, and the same job applies the migrations to a
scratch database and runs `alembic check` for model/migration drift); and an
advisory dependency audit.

## Key design decisions

- **Local embeddings by default** — the embedding step never sends document or query text to a third party. Swappable via the `EmbeddingProvider` interface.
- **Envelope encryption**, not "encrypt the disk" — every chunk has its own data key, wrapped by a master key held behind a `KeyProvider` interface. Dev default is a local file-backed key; a KMS/Vault-backed `KeyProvider` drops in with no other code changes.
- **Manual tool-use loop**, not a third-party agent framework — every request/response and every tool call is directly inspectable and audit-loggable, with no framework silently routing financial data elsewhere.
- **Isolation is enforced twice, by different mechanisms** — every query filters on `org_id`, *and* PostgreSQL Row Level Security independently refuses to return another tenant's rows. The application filter is correct today; the policy is what keeps it correct after the next query someone writes.
- **The cache key encodes clearance, not just tenancy** — because retrieval is clearance-filtered, an answer cached for a compliance officer is not the answer an analyst should receive. Getting this wrong is a disclosure, not a slowdown, so `scoped_key()` makes the role structurally impossible to omit.
- **The rate limiter fails open; everything else fails closed** — a deliberate, documented exception. Refusing every request because a counter is unreachable would make the limiter the denial of service it exists to prevent.
- **Data classification is the real access-control primitive** — it gates both who can retrieve a chunk and whether that chunk's content is ever allowed to reach the LLM at all.
- **The agent loop fails closed on model unreliability, not just outright errors** — smaller/free-tier models occasionally return a genuinely empty turn (no text, no tool call) or an HTTP 200 with a malformed/empty `choices` field instead of raising. `agents/base.py` retries an empty turn with a nudge a bounded number of times, and treats a malformed response the same as any other request failure — both cases raise a typed `AgentExecutionError` (or its `UpstreamProtocolError` subclass — see `errors.py`) that `Orchestrator.handle()` (and a route-level backstop in `api/routes.py`) turn into a clean, non-crashing response rather than a raw 500.
- **Provider promises are verified, not trusted** — `tool_choice` is the API-level guarantee that retrieval happens before an answer is composed, and several providers ignore it silently. The loop therefore treats it as a hint and runs the required tool itself when the first turn comes back as plain text, because the failure it prevents (an answer with no retrieval, hence no citations, hence nothing for citation verification to check) is a grounding failure that looks clean from the outside.
- **A 429 is not a 5xx** — rate limits get a separate, more patient retry budget than infra failures, because one is the upstream saying "wait and this succeeds" and the other is it saying "this is broken". Sharing one budget between them meant a per-minute quota aborted whole queries mid-flight.

## Known limitations / roadmap

**LLM calls go through OpenRouter, not Anthropic directly.** This was a deliberate change from the original design (see git history / project notes) to make the system runnable without a direct Anthropic API account. OpenRouter is an additional third party in the request path for the one step that was already documented as leaving the system boundary (step 4 above) — it sees whatever content reaches that step (already filtered by classification/ACL/externalization policy), and its own data-handling policy applies to that traffic. If your threat model requires calling Anthropic directly with no intermediary, swap the `OpenAI(base_url=...)` client construction in `agents/base.py` and `agents/compliance_agent.py` for the native `anthropic` SDK — the tool-loop logic is the only part that's provider-shaped; everything upstream of it (encryption, ACL, classification) is unaffected either way.

**The model you configure is part of the security boundary, not just a quality knob.** The Orchestrator uses `tool_choice` to guarantee `search_documents` runs before any answer is composed. Several providers accept that parameter and ignore it with no error — the model answers from nothing, the answer carries zero citations, and because `ComplianceAgent` only verifies citations when some exist, an ungrounded answer is reviewed as clean. `agents/base.py` closes this by running the required tool itself when the model skips it (counted by `finvault_forced_tool_synthesized_total`), so a badly-behaved model degrades *search quality* rather than grounding — but a sustained non-zero counter means you should change models. `FINVAULT_MODEL` defaults to `minimax/minimax-m2.7:free`, a free model verified against real requests to honor forcing; `.env.example` carries the full probe table, including which slugs are paid and which reject `tool_choice` outright. Verify any replacement the same way — OpenRouter's advertised `supported_parameters` claims support for models whose upstream provider does not implement it.

**Pin the route explicitly.** `FINVAULT_LLM_ROUTE` (`openrouter` | `moonshot` | `auto`) states which upstream to use. `auto` infers it from whichever API key is non-empty, which means adding a second provider's key silently redirects every LLM call — and since each route names models differently (`moonshotai/kimi-k2.5` vs bare `kimi-k3`), `FINVAULT_MODEL` then no longer matches the route it reaches, failing at request time instead of startup.

**Free tiers mean rate limits, and one query is not one request.** A single `/query` fans out into several nested LLM calls (Orchestrator → Retriever's own multi-turn search loop → Analyst → Compliance semantic review), so a per-minute cap is hit mid-question rather than between questions — a 3 RPM allowance cannot serve a query that makes 5–8 sequential calls. A 429 therefore gets its own, far more patient retry budget than a connection error or a 5xx (5 attempts, 4s→30s capped backoff, tracked separately so one does not consume the other's allowance): a rate limit is the upstream saying the same request *will* succeed after a wait, so giving up early is the only wrong move. Exhausting that budget still fails closed like any other LLM failure (`agent_execution_failed`, HTTP 200, no crash). Daily caps (observed: 50 requests/day on a fresh OpenRouter account) are not retryable — raise them with credits, or swap to a paid model for production use.

**Set `FINVAULT_LLM_TIMEOUT_SECONDS` to match your model.** Defaults to 180s. Reasoning models that think before answering routinely exceed a 60s per-request timeout, which turns a slow-but-working model into a stream of timeouts; this is a property of the model you chose, not of FinVault.

**Reasoning ("thinking") models need three ceilings raised, not one.** They are otherwise a poor fit for a fail-closed pipeline, because every ceiling they exceed converts into a *security-shaped* symptom rather than a timeout. Verified end-to-end on `kimi-k3` (Moonshot direct, 2026-09-01): set `FINVAULT_MAX_TOKENS_PER_REQUEST=80000` (one question spent ~48k against the 40000 default and died mid-chain), keep `FINVAULT_COMPLIANCE_REVIEW_MAX_TOKENS` at its 2048 default (the old hardcoded 200 let the reviewer think past its budget and return empty — failing closed, so a correct, fully-cited answer was blocked for "manual review"), and raise `FINVAULT_LLM_TIMEOUT_SECONDS`. Note also that Moonshot's thinking-enabled models (`kimi-k2.6`, `kimi-k2.7-code*`) reject `tool_choice` outright with a 400, so they cannot be used at all — `kimi-k3` is the only usable Kimi slug.

Explicitly out of scope for this pass:

- **Cloud KMS/Vault adapters** — `LocalKeyProvider` is the reference implementation; production deployments should implement `KeyProvider` against AWS KMS / HashiCorp Vault.
- **Multi-tenant hardening** — org isolation is enforced at the query/ACL layer, but this hasn't been hardened against a determined multi-tenant threat model (e.g. side channels, noisy-neighbor resource isolation).
- **Managed vector-DB adapter** — Qdrant is self-hosted via Docker; a managed Qdrant Cloud (or alternative) adapter is a config change behind the existing `VectorStore` interface, not built here.
- **Fine-grained field-level encryption** — encryption is at chunk granularity, not per-field within a chunk.
- **Embedding inversion** — embeddings of encrypted chunks are stored in plaintext-vector form to enable similarity search; this is standard practice, but note that vector embeddings can in principle leak some information about their source text (embedding-inversion attacks). True encrypted search is a substantially harder problem and out of scope here.
- **Agent-to-agent context fidelity** — the Orchestrator forwards retrieved context to the Analyst agent as a string it composes itself, rather than raw pass-through; the untrusted-content wrapping may not survive that hop verbatim. The Compliance agent's final review is the backstop regardless.
- **PostgresAuditLog concurrency** — correct for a single-writer/low-concurrency deployment; a high-throughput multi-writer deployment should serialize appends (dedicated writer process or DB advisory lock) to avoid a race on `prev_hash`.
- **Frontend auth is dev-only** — `frontend/index.html` calls the same `/auth/token` dev-login endpoint as the curl examples (any username/role/org gets a token, no password). Fine for local development and demos; replace with real credential verification before any non-local deployment (see the note in `api/routes.py`).
- **Multi-agent-of-agents, cloud deployment manifests** — not built this pass.

**Next phase (per the project's stated direction):** once this foundation is proven out, the agent scaffolding here becomes the base for a broader agent operating system — more agent types and workflows beyond finance RAG, built on the same `Agent`/tool/guardrail primitives.

## License

[MIT](LICENSE) — Copyright (c) 2026 Krishna-web-hub.
