"""Deployment-agnostic configuration.

Every infra dependency (key management, vector store, database) is read from
here as plain connection info — the concrete provider is chosen in the
respective module (security/encryption.py, retrieval/vector_store.py) behind
an interface, so swapping a local Docker service for a managed one in
production means changing environment variables, not code.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM — routed through Hugging Face Serverless / Router Inference API or OpenRouter
    llm_api_key: str | None = None
    llm_base_url: str | None = None

    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    kimi_api_key: str | None = None
    moonshot_api_key: str | None = None
    moonshot_base_url: str = "https://api.moonshot.ai/v1"
    # Which upstream to talk to, stated rather than inferred. The properties
    # below originally guessed the route from whichever key happened to be
    # non-empty, which meant that merely *pasting* a Moonshot key silently
    # moved every LLM call off OpenRouter — and since the two use different
    # model-name conventions ("moonshotai/kimi-k2.5" vs bare "kimi-k3"), the
    # slug in FINVAULT_MODEL would then be wrong for the route it reached,
    # failing at request time rather than at startup.
    # "auto" preserves the original inference for existing deployments;
    # "openrouter"/"moonshot" pin it explicitly.
    finvault_llm_route: str = "auto"
    # Per-request HTTP timeout for LLM calls. Was hardcoded at 60s, which is
    # below the single-call latency of a reasoning model that thinks before
    # answering (kimi-k3 direct exceeded it routinely), turning a slow-but-
    # working model into a stream of timeouts. Configurable because the right
    # value is a property of the chosen model, not of this codebase.
    # 180, not the old 60: the shipped default model must work out of the
    # box, and the code default has to agree with .env.example rather than
    # leave a clone slower-than-the-template. The cost is that a genuinely
    # hung upstream now stalls one request for 3 minutes instead of 1 —
    # acceptable, because a timeout that fires on a *working* call is a
    # false failure, while a slow hang is only slow.
    finvault_llm_timeout_seconds: float = 180.0
    # Default is a free model *verified* to honor tool_choice="required"
    # (probe table in .env.example). The model default is not a neutral
    # convenience here: a model that ignores tool forcing makes the
    # Orchestrator answer without retrieving, and base.py's fallback then
    # searches on a blunter query than the model would have written. Any
    # replacement default must be probed the same way first.
    finvault_model: str = "minimax/minimax-m2.7:free"
    # Per-agent model specialization (falls back to finvault_model if unset)
    finvault_orchestrator_model: str | None = None
    finvault_retriever_model: str | None = None
    finvault_analyst_model: str | None = None
    finvault_compliance_model: str | None = None

    @property
    def orchestrator_model(self) -> str:
        return self.finvault_orchestrator_model or self.finvault_model

    @property
    def retriever_model(self) -> str:
        return self.finvault_retriever_model or self.finvault_model

    @property
    def analyst_model(self) -> str:
        return self.finvault_analyst_model or self.finvault_model

    @property
    def compliance_model(self) -> str:
        return self.finvault_compliance_model or self.finvault_model

    finvault_max_tokens: int = 4096

    # Ceiling for ComplianceAgent's semantic-review call. Its own setting,
    # not finvault_max_tokens, because the review is a one-word verdict and
    # was capped at 200 on that basis — correct for a model that answers
    # directly, wrong for one that thinks first. A reasoning model spends
    # this budget on reasoning and gets truncated (finish_reason "length")
    # before emitting the verdict, so content comes back empty and the
    # reviewer fails closed: a correct answer blocked for "manual review"
    # because the reviewer was never given room to speak. Observed live on
    # kimi-k3.
    #
    # Raising it is close to free — max_tokens is a ceiling, not a spend, so
    # a non-thinking model still emits its ~5 tokens and is billed for those.
    finvault_compliance_review_max_tokens: int = 2048

    @property
    def effective_api_key(self) -> str | None:
        if self.llm_api_key:
            return self.llm_api_key
        if self.finvault_llm_route == "openrouter":
            return self.openrouter_api_key
        if self.finvault_llm_route == "moonshot":
            return self.kimi_api_key or self.moonshot_api_key
        return self.hf_token or self.kimi_api_key or self.moonshot_api_key or self.openrouter_api_key

    @property
    def effective_base_url(self) -> str:
        if self.llm_base_url:
            return self.llm_base_url
        if self.finvault_llm_route == "openrouter":
            return self.openrouter_base_url
        if self.finvault_llm_route == "moonshot":
            return self.moonshot_base_url
        if self.hf_token or (self.llm_api_key and "hf" in self.llm_api_key.lower()):
            return "https://router.huggingface.co/v1"
        if self.kimi_api_key or self.moonshot_api_key:
            return self.moonshot_base_url
        return self.openrouter_base_url

    # Storage — non-default ports (see docker-compose.yml): keeps FinVault's
    # data plane on its own containers rather than sharing another project's
    # Qdrant/Postgres that may already be running on the standard ports.
    qdrant_url: str = "http://localhost:6350"
    qdrant_collection: str = "finvault_chunks"
    # Empty for a local/self-hosted Qdrant, which has no auth. Required by
    # Qdrant Cloud. Kept here rather than embedded in qdrant_url because it
    # is a credential: it belongs in the deployment's secret store, and a
    # URL is the kind of value that ends up in logs (api/main.py logs
    # qdrant_url at startup).
    qdrant_api_key: str | None = None
    postgres_dsn: str = "postgresql://finvault:finvault_dev_only@localhost:5433/finvault"

    # Security
    finvault_master_key_path: Path = Path(".secrets/master.key")
    finvault_jwt_secret: str = "dev-only-change-me"

    # Embeddings & Hugging Face
    finvault_embedding_model: str = "BAAI/bge-small-en-v1.5"
    hf_token: str | None = None

    # Cross-encoder reranking — a second, optional relevance signal fused
    # (via reciprocal rank fusion) with vector similarity and BM25 in
    # retrieval/retriever.py. Off by default: unlike BM25, a cross-encoder
    # does one neural forward pass per retrieved candidate, so enabling it
    # is a deliberate latency/quality trade-off, not a free addition.
    finvault_enable_cross_encoder_rerank: bool = False
    finvault_cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Cross-agent token budget — a single ceiling shared across the whole
    # Orchestrator -> Retriever -> Analyst chain for one request (see
    # agents/base.py's TokenBudget), so a runaway loop anywhere in that
    # chain fails closed instead of accumulating unbounded cost. Doubled
    # from the original 20000 after a real repro (a vague question against
    # a CSV-shaped document) showed the retriever alone burning 23k tokens
    # across a handful of reformulated searches before finding nothing
    # useful — 20000 left no room to even finish that attempt, let alone
    # reach the analyst. See finvault_retriever_max_iterations below, which
    # addresses the actual thrashing; this just gives legitimate multi-hop
    # retrieval enough room to complete.
    finvault_max_tokens_per_request: int = 40000

    # Cap on the retriever agent's own reformulate-and-search-again loop
    # (agents/retriever_agent.py). Went 3 -> 5 -> 8 by live repro, each step
    # catching the previous value cutting the agent off mid-task ("exceeded
    # max_iterations without finishing") rather than actually preventing
    # thrashing — one search-then-reformulate cycle alone needs a tool-call
    # iteration plus a final text-response iteration, so a tight cap mostly
    # just breaks legitimate multi-round retrieval instead of stopping
    # runaway cost. Landed back on the generic Agent default (8, see
    # agents/base.py) now that the real cost driver — raw CSV row dumps
    # instead of a compact summary (ingestion/tabular.py) — is fixed:
    # iterations against a small summary are cheap, so affording more of
    # them to let a genuinely ambiguous question converge is the right
    # trade now, not the risk it was against a 60KB raw chunk.
    finvault_retriever_max_iterations: int = 8

    # Prompt caching — best-effort only. FinVault talks to models through the
    # OpenAI-compatible chat completions surface (see agents/base.py), not
    # Anthropic's native Messages API, so there is no guaranteed cache_control
    # support here — see the caching note in agents/base.py before enabling.
    # Off by default so no request shape changes unless explicitly opted in.
    finvault_enable_prompt_caching: bool = False

    # Externalization policy — classification tiers allowed to leave the
    # system boundary in an LLM prompt. Kept as a plain string and parsed
    # into a list so it round-trips cleanly through .env.
    finvault_allowed_external_classifications: str = "public,internal,confidential"

    # Observability — see observability.py. `json` is the deployment
    # default because the fields it emits (request_id, error_code, actor)
    # are only queryable if something indexes them; `text` is the readable
    # console format for local work. DEBUG is safe to enable: no log call
    # site in this codebase logs document plaintext or decrypted content.
    finvault_log_level: str = "INFO"
    finvault_log_format: str = "json"

    # --- Redis cache (see cache.py) ---
    # Empty means "no Redis": the process falls back to an in-memory cache,
    # which is correct but per-replica, so a multi-replica deployment gets a
    # correspondingly lower hit rate. Never a hard dependency — an
    # unreachable Redis degrades performance, never availability.
    finvault_redis_url: str = ""
    finvault_redis_timeout_seconds: float = 0.25

    # Cached answers for POST /query. Short by design: the corpus-generation
    # counter (cache.py) already retires an org's answers the moment anything
    # is ingested, so this TTL only bounds staleness from changes the counter
    # does not see — a role's clearance being changed in the identity
    # provider, say.
    finvault_enable_query_cache: bool = True
    finvault_query_cache_ttl_seconds: int = 300

    # Embeddings are a pure function of (model, text), so they are cached far
    # longer than answers. The key is HMAC'd, so a cache entry does not let a
    # reader confirm a guessed document (see cache.py).
    finvault_enable_embedding_cache: bool = True
    finvault_embedding_cache_ttl_seconds: int = 604800  # 7 days

    # --- Rate limiting (see api/rate_limit.py) ---
    # Two windows per identity: a burst allowance and a sustained hourly cap.
    # /query fans out into several LLM calls, so the limit protects real cost
    # and upstream quota, not just CPU.
    finvault_enable_rate_limit: bool = True
    finvault_rate_limit_per_minute: int = 20
    finvault_rate_limit_per_hour: int = 200
    # Unauthenticated requests are limited per-IP, and far more tightly:
    # there is no account behind them to hold accountable.
    finvault_rate_limit_anonymous_per_minute: int = 10

    # --- Row Level Security (see security/rls.py) ---
    # Off by default because turning it on changes what queries return: with
    # policies installed, any code path that reads a tenant table without an
    # org in context sees zero rows. That is the correct, fail-closed
    # direction, but it must be switched on deliberately after confirming
    # every such path declares its scope — not inherited silently by an
    # existing deployment on upgrade.
    finvault_enable_rls: bool = False

    # Whether the application creates its own tables at startup.
    #
    # True is right for local development, where one account owns everything.
    # It is wrong wherever RLS is enabled, because the two requirements
    # conflict: creating tables needs DDL rights and makes the creator their
    # owner, while enforcing policies requires connecting as a role that is
    # NOT the owner and has no way to exempt itself. Set this False in any
    # deployment with RLS on and run schema creation as a separate migration
    # step under the owning role (the Helm chart ships a pre-install Job that
    # runs `finvault-migrate` to do exactly that).
    #
    # Note the asymmetry: create_all can only add missing tables, so leaving
    # this True in a long-lived database means schema changes land only for
    # whoever creates it fresh. Migrations are the path that alters an
    # existing one — see finvault/migrate.py.
    finvault_auto_create_schema: bool = True

    # --- Metrics (see metrics.py) ---
    # GET /metrics is unauthenticated by design — Prometheus scrapes it, and
    # it exposes counters and latencies, never document content or identity.
    # Restrict it at the network layer (a NetworkPolicy, or the Helm chart's
    # ingress excluding the path), not with a token the scraper would have to
    # hold.
    finvault_enable_metrics: bool = True

    # Per-million-token rates for the configured model, used to turn token
    # counters into a spend metric (metrics.py). Zero by default because
    # there is no correct default: the rate depends on the model behind
    # FINVAULT_MODEL and on your provider's current pricing, and a guessed
    # number on a cost dashboard is worse than an obviously-unset zero.
    finvault_price_input_per_million: float = 0.0
    finvault_price_output_per_million: float = 0.0

    # Ingestion safety limits
    finvault_max_upload_size_mb: int = 20
    finvault_max_chunks_per_document: int = 1000

    @property
    def allowed_external_classifications(self) -> list[str]:
        return [c.strip() for c in self.finvault_allowed_external_classifications.split(",") if c.strip()]


settings = Settings()
