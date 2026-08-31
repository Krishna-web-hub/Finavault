# FinVault — Full-Stack Engineering Roadmap
### From Foundations to Production-Grade Multi-Agent AI

> **How to read this document**: Each of the 9 layers maps theory from the roadmap to actual, working code in this repository. Where FinVault already implements something, the relevant file is cited. Where a gap exists, actionable next steps are given. Every flow diagram shows real data movement through real code.

---

## Master Flow: How All 9 Layers Compose Together

```mermaid
flowchart TD
    U(["User / Client Browser"])

    subgraph L1["Layer 1: Foundations — Algorithms & OS"]
        DS["Data Structures: Arrays, Hash Maps, Trees"]
        OS["OS: Processes, Threads, Memory"]
        NET["Networking: TCP/IP, HTTP, DNS"]
    end

    subgraph L2["Layer 2: Language & Engineering Habits"]
        PY["Python Deep: OOP, Dataclasses, Pydantic"]
        TEST["Testing: 238 tests across 29 files"]
    end

    subgraph L3["Layer 3: Databases"]
        PG[("PostgreSQL: SQL, ACID, Indexing")]
        QDRANT[("Qdrant: Vector / NoSQL")]
        REDIS["Redis / Caching (Roadmap)"]
    end

    subgraph L4["Layer 4: Backend & APIs"]
        FASTAPI["FastAPI: REST, SSE, Streaming"]
        AUTH["JWT Auth: OAuth2, RBAC, Role Claims"]
    end

    subgraph L5["Layer 5: Frontend"]
        FE["Single-Page UI: HTML/CSS/JS"]
    end

    subgraph L6["Layer 6: Distributed Systems"]
        ORCH["Orchestrator Agent: Task Delegation"]
        RET["Retriever Agent: Multi-Hop RAG"]
        ANA["Analyst Agent: Financial Reasoning"]
        COMP["Compliance Agent: Deterministic Veto"]
        BUDGET["Token Budget: Cross-Agent Ceiling"]
    end

    subgraph L7["Layer 7: Cloud & DevOps"]
        DOCKER["Docker Compose: Qdrant + Postgres"]
        OBS["Observability: Audit Log + Prometheus (Roadmap)"]
    end

    subgraph L8["Layer 8: SaaS Security"]
        ENC["AES-256-GCM Encryption per-chunk"]
        ACL["RBAC + Classification ACL"]
        AUDIT["Hash-Chained Audit Log SHA-256"]
        REVIEW["Human Review Queue HITL"]
    end

    subgraph L9["Layer 9: Data & ML Layer"]
        EMBED["Local Embeddings bge-small-en-v1.5"]
        BM25["Hybrid Reranking: BM25 + RRF"]
        CLASS["Classification Suggester Zero-Shot"]
        GRAPH["Knowledge Graph: Entity + Relationship"]
    end

    U -->|"HTTPS + Bearer JWT"| L4
    L4 --> L6
    L6 --> L3
    L3 --> L8
    L8 --> L9
    L9 -->|"Redacted, Compliant Answer"| U
    L1 -.->|"Theoretical Foundation"| L2
    L2 -.->|"Code Quality"| L4
    L7 -.->|"Infrastructure"| L3
```

---

## Layer 1 — Foundations: Algorithms, OS, Networking

> *MIT 6.006 (Algorithms), MIT 6.033 (OS), Stanford CS144 (Networking)*

**Data Structures in FinVault:**
- **Hash Maps (`dict`)**: O(1) lookups for `ROLE_RANK`, `CLASSIFICATION_RANK`, `_PII_PATTERNS`, `_INJECTION_PATTERNS`, tool registry in `agents/base.py`.
- **Sorted Lists**: `sorted()` in `retriever.py` for chunk reordering by `chunk_index` and RRF scores.
- **Graphs**: `graph_nodes_table` + `graph_edges_table` in `db.py` model a real entity-relationship graph.
- **Arrays + BM25 Matrix**: `rank_bm25.BM25Okapi` uses an inverted index term-document matrix internally.

```mermaid
flowchart LR
    Q["User Query"] --> VEC["Embed to float vector\nnumpy array O(d)"]
    VEC --> KNN["Qdrant ANN Search\nHNSW Graph Index O(log n)"]
    KNN --> ACL["ACL Filter\nHash Map lookup O(1) per chunk"]
    ACL --> BM25_["BM25 Okapi\nInverted Index"]
    BM25_ --> RRF["Reciprocal Rank Fusion\nSort O(n log n)"]
    RRF --> TOP_K["Top-K RetrievedChunks\nreturned to Agent"]
```

**OS Concepts:**
- `uvicorn` runs FastAPI in an **async I/O event loop** (single process, multiple coroutines).
- `/query/stream` route uses a **background thread** + `queue.Queue` to bridge sync SQLAlchemy with async SSE.

**Networking:**
- FastAPI exposes **HTTP/1.1 REST** endpoints.
- `/query/stream` uses **Server-Sent Events (SSE)** — streaming chunks as `event:` / `data:` frames.

---

## Layer 2 — One Language Deep + Engineering Habits

> *Python (deep), git properly, testing (unit + integration)*

```mermaid
classDiagram
    class KeyProvider {
        <<Abstract>>
        +wrap_key(data_key bytes) bytes
        +unwrap_key(wrapped_key bytes) bytes
    }
    class LocalKeyProvider {
        -_kek bytes
        +wrap_key()
        +unwrap_key()
    }
    class AuditLog {
        <<Abstract>>
        +append(actor, action, resource, details) AuditEntry
        +entries() list
        +verify_chain() bool
    }
    class InMemoryAuditLog { }
    class PostgresAuditLog { }
    KeyProvider <|-- LocalKeyProvider
    AuditLog <|-- InMemoryAuditLog
    AuditLog <|-- PostgresAuditLog
```

**Testing — 238 Tests Across 29 Files:**
- `test_access_control.py` — Unit: RBAC clearance logic
- `test_encryption.py` — Unit: AES-256-GCM, AAD binding
- `test_injection_corpus.py` — Red-team: 30+ injection phrasings
- `test_orchestrator.py` — Integration: full agent pipeline with fakes
- `test_postgres_stores.py` — Integration: real Postgres

---

## Layer 3 — Databases

> *SQL + relational design, NoSQL tradeoffs, Caching (Redis)*

```mermaid
erDiagram
    DOCUMENTS {
        string id PK
        string org_id
        string title
        string classification
        datetime created_at
    }
    AUDIT_LOG {
        int seq PK
        string actor
        string action
        string prev_hash
        string entry_hash
    }
    SESSIONS {
        int id PK
        string session_id
        string user_id
        string question
        string answer
    }
    REVIEW_QUEUE {
        string id PK
        string org_id
        string user_id
        string status
    }
    GRAPH_NODES {
        string id PK
        string org_id
        string type
        string label_hash
        json label_encrypted
    }
    GRAPH_EDGES {
        string id PK
        string org_id
        string source_node_id
        string target_node_id
        json relation_encrypted
    }

    DOCUMENTS ||--o{ GRAPH_NODES : "source_document_id"
    GRAPH_NODES ||--o{ GRAPH_EDGES : "source / target"
    DOCUMENTS ||--o{ REVIEW_QUEUE : "blocks produce"
    DOCUMENTS ||--o{ AUDIT_LOG : "every action logged"
```

**Indexing strategy:** `org_id` indexed on every table for per-org scans, `label_hash` for O(log n) entity deduplication, `status` on `review_queue` for pending items lookup.

**ACID:** `PostgresAuditLog.append()` executes read-then-insert inside a single `engine.begin()` transaction — hash chain update is atomic.

**Redis (Roadmap):**
```mermaid
flowchart LR
    Q["User Query"] --> CACHE{"Redis Cache\nTTL 60s\nKey: SHA256(org_id + query)"}
    CACHE -->|"HIT"| RESP["Return cached OrchestratorResult"]
    CACHE -->|"MISS"| AGENT["Full Multi-Agent RAG Pipeline"]
    AGENT --> STORE["Store in Redis"]
    STORE --> RESP
```

---

## Layer 4 — Backend & APIs

> *REST, FastAPI, OAuth2, JWT, Security*

```mermaid
flowchart TD
    subgraph Auth
        A1["POST /auth/token\nJWT access_token"]
    end
    subgraph Documents
        D1["POST /documents\nmultipart/form-data + classification"]
        D2["GET /documents"]
        D3["POST /documents/compare\nComparisonHeatmap"]
    end
    subgraph Query
        Q1["POST /query\nQueryResponse{answer, blocked, citations}"]
        Q2["POST /query/stream\nSSE text/event-stream"]
    end
    subgraph ReviewQueue
        R1["GET /review-queue"]
        R2["POST /review-queue/id/resolve\nCOMPLIANCE_OFFICER only"]
    end
    subgraph KnowledgeGraph
        G1["GET /graph\nKnowledgeGraphData"]
    end

    User["Client JWT Bearer"] --> Auth
    Auth -->|"Token"| Documents
    Auth -->|"Token"| Query
    Auth -->|"Token"| ReviewQueue
    Auth -->|"Token"| KnowledgeGraph
```

### JWT Auth Flow

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI as FastAPI routes.py
    participant Auth as auth.py
    participant Downstream

    Client->>FastAPI: POST /auth/token {username, role, org_id}
    FastAPI->>Auth: create_access_token(user)
    Auth-->>FastAPI: JWT HS256 signed
    FastAPI-->>Client: {access_token}

    Client->>FastAPI: POST /query {Authorization: Bearer token}
    FastAPI->>Auth: get_current_user(credentials)
    Auth->>Auth: jwt.decode(token, secret)
    Auth-->>FastAPI: User{id, username, role, org_id}
    FastAPI->>Downstream: All decisions use this User object
    Note over Downstream: Never trusts client-supplied org_id or role
```

**OWASP Top 10 Coverage:**

| OWASP | Mitigation in FinVault |
|---|---|
| A01 Broken Access Control | `check_clearance()` + `require_same_org()` on every retrieval |
| A02 Cryptographic Failures | AES-256-GCM per-chunk, no plaintext at rest, local embeddings |
| A03 Injection | AST-walker arithmetic, `wrap_untrusted_content`, injection scanner |
| A04 Insecure Design | Fail-closed; Compliance is Python control flow, not a callable tool |
| A07 Auth Failures | JWT verified on every route; malformed tokens = HTTP 401 |
| A09 Logging Failures | Every decision logged to hash-chained audit trail |

---

## Layer 5 — Frontend

> *HTML/CSS/JS, enough to ship*

```mermaid
stateDiagram-v2
    [*] --> LoginScreen : app loads
    LoginScreen --> Authenticated : POST /auth/token stores JWT
    Authenticated --> UploadDoc : user selects file + classification
    UploadDoc --> Authenticated : POST /documents
    Authenticated --> QueryChat : user types question
    QueryChat --> StreamingResponse : POST /query/stream SSE
    StreamingResponse --> ShowExecutionCanvas : agent steps arrive as events
    ShowExecutionCanvas --> ShowAnswer : compliance-approved answer
    ShowAnswer --> QueryChat : next question session_id carried forward
    ShowAnswer --> ReviewQueue : if blocked enqueued for compliance officer
```

---

## Layer 6 — Distributed Systems & Multi-Agent Architecture

> *CAP theorem, event-driven, microservices vs monolith, message queues*

### Four-Agent Delegation Pattern

```mermaid
sequenceDiagram
    participant API as FastAPI Route
    participant ORC as Orchestrator Agent
    participant RET as Retriever Agent
    participant ANA as Analyst Agent
    participant CMP as Compliance Agent
    participant AUDIT as Audit Log
    participant LLM as LLM OpenRouter

    API->>+ORC: orchestrator.handle(question, user)
    ORC->>AUDIT: log query action
    ORC->>+LLM: system=SYSTEM_PROMPT user=question
    LLM-->>ORC: tool_call search_documents

    ORC->>+RET: retriever_agent.run(query)
    RET->>RET: retrieve() ACL filter decrypt BM25 rerank
    RET->>RET: wrap_untrusted_content() detect_injection()
    RET-->>-ORC: context_string injection_flags

    ORC->>+ANA: analyst_agent.run_structured(question + context)
    ANA->>ANA: _safe_eval AST no eval()
    ANA-->>-ORC: AnalystAnswer{answer, citations, calculations}

    ORC->>+CMP: compliance_agent.review_output(draft, max_classification)
    CMP->>CMP: enforce_externalization_policy()
    CMP->>CMP: scan_and_redact() verify_citations()
    CMP->>+LLM: semantic review APPROVE or BLOCK
    LLM-->>-CMP: APPROVE
    CMP-->>-ORC: ComplianceVerdict allowed=True

    ORC->>AUDIT: log compliance_review action
    ORC-->>-API: OrchestratorResult{answer, citations, execution_steps}
```

### Fail-Closed Token Budget

```mermaid
flowchart LR
    START["Request Starts"] --> BUDGET["TokenBudget limit=40,000"]
    BUDGET --> ORCH_CALL["Orchestrator LLM call\nbudget.charge(tokens)"]
    ORCH_CALL --> RET_CALL["Retriever LLM call\nbudget.charge(tokens)"]
    RET_CALL --> ANA_CALL["Analyst LLM call\nbudget.charge(tokens)"]

    ORCH_CALL -->|"over 40k"| EXCEEDED["TokenBudgetExceeded"]
    RET_CALL -->|"over 40k"| EXCEEDED
    ANA_CALL -->|"over 40k"| EXCEEDED

    EXCEEDED --> FAIL["blocked=True\nblock_reason=token_budget_exceeded"]
    ANA_CALL -->|"budget OK"| COMP["Compliance Review"]
    COMP --> ANSWER["Final Answer"]
```

---

## Layer 7 — Cloud & DevOps

> *AWS, Docker, Kubernetes, Terraform, CI/CD, Observability*

### Current Infrastructure

```mermaid
flowchart TB
    subgraph Host["Developer Machine / Production VM"]
        UVICORN["uvicorn FinVault FastAPI :8000"]
        subgraph DOCKER["docker-compose network"]
            QDRANT_C["qdrant/qdrant:v1.12.4\nPort 6350:6333 — qdrant_data volume"]
            POSTGRES_C["postgres:16-alpine\nPort 5433:5432 — postgres_data volume"]
        end
        SECRETS[".secrets/master.key\nchmod 600 owner-only"]
    end

    UVICORN --> QDRANT_C
    UVICORN --> POSTGRES_C
    UVICORN --> SECRETS
```

### Production AWS Roadmap

```mermaid
flowchart TD
    CF["CloudFront CDN\nEdge TLS Termination"]
    ALB["Application Load Balancer"]
    ECS["ECS Fargate\nFinVault API auto-scaled"]
    RDS["AWS RDS PostgreSQL\nMulti-AZ encrypted"]
    KMS["AWS KMS\nKEK management"]
    BEDROCK["AWS Bedrock PrivateLink\nor Self-hosted vLLM\nZero data egress"]
    CW["CloudWatch Logs + Alarms"]
    PROM["Prometheus + Grafana\nAgent latency token spend"]

    CF --> ALB --> ECS
    ECS --> RDS
    ECS --> KMS
    ECS --> BEDROCK
    ECS --> CW
    ECS --> PROM
```

---

## Layer 8 — SaaS-Specific Security Mechanics

> *Multi-tenancy, OWASP Top 10, rate limiting, PCI-DSS awareness*

```mermaid
flowchart TD
    DOC["Document uploaded\nby Analyst CONFIDENTIAL"]

    DOC --> C1{"Size under 20MB?\nchunks under 1000?"}
    C1 -->|"No"| REJECT1["Rejected config limits"]
    C1 -->|"Yes"| C2{"File type valid?"}
    C2 -->|"Unsupported"| REJECT2["Rejected loader"]
    C2 -->|"OK"| EMBED["Embed locally\nbge-small-en on-device"]
    EMBED --> ENCRYPT["AES-256-GCM\nper-chunk DEK wrapped by KEK"]
    ENCRYPT --> STORE["ciphertext only\nno plaintext in Qdrant or Postgres"]
    STORE --> AUDIT1["Audit log entry written"]

    QUERY["User Query\nViewer same org"] --> JWT_CHECK{"JWT valid?\norg_id matches?"}
    JWT_CHECK -->|"No"| HTTP401["HTTP 401"]
    JWT_CHECK -->|"Yes"| ACL_CHECK{"Role >= required min_role?"}
    ACL_CHECK -->|"No"| SILENT_DROP["Chunk silently dropped\nno 403 no information leak"]
    ACL_CHECK -->|"Yes"| EXT_CHECK{"Classification in\nexternalization allowlist?"}
    EXT_CHECK -->|"RESTRICTED No"| WITHHELD["Text withheld from LLM\nplaceholder shown"]
    EXT_CHECK -->|"CONFIDENTIAL Yes"| LLM_SAFE["Wrapped as untrusted content\nsent to LLM"]
    LLM_SAFE --> PII_SCAN["Compliance Agent:\nPII scan + citation verify\n+ semantic LLM review"]
    PII_SCAN -->|"Clean"| REDACTED_ANSWER["Redacted answer to User"]
    PII_SCAN -->|"Flagged"| REVIEW_Q["Human Review Queue\ncompliance officer"]
```

---

## Layer 9 — Data & ML Layer

> *ETL, embeddings, recommendation, ranking, fraud detection, A/B testing*

```mermaid
flowchart TD
    subgraph Ingestion["Ingestion Pipeline ETL"]
        RAW["Raw File PDF DOCX CSV MD TXT"]
        LOAD["Loader plain text extraction"]
        CHUNK["Chunking paragraph-aware 200-word overlap"]
        CLASSIFY_S["ClassificationSuggester\nCosine similarity to exemplar centroids"]
        EMBED_I["Local Embedding bge-small-en-v1.5"]
        ENCRYPT_I["Envelope Encrypt AES-256-GCM + AAD"]
        STORE_I["Store: Qdrant vector + encrypted payload\nPostgres doc record + Audit log"]
        EXTRACT["ExtractionAgent\nEntity + Relationship extraction\nentities encrypted in graph tables"]

        RAW --> LOAD --> CHUNK --> CLASSIFY_S
        CHUNK --> EMBED_I --> ENCRYPT_I --> STORE_I
        CHUNK --> EXTRACT --> STORE_I
    end

    subgraph Retrieval["Retrieval & Reranking"]
        Q_EMBED["Query Embedding bge-small-en on-device"]
        VECT_SEARCH["Vector Search Qdrant HNSW ANN org_id scoped"]
        ACL_FILTER["ACL Filter clearance + org check silent drop"]
        DECRYPT_R["Decrypt chunks EnvelopeEncryptor.decrypt()"]
        HYBRID["Hybrid Reranking\nVector + BM25 + Cross-Encoder\nReciprocal Rank Fusion"]
        TOP_K_R["Top-K RetrievedChunks"]

        Q_EMBED --> VECT_SEARCH --> ACL_FILTER --> DECRYPT_R --> HYBRID --> TOP_K_R
    end

    subgraph Reasoning["Agent Reasoning Layer"]
        INJ_WRAP["wrap_untrusted_content()\nDelimiter escape neutralization"]
        EXT_GATE["Externalization Gate\nRESTRICTED chunks withheld"]
        ORCHESTRATE["Orchestrator Agent\nTask planning + delegation"]
        ANALYSE["Analyst Agent\nFinancial reasoning + AST calculator"]
        CITATION_V["Citation Verification\nverbatim quoted_text grounding"]

        TOP_K_R --> INJ_WRAP --> EXT_GATE --> ORCHESTRATE
        ORCHESTRATE --> ANALYSE --> CITATION_V
    end

    subgraph Output["Output Guard Layer"]
        PII_R["scan_and_redact() Regex PII masking"]
        SEM_R["Semantic LLM Review APPROVE / BLOCK"]
        AUDIT_FINAL["Audit Log hash-chained tamper-evident"]
        USER_FINAL["User sees:\nRedacted Answer + Citations\n+ Execution Canvas + Knowledge Graph"]
        HITL["Human Review Queue if blocked"]

        CITATION_V --> PII_R --> SEM_R
        SEM_R -->|"APPROVE"| AUDIT_FINAL --> USER_FINAL
        SEM_R -->|"BLOCK"| HITL --> AUDIT_FINAL
    end
```

### ML Techniques Used

| Technique | Implementation | File |
|---|---|---|
| Dense Retrieval ANN | HNSW-indexed vector search via Qdrant | `retrieval/vector_store.py` |
| Sparse Retrieval BM25 | `rank_bm25.BM25Okapi` over decrypted candidates | `retrieval/retriever.py` |
| Cross-Encoder Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` optional | `retrieval/reranker.py` |
| Reciprocal Rank Fusion | Combines vector, BM25, cross-encoder signals | `retrieval/retriever.py` |
| Zero-Shot Classification | Cosine similarity to exemplar centroids | `ingestion/classification.py` |
| Named Entity Recognition | LLM-based extraction ExtractionAgent | `ingestion/extraction.py` |
| Deterministic Risk Scoring | Coefficient-of-range variance not LLM-scored | `agents/comparison_agent.py` |
| AST-Sandboxed Arithmetic | Python AST walker no eval | `agents/analyst_agent.py` |

---

## Coverage Matrix: Roadmap vs. FinVault

| Layer | Topic | Status | Gap / Next Step |
|---|---|:---:|---|
| 1 | Data Structures & Big-O | ✅ | — |
| 1 | OS: Threads, Async I/O | ✅ | — |
| 1 | Networking: HTTP, SSE | ✅ | Add TLS termination docs |
| 2 | Python OOP, ABCs, Pydantic | ✅ | — |
| 2 | Testing: Unit + Integration | ✅ 238 tests | Add coverage badge + CI |
| 3 | PostgreSQL: ACID, Indexing | ✅ | Add RLS policies |
| 3 | NoSQL: Vector DB | ✅ | — |
| 3 | Caching: Redis | ❌ | Add query + embedding cache |
| 4 | REST API FastAPI | ✅ | — |
| 4 | JWT Auth + RBAC | ✅ | Connect to real IdP Okta |
| 4 | OWASP Security | ✅ | Add rate limiting middleware |
| 5 | Frontend SPA | ✅ | Upgrade to React for scale |
| 6 | Multi-Agent Orchestration | ✅ 5 agents | — |
| 6 | Event-driven message bus | ✅ ExecutionEventBus | Externalize to Kafka |
| 6 | Token budgeting backpressure | ✅ | — |
| 7 | Docker | ✅ | — |
| 7 | Kubernetes | ❌ | Add K8s manifests / Helm chart |
| 7 | CI/CD | ❌ | GitHub Actions: pytest + lint |
| 7 | Observability | 🔶 Audit log only | Add Prometheus + Grafana |
| 8 | Multi-tenancy isolation | ✅ 5-layer | Add PostgreSQL RLS |
| 8 | Billing / Rate Limiting | ❌ | Stripe + rate-limit middleware |
| 8 | PCI-DSS awareness | ✅ IBAN/CC redaction | — |
| 9 | Local Embeddings ETL | ✅ | — |
| 9 | Hybrid Reranking | ✅ BM25 + Cross-Encoder + RRF | — |
| 9 | Zero-shot Classification | ✅ | Train a real classifier |
| 9 | Knowledge Graph | ✅ | Add graph traversal queries |
| 9 | Deterministic Risk Scoring | ✅ | Extend to time-series trends |
| 9 | A/B Testing Infrastructure | ❌ | Add experiment framework |

---

## Recommended Reading, Mapped to FinVault

| Book / Course | Layer | Seen In FinVault |
|---|---|---|
| MIT 6.006 Introduction to Algorithms | 1 | BM25 term-doc matrix, RRF sort, HNSW graph |
| MIT 6.033 Computer Systems | 1 | uvicorn async event loop, threading in SSE |
| CMU 15-445 Database Systems | 3 | Indexing in `db.py`, ACID in `audit.py` |
| FastAPI Advanced Docs | 4 | SSE streaming, Depends() injection |
| Designing Data-Intensive Applications Kleppmann | 6 | Multi-agent consistency, hash-chained audit log |
| MIT 6.824 Distributed Systems | 6 | Agent failure modes, fail-closed posture |
| AWS Solutions Architect | 7 | Docker isolation, production AWS roadmap |
| OWASP Top 10 | 8 | ACL, injection defense, auth, logging |
| The Little Book of Deep Learning | 9 | Embedding models, BM25, cross-encoder |

---

*Generated: 2026-08-31 | FinVault Phase 1 — 73 Python files, 238 tests, 236 passing*
