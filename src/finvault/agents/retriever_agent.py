"""Retriever agent: RAG search over the encrypted document corpus.

Every chunk it hands back to the caller has already passed the ACL check
(in Retriever.retrieve) and is wrapped as untrusted content with the
injection-defense instruction. It also tracks the highest classification
tier actually surfaced this turn — kept as a second, defense-in-depth gate
the Orchestrator checks before the final answer leaves the system (see
security/guardrails.enforce_externalization_policy) — but the primary
enforcement is here: a chunk whose classification isn't in the
externalization allowlist never has its text included in a tool result at
all, so it never reaches the LLM's prompt in the first place. ACL clearance
(can this user *retrieve* the chunk) and externalization policy (can this
chunk's content ever be sent to the LLM) are independent checks — a user can
be cleared to retrieve `restricted` content without that content being
permitted to leave the system boundary.
"""

from __future__ import annotations

from openai import OpenAI

from finvault.agents.base import Agent, TokenBudget, ToolDefinition
from finvault.config import settings
from finvault.models import CLASSIFICATION_RANK, Classification, User
from finvault.retrieval.retriever import Retriever
from finvault.security.guardrails import INJECTION_DEFENSE_INSTRUCTION, detect_injection_attempt, wrap_untrusted_content

SYSTEM_PROMPT = f"""You are the Retriever agent in a financial-document RAG system.
Your only job is to find and return the most relevant passages from the
document corpus for the user's question, using the search_documents tool.
Call it with focused search queries — reformulate and search again if the
first results are thin or don't answer the question. When you have enough
context, respond with the passages you found, clearly attributed to their
source document. Do not answer the user's underlying question yourself —
that is the Analyst agent's job; your job is retrieval.

{INJECTION_DEFENSE_INSTRUCTION}
"""


class RetrieverAgent:
    def __init__(
        self, *, retriever: Retriever, user: User, model: str | None = None, client: OpenAI | None = None
    ) -> None:
        self._retriever = retriever
        self._user = user
        self._last_injection_flags: list[dict] = []
        self._last_max_classification: Classification = Classification.PUBLIC
        self._last_retrieved_document_ids: set[str] = set()

        def search_documents(input_: dict) -> str:
            query = input_["query"]
            top_k = int(input_.get("top_k", 5))
            hits = self._retriever.retrieve(query, user=user, top_k=top_k)
            if not hits:
                return "No matching passages found."

            parts = []
            for hit in hits:
                # Every ACL-cleared hit counts toward "documents this query
                # touched" — the same population `max_classification_seen`
                # already tracks over, regardless of whether this particular
                # hit's text also clears the externalization allowlist below.
                # This only ever scopes which document_ids the Knowledge
                # Graph query later asks for; GraphRetriever re-applies its
                # own per-node ACL check independently, so it's a relevance
                # filter, not a security boundary.
                self._last_retrieved_document_ids.add(hit.chunk.document_id)
                if CLASSIFICATION_RANK[hit.chunk.classification] > CLASSIFICATION_RANK[self._last_max_classification]:
                    self._last_max_classification = hit.chunk.classification

                # Primary externalization enforcement: a chunk above the
                # allowed tier is withheld here, before its text is ever
                # assembled into a tool result — it never becomes part of a
                # prompt sent to the LLM. (max_classification_seen above is
                # still updated so the Orchestrator's post-hoc check remains
                # a working second gate, not the only one.)
                if hit.chunk.classification.value not in settings.allowed_external_classifications:
                    parts.append(
                        f"[Source: {hit.document_title} | classification: {hit.chunk.classification.value}] "
                        "Content withheld: this classification tier may not be sent to the LLM per the "
                        "externalization policy, even though you are cleared to retrieve it."
                    )
                    continue

                flags = detect_injection_attempt(hit.text)
                if flags:
                    self._last_injection_flags.append(
                        {"document": hit.document_title, "chunk_id": hit.chunk.id, "patterns": flags}
                    )

                parts.append(
                    f"[Source: {hit.document_title} | relevance {hit.score:.2f} | "
                    f"classification: {hit.chunk.classification.value}]\n" + wrap_untrusted_content(hit.text)
                )
            return "\n\n".join(parts)

        self._agent = Agent(
            name="retriever",
            system_prompt=SYSTEM_PROMPT,
            model=model,
            client=client,
            max_iterations=settings.finvault_retriever_max_iterations,
            tools=[
                ToolDefinition(
                    name="search_documents",
                    description="Search the encrypted document corpus for passages relevant to a query.",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Focused search query"},
                            "top_k": {"type": "integer", "description": "Number of passages to retrieve (default 5)"},
                        },
                        "required": ["query"],
                    },
                    handler=search_documents,
                )
            ],
        )

    @property
    def max_classification_seen(self) -> Classification:
        return self._last_max_classification

    @property
    def retrieved_document_ids(self) -> set[str]:
        return self._last_retrieved_document_ids

    def run(self, question: str, *, budget: TokenBudget | None = None) -> tuple[str, list[dict]]:
        self._last_injection_flags = []
        self._last_max_classification = Classification.PUBLIC
        self._last_retrieved_document_ids = set()
        result = self._agent.run(question, budget=budget)
        return result, self._last_injection_flags
