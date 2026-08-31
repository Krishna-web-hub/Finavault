"""ACL-filtered read layer over GraphStore — the Knowledge Graph canvas's
enforcement point, mirroring retrieval/retriever.py exactly.

This is the only place decrypted entity labels/relationship names are
produced. A node whose classification the user isn't cleared for is
dropped silently, not surfaced as a denial — same non-inference posture as
chunk retrieval (whether a matching restricted entity even exists isn't
itself information a lower-clearance user should be able to infer).

An edge is shown only if BOTH endpoint nodes are visible AND the edge's own
classification clears — a dangling edge pointing at a withheld node would
itself leak that the node exists, and a relationship between two otherwise-
visible entities can carry its own, more sensitive classification (see
db.py's comment on graph_edges.classification).
"""

from __future__ import annotations

from finvault.agents.canvas_models import GraphEdge, GraphNode, KnowledgeGraphData
from finvault.models import Classification, User
from finvault.retrieval.graph_store import GraphStore, edge_aad, node_aad
from finvault.security.access_control import check_clearance
from finvault.security.encryption import EnvelopeEncryptor


class GraphRetriever:
    def __init__(self, *, graph_store: GraphStore, encryptor: EnvelopeEncryptor) -> None:
        self._graph_store = graph_store
        self._encryptor = encryptor

    def get_graph(self, *, user: User, document_ids: set[str] | None = None) -> KnowledgeGraphData:
        """`document_ids`: optional relevance filter (e.g. "only entities
        from documents this query actually retrieved") — not a security
        boundary. ACL clearance below is enforced unconditionally either way.
        """
        raw_nodes, raw_edges = self._graph_store.get_nodes_and_edges(org_id=user.org_id)

        visible_nodes: list[GraphNode] = []
        visible_node_ids: set[str] = set()
        for raw in raw_nodes:
            if document_ids is not None and raw.source_document_id not in document_ids:
                continue
            if not check_clearance(user.role, Classification(raw.classification)):
                continue
            label = self._encryptor.decrypt(raw.label_encrypted, aad=node_aad(raw.org_id, raw.type, raw.label_hash))
            visible_nodes.append(
                GraphNode(
                    id=raw.id,
                    label=label,
                    type=raw.type,
                    classification=raw.classification,
                    source_document_id=raw.source_document_id,
                    details=raw.details,
                )
            )
            visible_node_ids.add(raw.id)

        visible_edges: list[GraphEdge] = []
        for raw in raw_edges:
            if raw.source_node_id not in visible_node_ids or raw.target_node_id not in visible_node_ids:
                continue
            if not check_clearance(user.role, Classification(raw.classification)):
                continue
            relation = self._encryptor.decrypt(raw.relation_encrypted, aad=edge_aad(raw.id))
            visible_edges.append(
                GraphEdge(source=raw.source_node_id, target=raw.target_node_id, label=relation, weight=raw.weight)
            )

        return KnowledgeGraphData(nodes=visible_nodes, edges=visible_edges)
