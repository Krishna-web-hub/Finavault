"""Canvas Models: Pydantic schemas for Knowledge Graph, Live Agent Execution DAG,
and Dynamic Multi-Document Difference Heatmaps.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

GraphEntityType = Literal["company", "metric", "risk", "date", "document", "person"]


class GraphNode(BaseModel):
    id: str
    label: str
    type: GraphEntityType
    classification: str = "public"
    # The "Lineage" half of "Knowledge Graph & Lineage Canvas" — which
    # document this entity was extracted from (see ingestion/extraction.py),
    # so a click-to-inspect UI can show real provenance, not just a label.
    source_document_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    source: str
    target: str
    label: str
    weight: float = 1.0


class KnowledgeGraphData(BaseModel):
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)


ExecutionStepStatus = Literal["running", "success", "vetoed", "error"]


class ExecutionStepNode(BaseModel):
    step_id: str
    agent_name: str
    action: str
    status: ExecutionStepStatus
    duration_ms: float
    payload_preview: str
    veto_reason: str | None = None
    timestamp: float


class ExecutionDAG(BaseModel):
    session_id: str
    steps: list[ExecutionStepNode] = Field(default_factory=list)


class HeatmapCell(BaseModel):
    doc_title: str
    metric_name: str
    value: str
    risk_score: float  # 0.0 (safe) to 1.0 (high risk)
    variance_note: str | None = None


class ComparisonHeatmap(BaseModel):
    metrics: list[str]
    documents: list[str]
    cells: list[HeatmapCell]
