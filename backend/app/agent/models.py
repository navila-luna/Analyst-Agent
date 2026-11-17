from dataclasses import dataclass, field
from typing import Any, Literal

from app.agent.config import AgentConfig
from app.agent.enums import AgentAction, TerminationReason
from app.document_repository import DEFAULT_CHUNKS_PER_READ

@dataclass(frozen=True)
class Evidence:
    text: str
    doc_name: str
    chunk_index: int
    similarity: float | None
    source_query: str
    source_method: Literal["search", "document_read"] = "search"
    
@dataclass
class AgentState:
    question: str
    config: AgentConfig = field(default_factory=AgentConfig)
    searches: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    available_documents: list[dict[str, Any]] = field(default_factory=list)
    document_reads: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    evidence_chars: int = 0
    
@dataclass(frozen=True)
class ModelDecision:
    action: AgentAction
    search_query: str | None = None
    doc_name: str | None = None
    start_chunk: int = 0
    max_chunks: int = DEFAULT_CHUNKS_PER_READ
    answer: str | None = None

@dataclass(frozen=True)
class AnalystAnswer:
    text: str
    evidence: list[Evidence]
    searches_performed: int
    termination_reason: TerminationReason
    tool_calls_performed: int = 0

@dataclass(frozen=True)
class TraceEvent:
    run_id: str
    event: str
    question: str
    decision: AgentAction | None = None
    tool_iteration: int | None = None
    search_iteration: int | None = None
    tool_name: str | None = None
    search_query: str | None = None
    result_count: int | None = None
    results: list[dict[str, Any]] | None = None
    termination_reason: TerminationReason | None = None

@dataclass(frozen=True)
class ToolRequest:
    name: str
    arguments: dict[str, Any]
