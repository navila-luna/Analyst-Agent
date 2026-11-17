from typing import Any, Protocol

from app.agent.models import (
    AgentState,
    ModelDecision,
    TraceEvent,
)

class AnalystModel(Protocol):
    async def decide(self, state: AgentState, allow_tools: bool) -> ModelDecision: ...

class MCPClient(Protocol):
    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None): ...

class TraceRecorder(Protocol):
    def record(self, event: TraceEvent) -> None: ...