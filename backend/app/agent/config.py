from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    max_tool_calls: int = 8
    max_evidence_characters: int = 40_000
