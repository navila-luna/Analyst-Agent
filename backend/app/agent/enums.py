from enum import StrEnum


class AgentAction(StrEnum):
    SEARCH = "SEARCH"
    LIST = "LIST"
    READ = "READ"
    ANSWER = "ANSWER"


class TerminationReason(StrEnum):
    MODEL_ANSWERED = "model_answered"
    NO_EVIDENCE = "no_evidence"
    TOOL_ERROR = "tool_error"
    TOOL_ERROR_RECOVERED = "tool_error_recovered"
    TOOL_LIMIT_REACHED = "tool_limit_reached"
