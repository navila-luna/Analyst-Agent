import logging
from uuid import uuid4

from app.agent.config import AgentConfig
from app.agent.constants import INSUFFICIENT_EVIDENCE_MESSAGE
from app.agent.enums import AgentAction, TerminationReason
from app.agent.models import (
    AgentState,
    AnalystAnswer,
    Evidence,
    ModelDecision,
    TraceEvent,
)
from app.agent.protocols import (
    AnalystModel,
    MCPClient,
    TraceRecorder,
)
from app.agent.tool_executor import ToolExecutor
from app.agent.tracing import LoggingTraceRecorder

class AnalystAgent:
    def __init__(
        self,
        model: AnalystModel,
        mcp_client: MCPClient,
        config: AgentConfig | None = None,
        trace_recorder: TraceRecorder | None = None,
    ):
        self.model = model
        self._tool_executor = ToolExecutor(mcp_client)
        self._config = config or AgentConfig()
        self.trace_recorder = trace_recorder or LoggingTraceRecorder()

    async def run(self, question: str) -> AnalystAnswer:
        state = AgentState(question=question, config=self._config)
        run_id = str(uuid4())
        self._trace(TraceEvent(run_id, "run_started", question))

        while True:
            allow_tools = (
                len(state.tool_calls) < state.config.max_tool_calls
                and state.evidence_chars < state.config.max_evidence_characters
            )
            decision = await self.model.decide(state, allow_tools)
            self._trace(TraceEvent(
                run_id,
                "model_decision",
                question,
                decision=decision.action,
                tool_iteration=len(state.tool_calls),
                search_query=decision.search_query,
            ))

            if decision.action == AgentAction.ANSWER:
                has_context = bool(state.evidence or state.available_documents)
                if state.tool_calls and not has_context:
                    answer = INSUFFICIENT_EVIDENCE_MESSAGE
                    reason = TerminationReason.NO_EVIDENCE
                else:
                    answer = decision.answer or INSUFFICIENT_EVIDENCE_MESSAGE
                    reason = (
                        TerminationReason.MODEL_ANSWERED
                        if allow_tools
                        else TerminationReason.TOOL_LIMIT_REACHED
                    )
                return self._finish(run_id, state, answer, reason)

            if not allow_tools:
                return self._finish(
                    run_id,
                    state,
                    INSUFFICIENT_EVIDENCE_MESSAGE,
                    TerminationReason.TOOL_LIMIT_REACHED,
                )

            try:
                await self._execute_tool(run_id, state, decision)
            except Exception:
                logging.getLogger("analyst_agent").exception(
                    "Agent tool failed during %s after %s tool calls",
                    decision.action,
                    len(state.tool_calls),
                )
                if state.evidence:
                    try:
                        fallback = await self.model.decide(state, allow_tools=False)
                    except Exception:
                        logging.getLogger("analyst_agent").exception(
                            "Agent could not produce an answer from existing evidence"
                        )
                    else:
                        if (
                            fallback.action == AgentAction.ANSWER
                            and fallback.answer
                        ):
                            return self._finish(
                                run_id,
                                state,
                                fallback.answer,
                                TerminationReason.TOOL_ERROR_RECOVERED,
                            )
                return self._finish(
                    run_id,
                    state,
                    INSUFFICIENT_EVIDENCE_MESSAGE,
                    TerminationReason.TOOL_ERROR,
                )

    async def _execute_tool(
        self, run_id: str, state: AgentState, decision: ModelDecision
    ) -> None:
        request = self._tool_executor.build_request(decision, state.question)
        name = request.name
        arguments = request.arguments
            
        state.tool_calls.append({"name": name, "arguments": arguments})
        if name == "search_documents":
            state.searches.append(arguments["question"])

        self._trace(TraceEvent(
            run_id,
            "search_started" if name == "search_documents" else "tool_started",
            state.question,
            decision=decision.action,
            tool_iteration=len(state.tool_calls),
            search_iteration=len(state.searches) if name == "search_documents" else None,
            tool_name=name,
            search_query=decision.search_query,
        ))

        payload = await self._tool_executor.execute(request)

        if name == "search_documents":
            evidence = [
                Evidence(
                    item["text"], item["doc_name"], item["chunk_index"],
                    item["similarity"], arguments["question"]
                )
                for item in payload
            ]
            added = self._add_evidence(state, evidence)
        elif name == "list_documents":
            state.available_documents = payload
            added = []
        else:
            state.document_reads.append({
                key: payload.get(key)
                for key in (
                    "doc_name", "start_chunk", "chunks_returned",
                    "next_chunk", "has_more", "total_chunks",
                )
            })
            evidence = [
                Evidence(
                    item["text"], payload["doc_name"], item["chunk_index"], None,
                    f"read:{payload['doc_name']}", "document_read"
                )
                for item in payload.get("chunks", [])
            ]
            added = self._add_evidence(state, evidence)

        self._trace(TraceEvent(
            run_id,
            "retrieval_completed" if name == "search_documents" else "tool_completed",
            state.question,
            decision=decision.action,
            tool_iteration=len(state.tool_calls),
            search_iteration=len(state.searches) if name == "search_documents" else None,
            tool_name=name,
            search_query=decision.search_query,
            result_count=len(payload) if name == "list_documents" else len(added),
            results=[
                {
                    "doc_name": item.doc_name,
                    "chunk_index": item.chunk_index,
                    "similarity": item.similarity,
                }
                for item in added
            ],
        ))

    def _add_evidence(
        self, state: AgentState, candidates: list[Evidence]
    ) -> list[Evidence]:
        existing = {(item.doc_name, item.chunk_index) for item in state.evidence}
        added = []
        for item in candidates:
            identifier = (item.doc_name, item.chunk_index)
            if identifier in existing:
                continue
            if (
                state.evidence_chars + len(item.text)
                > state.config.max_evidence_characters
            ):
                break
            state.evidence.append(item)
            state.evidence_chars += len(item.text)
            existing.add(identifier)
            added.append(item)
        return added

    def _finish(
        self,
        run_id: str,
        state: AgentState,
        answer: str,
        reason: TerminationReason,
    ) -> AnalystAnswer:
        self._trace(TraceEvent(
            run_id,
            "run_terminated",
            state.question,
            tool_iteration=len(state.tool_calls),
            termination_reason=reason,
        ))
        return AnalystAnswer(
            answer,
            list(state.evidence),
            len(state.searches),
            reason,
            len(state.tool_calls),
        )

    def _trace(self, event: TraceEvent) -> None:
        self.trace_recorder.record(event)
