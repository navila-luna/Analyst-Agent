import json
from typing import Any

from app.agent.protocols import MCPClient
from app.agent.models import ModelDecision, ToolRequest
from app.document_repository import MAX_CHUNKS_PER_READ


class ToolExecutor:
    def __init__(self, mcp_client: MCPClient):
        self._mcp_client = mcp_client

    def build_request(
        self,
        decision: ModelDecision,
        question: str,
    ) -> ToolRequest:
        if decision.action == "SEARCH":
            return ToolRequest(
                name="search_documents",
                arguments={
                    "question": decision.search_query or question,
                },
            )

        if decision.action == "LIST":
            return ToolRequest(
                name="list_documents",
                arguments={},
            )

        if decision.action == "READ":
            return ToolRequest(
                name="read_document",
                arguments={
                    "doc_name": decision.doc_name or "",
                    "start_chunk": decision.start_chunk,
                    "max_chunks": min(
                        decision.max_chunks,
                        MAX_CHUNKS_PER_READ,
                    ),
                },
            )

        raise ValueError(f"Unsupported decision: {decision.action}")

    async def execute(self, request: ToolRequest) -> Any:
        result = await self._mcp_client.call_tool(
            request.name,
            request.arguments,
        )

        if getattr(result, "is_error", False):
            raise RuntimeError(
                f"{request.name} returned an MCP tool error"
            )

        return decode_mcp_result(result)


def decode_mcp_result(result: Any) -> Any:
    structured = getattr(result, "structured_content", None)

    if isinstance(structured, dict):
        if "result" in structured:
            return structured["result"]

        return structured

    decoded = []

    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)

        if text is None:
            continue

        try:
            decoded.append(json.loads(text))
        except json.JSONDecodeError as error:
            raise ValueError(
                "MCP tool returned non-JSON text content"
            ) from error

    if len(decoded) == 1:
        return decoded[0]

    if decoded:
        return decoded

    raise ValueError("MCP tool returned no usable content")

