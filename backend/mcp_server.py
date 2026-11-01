from mcp.server import MCPServer
from mcp.types import ToolAnnotations

from app.document_repository import list_documents, read_document
from app.search import search_documents

mcp = MCPServer("Analyst Agent")
mcp.add_tool(
    search_documents,
    annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
)
mcp.add_tool(
    list_documents,
    annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
)
mcp.add_tool(
    read_document,
    annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
)
