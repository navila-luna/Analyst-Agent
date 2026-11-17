import sys
from contextlib import asynccontextmanager
from pathlib import Path

from mcp import Client, StdioServerParameters
from openai import AsyncOpenAI

from app.agent.agent import AnalystAgent
from app.agent.openai_model import OpenAIAnalystModel

BACKEND_DIR = Path(__file__).parent.parent


@asynccontextmanager
async def analyst_agent_runtime():
    """Start the analyst agent with its MCP server over stdio."""
    server = StdioServerParameters(
        command=str(Path(sys.executable).with_name("mcp")),
        args=["run", "mcp_server.py:mcp"],
        cwd=BACKEND_DIR,
    )

    async with Client(server) as mcp_client:
        model = OpenAIAnalystModel(AsyncOpenAI())
        yield AnalystAgent(model, mcp_client)
