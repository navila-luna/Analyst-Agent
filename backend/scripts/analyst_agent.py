import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from mcp import Client, StdioServerParameters
from openai import AsyncOpenAI

from app.analyst_agent import AnalystAgent, OpenAIAnalystModel

BACKEND_DIR = Path(__file__).parent.parent


async def run(question: str):
    load_dotenv()
    server = StdioServerParameters(
        command=str(Path(sys.executable).with_name("mcp")),
        args=["run", "mcp_server.py:mcp"],
        cwd=BACKEND_DIR,
    )

    async with Client(server) as mcp_client:
        model = OpenAIAnalystModel(AsyncOpenAI())
        return await AnalystAgent(model, mcp_client).run(question)


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]).strip() or input("Question: ").strip()
    result = asyncio.run(run(question))
    print(result.text)

    if result.evidence:
        print("\nSources considered:")
        for item in result.evidence:
            print(
                f"- {item.doc_name} (chunk {item.chunk_index}, "
                f"similarity={item.similarity})"
            )
