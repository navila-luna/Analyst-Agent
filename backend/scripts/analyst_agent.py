import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from app.agent_runtime import analyst_agent_runtime


async def run(question: str):
    load_dotenv()
    async with analyst_agent_runtime() as analyst_agent:
        return await analyst_agent.run(question)


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
