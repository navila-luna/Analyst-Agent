import unittest

from httpx import ASGITransport, AsyncClient

from app.analyst_agent import AnalystAnswer, Evidence
from app.main import app, get_analyst_agent


class FakeAnalystAgent:
    def __init__(self, result):
        self.result = result
        self.questions = []

    async def run(self, question):
        self.questions.append(question)
        return self.result


class AnalystAgentAPITests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        app.dependency_overrides.clear()

    async def post_question(self, agent, question):
        app.dependency_overrides[get_analyst_agent] = lambda: agent
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/analyst-agent/v1/ask",
                json={"question": question},
            )

    async def test_agent_endpoint_preserves_existing_chat_response_shape(self):
        duplicate = Evidence("text", "guide.md", 2, 0.87, "query")
        agent = FakeAnalystAgent(AnalystAnswer(
            text="Grounded answer.",
            evidence=[duplicate, duplicate],
            searches_performed=2,
            termination_reason="model_answered",
        ))

        response = await self.post_question(agent, "How does this work?")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(agent.questions, ["How does this work?"])
        self.assertEqual(response.json(), {
            "answer": "Grounded answer.",
            "in_docs": True,
            "references": [{
                "index": 1,
                "doc_name": "guide.md",
                "chunk_index": 2,
                "similarity": 0.87,
            }],
        })

    async def test_agent_endpoint_reports_no_document_evidence(self):
        agent = FakeAnalystAgent(AnalystAnswer(
            text="There is not enough evidence.",
            evidence=[],
            searches_performed=1,
            termination_reason="no_evidence",
        ))

        response = await self.post_question(agent, "Unknown topic")

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["in_docs"])
        self.assertEqual(response.json()["references"], [])
