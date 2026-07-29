"""Minimal check that the OpenAI API wiring works before any RAG logic is added."""

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say hello in exactly five words."}],
)

print(response.choices[0].message.content)
