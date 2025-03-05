import asyncio
import os

from arsenalpy.agents.agent import Agent, AgentConfig
from dotenv import load_dotenv

load_dotenv()

agent = Agent(
    name="basic_agent",
    provider="openrouter",
    system_prompt="You are a helpful assistant that can answer questions and help with tasks.",
    model="x-ai/grok-2-1212",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    config=AgentConfig(),
)


async def main():
    # stream the response
    async for chunk in agent.do_stream(
        prompt="tell me what is the meaning of life?",
    ):
        print(chunk, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
