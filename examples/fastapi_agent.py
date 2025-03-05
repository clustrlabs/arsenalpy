# api agent

import os

import uvicorn
from arsenalpy.agents.agent import Agent, AgentConfig
from arsenalpy.skills.search_skill import search
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()


agent = Agent(
    name="fastapi_agent",
    provider="openrouter",
    model="x-ai/grok-2-1212",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    config=AgentConfig(),
    system_prompt="You are a helpful API agent.",
    skills=[search],
)


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "I am a fastapi agent"}


class QueryRequest(BaseModel):
    query: str


@app.post("/query")
async def query_agent(req: QueryRequest):
    result = await agent.do(prompt=req.query)
    return {"result": result}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)
