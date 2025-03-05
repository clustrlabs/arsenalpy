"""
Arsenal Agents Module

This module provides the core Agent functionality for Arsenal, enabling interaction with various
AI models through a unified interface. It supports multiple providers including OpenAI, X-AI,
and OpenRouter.

Example:
    ```python
    from arsenal.agents import Agent, AgentConfig

    # Create an agent configuration
    config = AgentConfig(
        temperature=0.7,
        max_completion_tokens=1000
    )

    # Initialize an agent
    agent = Agent(
        name="my_assistant",
        system_prompt="You are a helpful assistant",
        model="gpt-4",
        api_key="your-api-key",
        provider="openai",
        config=config
    )

    # Use the agent
    async def main():
        response = await agent.do("What is the capital of France?")
        print(response)
    ```

Classes:
    - Agent: Main class for creating and managing AI agents
    - AgentConfig: Configuration class for customizing agent behavior
"""

from .agent import Agent, AgentConfig

__all__ = ["Agent", "AgentConfig"]
