"""
Core Agent Implementation Module

This module contains the main Agent implementation for Arsenal.
It provides a flexible and powerful interface for interacting with various AI models
and executing skills through them.

Key Components:
    - Agent: The main class for creating and managing AI agents
    - AgentConfig: Configuration settings for agents
    - Skill: Model for defining executable skills
    - ChatMessage: Model for chat messages
    - Choice: Model for completion choices
    - ChatCompletionResponse: Model for API responses

Example:
    ```python
    from arsenal.agents import Agent, AgentConfig

    config = AgentConfig(temperature=0.7)
    agent = Agent(
        name="research_assistant",
        system_prompt="You are a helpful research assistant",
        model="gpt-4",
        api_key="your-api-key",
        provider="openai",
        config=config
    )

    async def main():
        # Basic usage
        response = await agent.do("Research quantum computing")

        # Streaming response
        async for chunk in agent.do_stream("Explain neural networks"):
            print(chunk, end="")
    ```
"""

import asyncio
import json
from typing import Any, AsyncGenerator, Callable, Dict, List, Literal, Optional, cast

from openai import AsyncOpenAI
from pydantic import BaseModel, PrivateAttr, ValidationError

from ..skills.skill_utils import ConfiguredSkillCall
from .agent_config import AgentConfig
from .types import ChatCompletionResponse, ChatMessage, Skill


class Agent(BaseModel):
    """
    Main agent class for managing AI interactions and skill execution.

    This class provides a unified interface for interacting with various AI models
    and executing skills through them. It supports multiple providers and can be
    configured for different use cases.

    Attributes:
        name (str): The name of the agent
        system_prompt (str): The system prompt that defines the agent's behavior
        model (str): The AI model to use
        api_key (str): API key for the provider
        skills (Optional[List]): List of skills available to the agent
        provider (str): The AI provider to use ("openai", "x-ai", or "openrouter")
        config (AgentConfig): Configuration settings for the agent

    Example:
        ```python
        agent = Agent(
            name="assistant",
            system_prompt="You are a helpful assistant",
            model="gpt-4",
            api_key="your-api-key",
            provider="openai",
            config=AgentConfig(temperature=0.7)
        )

        response = await agent.do("What is the weather?")
        ```
    """

    name: str
    system_prompt: str
    model: str
    api_key: str
    skills: Optional[List] = None  # now can be callables, dicts, or Skill models
    provider: Literal["openai", "x-ai", "openrouter"] = "openrouter"
    config: AgentConfig

    # _client is used by the agent to talk to the external API.
    _client: Any = PrivateAttr()
    # _skill_map maps function names to the actual callable skills.
    _skill_map: Dict[str, Callable] = PrivateAttr(default_factory=dict)

    def __init__(self, **kwargs: Any) -> None:
        # Call BaseModel's __init__ first.
        super().__init__(**kwargs)

        if self.model in ["o1", "o3-mini"]:
            self.config.temperature = None

        # Handle empty skills arrays generically - avoid sending empty tools array to any provider
        if self.skills is not None and len(self.skills) == 0:
            self.skills = None
            self.config.tool_choice = None

        # When skills is None, ensure tool_choice is also None
        if self.skills is None:
            self.config.tool_choice = None

        # Ensure skills is at least an empty list, not None (except when explicitly set to None above)
        if self.skills is None and self.config.tool_choice is not None:
            self.skills = []

        # Process skills and build a mapping of function name to callable
        if self.skills:
            new_skills = []
            for skill_item in self.skills:
                if isinstance(skill_item, ConfiguredSkillCall):
                    new_skills.append(skill_item.skill_definition)
                    self._skill_map[skill_item.__name__] = skill_item
                elif callable(skill_item):
                    if hasattr(skill_item, "skill_definition"):
                        new_skills.append(skill_item.skill_definition)
                        self._skill_map[skill_item.__name__] = skill_item
                    elif asyncio.iscoroutine(skill_item):
                        configured_skill = skill_item
                        if hasattr(configured_skill, "skill_definition"):
                            new_skills.append(configured_skill.skill_definition)
                            self._skill_map[configured_skill.__name__] = cast(
                                Callable, configured_skill
                            )
                elif isinstance(skill_item, dict):
                    new_skills.append(skill_item)
                elif isinstance(skill_item, Skill):
                    new_skills.append(skill_item.model_dump())
                else:
                    raise ValueError(
                        "Each skill must be either a callable with 'skill_definition', "
                        "a configured skill call, or a dict."
                    )
            self.skills = new_skills

        # Based on provider, assign the proper client.
        if self.provider == "openai":
            self._client = AsyncOpenAI(
                base_url="https://api.openai.com/v1",
                api_key=self.api_key,
            )
        elif self.provider == "x-ai":
            self._client = AsyncOpenAI(
                base_url="https://api.x.ai/v1",
                api_key=self.api_key,
            )
        elif self.provider == "openrouter":
            self._client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key,
            )

    async def do(self, prompt: str, image_url: Optional[str] = None) -> str:
        """
        Execute a single interaction with the AI model.

        This method handles both direct responses and tool-based interactions.
        It will automatically execute any requested tools and provide their
        results back to the AI for final response generation.

        Args:
            prompt (str): The user's input prompt
            image_url (Optional[str]): URL of an image to include in the prompt

        Returns:
            str: The AI's response

        Raises:
            Exception: If the API response is invalid or contains errors
            ValueError: If function arguments are invalid
        """
        # Using a simple string for the user message.
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        if image_url is not None:
            messages.append({"role": "user", "content": image_url})

        params = self.config.to_api_params()

        # First round: get response from LLM.
        api_params = {
            "model": self.model,
            "messages": messages,
            **params,
        }

        # Only include tools parameter if skills is not None
        if self.skills is not None:
            api_params["tools"] = self.skills
        else:
            # Remove tool_choice if no tools/skills are provided
            api_params.pop("tool_choice", None)

        completion = await self._client.chat.completions.create(**api_params)

        # Safely extract the message from the completion response.
        if hasattr(completion, "choices") and completion.choices:
            message = completion.choices[0].message
        elif isinstance(completion, dict) and completion.get("choices"):
            message = completion["choices"][0]["message"]
            # Convert dict message to object with content attribute if needed
            if isinstance(message, dict):
                # Proper validation of message format
                if "role" not in message or message["role"] not in [
                    "assistant",
                    "system",
                    "user",
                    "function",
                    "tool",
                ]:
                    raise Exception("Invalid response format from provider") from None

                if "content" not in message:
                    try:
                        # Try to validate and build a proper message object
                        message = ChatMessage(
                            role=message.get("role", "assistant"),
                            content=message.get("content", ""),
                        )
                    except Exception as err:
                        # Validation of the message failed, so raise proper exception
                        raise Exception(
                            f"Invalid response format from provider: {err}"
                        ) from err
        elif isinstance(completion, dict) and "text" in completion:
            message = ChatMessage(role="assistant", content=completion["text"])
        else:
            raise Exception(
                "LLM returned unexpected response format: " + str(completion)
            )

        # Check if the LLM requested a tool call.
        if getattr(message, "tool_calls", None):
            tool_call = message.tool_calls[0]
            func_name = tool_call.function.name
            try:
                func_args = json.loads(tool_call.function.arguments)
            except Exception as ex:
                raise ValueError("Invalid function arguments") from ex

            # Execute the skill.
            result = await self._skill_map[func_name](**func_args)
            # Ensure the function result is a string; if not, convert it.
            if not isinstance(result, str):
                result = str(result)

            # Append a function response with the function result.
            messages.append(
                {
                    "role": "assistant",
                    "content": result,
                }
            )

            # Second round: get final answer from the LLM now that it has the function result.
            followup_params = params.copy()
            followup_params.pop(
                "tool_choice", None
            )  # Remove tool_choice for final call as it's not applicable
            final_completion = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                **followup_params,
            )

            # Validate the response with Pydantic to catch invalid formats
            try:
                if not isinstance(final_completion, dict):
                    if hasattr(final_completion, "to_dict"):
                        final_completion = final_completion.to_dict()
                    else:
                        final_completion = json.loads(
                            json.dumps(final_completion, default=lambda o: o.__dict__)
                        )

                if final_completion.get("choices") is None:
                    final_completion["choices"] = []

                validated_response = ChatCompletionResponse.model_validate(
                    final_completion
                )
            except ValidationError as err:
                raise Exception(
                    f"Invalid response format from provider: {err}"
                ) from err

            if validated_response.error:
                raise Exception(f"Provider returned error: {validated_response.error}")

            if not validated_response.choices:
                raise Exception(
                    "Provider returned an empty response (no choices found)."
                )

            final_message = validated_response.choices[0].message
            return final_message.content or ""

        # Otherwise, simply return the direct answer.
        return message.content or ""

    async def do_stream(
        self, prompt: str, image_url: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream the AI's response token by token.

        This method provides the same functionality as `do()` but streams the
        response as it's generated, which can provide a better user experience
        for longer responses.

        Args:
            prompt (str): The user's input prompt
            image_url (Optional[str]): URL of an image to include in the prompt

        Yields:
            str: Individual tokens from the AI's response

        Raises:
            Exception: If the API response is invalid or contains errors
            ValueError: If function arguments are invalid
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        if image_url is not None:
            messages.append({"role": "user", "content": image_url})
        params = self.config.to_api_params()

        # First round: non-streaming call for initial response.
        api_params = {
            "model": self.model,
            "messages": messages,
            "stream": False,  # non-streaming call for initial round
            **params,
        }

        # Only include tools parameter if skills is not None
        if self.skills is not None:
            api_params["tools"] = self.skills
        else:
            # Remove tool_choice if no tools/skills are provided
            api_params.pop("tool_choice", None)

        completion = await self._client.chat.completions.create(**api_params)

        if hasattr(completion, "choices") and completion.choices:
            message = completion.choices[0].message
        elif isinstance(completion, dict) and completion.get("choices"):
            message = completion["choices"][0]["message"]
            # Convert dict message to object with content attribute if needed
            if isinstance(message, dict):
                # Proper validation of message format
                if "role" not in message or message["role"] not in [
                    "assistant",
                    "system",
                    "user",
                    "function",
                    "tool",
                ]:
                    raise Exception("Invalid response format from provider") from None

                if "content" not in message:
                    try:
                        # Try to validate and build a proper message object
                        message = ChatMessage(
                            role=message.get("role", "assistant"),
                            content=message.get("content", ""),
                        )
                    except Exception as err:
                        # Validation of the message failed, so raise proper exception
                        raise Exception(
                            f"Invalid response format from provider: {err}"
                        ) from err
        elif isinstance(completion, dict) and "text" in completion:
            message = ChatMessage(role="assistant", content=completion["text"])
        else:
            raise Exception(
                "LLM returned unexpected response format: " + str(completion)
            )

        if getattr(message, "tool_calls", None):
            tool_call = message.tool_calls[0]
            func_name = tool_call.function.name
            try:
                func_args = json.loads(tool_call.function.arguments)
            except Exception as ex:
                raise ValueError("Invalid function arguments") from ex

            result = await self._skill_map[func_name](**func_args)
            if not isinstance(result, str):
                result = str(result)
            messages.append(
                {
                    "role": "assistant",
                    "content": result,
                }
            )

            followup_params = params.copy()
            followup_params.pop(
                "tool_choice", None
            )  # Remove tool_choice as it's not applicable for final call
            stream_completion = await self._client.chat.completions.create(
                model=self.model, messages=messages, stream=True, **followup_params
            )
            async for chunk in stream_completion:
                token = getattr(chunk.choices[0].delta, "content", "")
                if token:
                    yield token
        else:
            followup_params = params.copy()
            followup_params.pop(
                "tool_choice", None
            )  # Remove tool_choice as it's not applicable for streaming calls
            stream_completion = await self._client.chat.completions.create(
                model=self.model, messages=messages, stream=True, **followup_params
            )
            async for chunk in stream_completion:
                token = getattr(chunk.choices[0].delta, "content", "")
                if token:
                    yield token
