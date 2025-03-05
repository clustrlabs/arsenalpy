import asyncio
from typing import AsyncGenerator

import pytest
from arsenalpy.agents.agent import Agent, AgentConfig

# --- Fake Client Implementation for non-streaming responses ---


class FakeChoice:
    def __init__(self, content: str):
        # Create a simple object to mimic a message with content.
        self.message = type("Message", (), {"content": content})()


# --- Fake Client Implementation for streaming responses ---
class FakeResponse:
    def __init__(self, content: str):
        # The Agent.do method expects a "choices" list.
        self.choices = [FakeChoice(content)]


class FakeChatCompletions:
    async def create(self, **kwargs):
        # Create a fake completion object with a choices list containing a fake message.
        class FakeMessage:
            content = "test content"

        class FakeChoice:
            message = FakeMessage()

        # Dynamically create a fake completion type.
        FakeCompletion = type("FakeCompletion", (), {"choices": [FakeChoice()]})
        return FakeCompletion()


class FakeChat:
    @property
    def completions(self):
        return FakeChatCompletions()


class FakeClient:
    @property
    def chat(self):
        return FakeChat()


class FakeAsyncOpenAI:
    def __init__(self, *args, **kwargs):
        # The actual client has a "chat" attribute.
        self.chat = FakeChat()


def fake_async_openai(*args, **kwargs):
    # Replace the AsyncOpenAI initializer with our fake implementation.
    return FakeAsyncOpenAI(*args, **kwargs)


@pytest.mark.asyncio
async def test_agent_do(monkeypatch):
    # Create an Agent instance with dummy values.
    agent = Agent(
        name="test_agent",
        system_prompt="You are a helpful assistant that provides accurate information.",
        model="gpt-4o",
        api_key="fake_key",
        config=AgentConfig(temperature=0.5),
        skills=[],
        provider="openai",
    )

    # Override the _client attribute with our fake client.
    agent._client = FakeClient()

    # Execute the agent's do method.
    result = await agent.do(prompt="What is the weather in Tokyo?")
    assert result == "test content"


# --- Fake Client Implementation for streaming tests ---


class FakeStreamingChoice:
    def __init__(self, content: str):
        # Mimic the structure of the streaming response's choice having a delta with content.
        self.delta = type("Delta", (), {"content": content})()


class FakeChunk:
    def __init__(self, content: str):
        self.choices = [FakeStreamingChoice(content)]


async def fake_stream() -> AsyncGenerator[FakeChunk, None]:
    # Simulate chunks that yield one character at a time.
    for char in "hello":
        await asyncio.sleep(0)  # simulate async I/O
        yield FakeChunk(char)


class FakeStreamingChatCompletions:
    async def create(self, model: str, messages: list, stream=False, **kwargs):
        if stream:
            return fake_stream()
        return FakeResponse("Test response")


class FakeStreamingChat:
    def __init__(self):
        self.completions = FakeStreamingChatCompletions()


class FakeStreamingClient:
    def __init__(self):
        self.chat = FakeStreamingChat()


@pytest.mark.asyncio
async def test_do_stream():
    # Setup a dummy agent.
    config = AgentConfig()
    agent = Agent(
        name="TestAgent",
        system_prompt="You are a test agent.",
        model="gpt-4o-mini",
        api_key="dummy-key",
        config=config,
    )
    # Inject our FakeStreamingClient to bypass actual HTTP requests.
    agent._client = FakeStreamingClient()

    output = ""
    # Note: do_stream() is an asynchronous generator.
    async for chunk in agent.do_stream(prompt="Test prompt"):
        output += chunk

    assert output == "hello"


@pytest.mark.asyncio
async def test_agent_do_response_format_transformation():
    # Define fake objects to simulate API response structure.
    class FakeMessage:
        def __init__(self, content):
            self.content = content

    class FakeChoice:
        def __init__(self, content):
            self.message = FakeMessage(content)

    class FakeCompletion:
        def __init__(self, content):
            self.choices = [FakeChoice(content)]

    fake_completion = FakeCompletion("test response")

    # Create a dummy client structure matching the openai client API.
    class DummyChatCompletions:
        async def create(self, *args, **kwargs):
            # Verify that response_format is properly transformed.
            response_format = kwargs.get("response_format")
            assert isinstance(response_format, dict)
            assert response_format == {"type": "json_object"}
            return fake_completion

    class DummyChat:
        completions = DummyChatCompletions()

    class DummyClient:
        chat = DummyChat()

    # Build an agent with the "json_object" response_format.
    config = AgentConfig(response_format={"type": "json_object"})
    agent = Agent(
        name="agent1",
        system_prompt="System prompt",
        model="model-1",
        api_key="dummy_key",
        config=config,
    )
    # Inject our dummy client to override the real one.
    agent._client = DummyClient()

    result = await agent.do("Hello")
    assert result == "test response"


def test_agent_provider_initialization():
    """Test agent initialization with different providers."""
    # Test OpenAI provider
    agent_openai = Agent(
        name="test_agent",
        system_prompt="test prompt",
        model="gpt-4",
        api_key="test_key",
        provider="openai",
        config=AgentConfig(),
    )
    assert str(agent_openai._client.base_url) == "https://api.openai.com/v1/"

    # Test X-AI provider
    agent_xai = Agent(
        name="test_agent",
        system_prompt="test prompt",
        model="gpt-4",
        api_key="test_key",
        provider="x-ai",
        config=AgentConfig(),
    )
    assert str(agent_xai._client.base_url) == "https://api.x.ai/v1/"

    # Test OpenRouter provider
    agent_openrouter = Agent(
        name="test_agent",
        system_prompt="test prompt",
        model="gpt-4",
        api_key="test_key",
        provider="openrouter",
        config=AgentConfig(),
    )
    assert str(agent_openrouter._client.base_url) == "https://openrouter.ai/api/v1/"


def test_agent_skill_handling():
    """Test agent handling of different skill types."""
    # Test dictionary skill
    dict_skill = {
        "name": "test_skill",
        "description": "A test skill",
        "parameters": {"type": "object", "properties": {}},
    }

    agent = Agent(
        name="test_agent",
        system_prompt="test prompt",
        model="gpt-4",
        api_key="test_key",
        skills=[dict_skill],
        config=AgentConfig(),
    )
    assert len(agent.skills) == 1
    assert agent.skills[0] == dict_skill

    # Test invalid skill
    with pytest.raises(ValueError, match="Each skill must be either"):
        Agent(
            name="test_agent",
            system_prompt="test prompt",
            model="gpt-4",
            api_key="test_key",
            skills=[123],  # Invalid skill type
            config=AgentConfig(),
        )


def test_agent_special_model_handling():
    """Test agent handling of special models."""
    # Test o1 model
    agent_o1 = Agent(
        name="test_agent",
        system_prompt="test prompt",
        model="o1",
        api_key="test_key",
        config=AgentConfig(temperature=0.7),
    )
    assert agent_o1.config.temperature is None

    # Test o3-mini model
    agent_o3 = Agent(
        name="test_agent",
        system_prompt="test prompt",
        model="o3-mini",
        api_key="test_key",
        config=AgentConfig(temperature=0.7),
    )
    assert agent_o3.config.temperature is None


@pytest.mark.asyncio
async def test_agent_do_error_handling():
    """Test agent error handling in do method."""

    class ErrorClient:
        class ChatCompletions:
            async def create(self, **kwargs):
                return {"error": "Test error"}

        class Chat:
            def __init__(self):
                self.completions = ErrorClient.ChatCompletions()

        def __init__(self):
            self.chat = self.Chat()

    agent = Agent(
        name="test_agent",
        system_prompt="test prompt",
        model="gpt-4",
        api_key="test_key",
        config=AgentConfig(),
    )
    agent._client = ErrorClient()

    with pytest.raises(Exception, match="LLM returned unexpected response format"):
        await agent.do("test prompt")


@pytest.mark.asyncio
async def test_agent_do_with_tool_calls():
    """Test agent handling of tool calls."""

    class Message:
        def __init__(self, content, tool_calls=None, role="assistant"):
            self.content = content
            self.tool_calls = tool_calls
            self.role = role

    class Function:
        def __init__(self, name, arguments):
            self.name = name
            self.arguments = arguments

    class ToolCall:
        def __init__(self, function):
            self.function = function

    class Choice:
        def __init__(self, message):
            self.message = message

    class Response:
        def __init__(self, choices):
            self.choices = choices

    class ToolCallClient:
        class ChatCompletions:
            async def create(self, **kwargs):
                if not hasattr(self, "first_call"):
                    self.first_call = True
                    return Response(
                        [
                            Choice(
                                Message(
                                    "Using tool",
                                    [
                                        ToolCall(
                                            Function("test_tool", '{"arg": "value"}')
                                        )
                                    ],
                                )
                            )
                        ]
                    )
                return Response([Choice(Message("Final response", role="assistant"))])

        class Chat:
            def __init__(self):
                self.completions = ToolCallClient.ChatCompletions()

        def __init__(self):
            self.chat = self.Chat()

    # Define the test tool with proper skill definition
    async def test_tool(**kwargs):
        return "Tool result"

    test_tool.skill_definition = {
        "name": "test_tool",
        "description": "A test tool",
        "parameters": {"type": "object", "properties": {"arg": {"type": "string"}}},
    }
    test_tool.__name__ = "test_tool"

    agent = Agent(
        name="test_agent",
        system_prompt="test prompt",
        model="gpt-4",
        api_key="test_key",
        skills=[test_tool],
        config=AgentConfig(),
    )
    agent._client = ToolCallClient()

    result = await agent.do("Use tool")
    assert result == "Final response"


@pytest.mark.asyncio
async def test_agent_do_with_invalid_tool_args():
    """Test agent handling of invalid tool arguments."""

    class InvalidToolArgsClient:
        class ChatCompletions:
            async def create(self, **kwargs):
                return type(
                    "Response",
                    (),
                    {
                        "choices": [
                            type(
                                "Choice",
                                (),
                                {
                                    "message": type(
                                        "Message",
                                        (),
                                        {
                                            "content": "Using tool",
                                            "tool_calls": [
                                                type(
                                                    "ToolCall",
                                                    (),
                                                    {
                                                        "function": type(
                                                            "Function",
                                                            (),
                                                            {
                                                                "name": "test_tool",
                                                                "arguments": "invalid json",
                                                            },
                                                        )
                                                    },
                                                )
                                            ],
                                        },
                                    )
                                },
                            )
                        ]
                    },
                )

        class Chat:
            def __init__(self):
                self.completions = InvalidToolArgsClient.ChatCompletions()

        def __init__(self):
            self.chat = self.Chat()

    # Define the test tool with proper skill definition
    async def test_tool(**kwargs):
        return "Tool result"

    test_tool.skill_definition = {
        "name": "test_tool",
        "description": "A test tool",
        "parameters": {"type": "object", "properties": {"arg": {"type": "string"}}},
    }
    test_tool.__name__ = "test_tool"

    agent = Agent(
        name="test_agent",
        system_prompt="test prompt",
        model="gpt-4",
        api_key="test_key",
        skills=[test_tool],
        config=AgentConfig(),
    )
    agent._client = InvalidToolArgsClient()

    with pytest.raises(ValueError, match="Invalid function arguments"):
        await agent.do("Use tool")


@pytest.mark.asyncio
async def test_agent_do_stream_with_tool_calls():
    """Test streaming with tool calls."""

    class StreamToolCallClient:
        class ChatCompletions:
            async def create(self, **kwargs):
                if kwargs.get("stream", False):

                    async def stream():
                        for char in "streamed":
                            yield type(
                                "Chunk",
                                (),
                                {
                                    "choices": [
                                        type(
                                            "Choice",
                                            (),
                                            {
                                                "delta": type(
                                                    "Delta", (), {"content": char}
                                                )
                                            },
                                        )
                                    ]
                                },
                            )

                    return stream()

                if not hasattr(self, "first_call"):
                    self.first_call = True
                    return type(
                        "Response",
                        (),
                        {
                            "choices": [
                                type(
                                    "Choice",
                                    (),
                                    {
                                        "message": type(
                                            "Message",
                                            (),
                                            {
                                                "content": "Using tool",
                                                "tool_calls": [
                                                    type(
                                                        "ToolCall",
                                                        (),
                                                        {
                                                            "function": type(
                                                                "Function",
                                                                (),
                                                                {
                                                                    "name": "test_tool",
                                                                    "arguments": '{"arg": "value"}',
                                                                },
                                                            )
                                                        },
                                                    )
                                                ],
                                            },
                                        )
                                    },
                                )
                            ]
                        },
                    )
                return type(
                    "Response",
                    (),
                    {
                        "choices": [
                            type(
                                "Choice",
                                (),
                                {
                                    "message": type(
                                        "Message",
                                        (),
                                        {
                                            "content": "Final response",
                                            "role": "assistant",
                                        },
                                    )
                                },
                            )
                        ]
                    },
                )

        class Chat:
            def __init__(self):
                self.completions = StreamToolCallClient.ChatCompletions()

        def __init__(self):
            self.chat = self.Chat()

    # Define the test tool with proper skill definition
    async def test_tool(**kwargs):
        return "Tool result"

    test_tool.skill_definition = {
        "name": "test_tool",
        "description": "A test tool",
        "parameters": {"type": "object", "properties": {"arg": {"type": "string"}}},
    }
    test_tool.__name__ = "test_tool"

    agent = Agent(
        name="test_agent",
        system_prompt="test prompt",
        model="gpt-4",
        api_key="test_key",
        skills=[test_tool],
        config=AgentConfig(),
    )
    agent._client = StreamToolCallClient()

    output = ""
    async for chunk in agent.do_stream("Use tool"):
        output += chunk
    assert output == "streamed"


@pytest.mark.asyncio
async def test_agent_do_validation_error():
    """Test handling of validation errors in responses."""

    class ValidationErrorClient:
        class ChatCompletions:
            async def create(self, **kwargs):
                return {
                    "choices": [{"message": {"role": "invalid"}}]
                }  # Invalid message format

        class Chat:
            def __init__(self):
                self.completions = ValidationErrorClient.ChatCompletions()

        def __init__(self):
            self.chat = self.Chat()

    agent = Agent(
        name="test_agent",
        system_prompt="test prompt",
        model="gpt-4",
        api_key="test_key",
        config=AgentConfig(),
    )
    agent._client = ValidationErrorClient()

    with pytest.raises(Exception, match="Invalid response format from provider"):
        await agent.do("test prompt")
