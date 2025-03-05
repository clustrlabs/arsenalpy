# Arsenal: A Unified AI Agent Framework

<div align="center">
  <img src="https://github.com/clustrlabs/arsenalpy/raw/main/assets/arsenalpy.png" alt="Arsenal Banner" />
</div>

<!-- [![PyPI version](https://img.shields.io/pypi/v/arsenalpy.svg)](https://pypi.org/project/arsenalpy/)
[![Python](https://img.shields.io/pypi/pyversions/arsenalpy.svg)](https://pypi.org/project/arsenalpy/)
[![License](https://img.shields.io/github/license/clustrlabs/arsenalpy.svg)](https://github.com/clustrlabs/arsenalpy/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://clustrlabs.github.io/arsenalpy/) -->

## Overview

**Arsenal** is [Clustr's](https://clustr.network) lightweight, unified AI agent framework, designed to help you build intelligent applications effortlessly. With its powerful API, you can seamlessly interact with multiple AI models, execute complex skills. 

## Key Features

- **Universal Model Support**: Work with 300+ LLMs from OpenAI, X-AI, Anthropic, OpenRouter, and more through a single unified interface
- **Flexible Skill System**: Create custom tool-using agents with an extensible skill system
- **Built-in Web Capabilities**: Search the web and extract content with included skills
- **Async First**: Built with modern async/await patterns for efficient I/O operations
- **Type Safe**: Full type hints and Pydantic models
- **Stream Support**: Stream responses token by token for better UX
- **Lightweight**: Core package is just 12KB, no bloated dependencies

## Installation

install latest version of uv:

```bash
# Using uv

curl -LsSf https://astral.sh/uv/install.sh | sh

```

then:

```bash
# Using uv
uv pip install "git+https://github.com/clustrlabs/arsenalpy@main"

or

uv pip install "git+https://github.com/clustrlabs/arsenalpy@v0.1.0"

```

For devs :

```Bash
git clone https://github.com/clustrlabs/arsenalpy.git
cd arsenalpy
pip install -e ".[dev]"
```

## Quick Start

```python
import os
from dotenv import load_dotenv
from arsenalpy.agents.agent import Agent, AgentConfig
import asyncio

load_dotenv()

# Create a basic agent
agent = Agent(
    name="basic_agent",
    provider="openrouter",  # Supports openai, anthropic, openrouter, etc.
    system_prompt="You are a helpful assistant that can answer questions and help with tasks.",
    model="x-ai/grok-2-1212",  # Choose any model from supported providers
    api_key=os.getenv("OPENROUTER_API_KEY"),
    config=AgentConfig(),
)

async def main():
    # Stream the response
    async for chunk in agent.do_stream(
        prompt="Tell me what is the meaning of life?",
    ):
        print(chunk, end="", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
```



## Documentation

For detailed documentation, examples, and API reference, visit our [official documentation](https://clustrlabs.github.io/arsenalpy/).

## Examples

Find more examples in our [examples directory](https://github.com/clustrlabs/arsenalpy/tree/main/examples) or check the [documentation examples section](https://clustrlabs.github.io/arsenalpy/examples/).

## Contributing

We welcome contributions! Please see our [contributing guidelines](https://github.com/clustrlabs/arsenalpy/blob/main/CONTRIBUTING.md) for more details.

## License

Arsenal is released under the MIT License. See the [LICENSE](https://github.com/clustrlabs/arsenalpy/blob/main/LICENSE) file for more details.

## Support

For questions, issues, or feature requests, please [open an issue](https://github.com/clustrlabs/arsenalpy/issues) on our GitHub repository.

---

<div align="center">
  <p>Built with ❤️ by <a href="https://clustr.network">Clustr Labs</a></p>
</div>
