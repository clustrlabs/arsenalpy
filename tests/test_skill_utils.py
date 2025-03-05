import asyncio

import pytest
from arsenalpy.skills.skill_utils import ConfiguredSkillCall, skill
from pydantic import BaseModel, Field


class SyncInput(BaseModel):
    x: int = Field(..., description="A number")


@skill(SyncInput)
def sync_skill(x: int):
    """A synchronous skill that multiplies by two"""
    return x * 2


class AsyncInput(BaseModel):
    msg: str = Field(..., description="A message")


@skill(AsyncInput)
async def async_skill(msg: str):
    """An asynchronous skill that echoes the message"""
    return f"Echo: {msg}"


class ComplexInput(BaseModel):
    a: int = Field(..., description="Field a")
    b: str = Field("default_b", description="Field b")


@skill(ComplexInput)
def complex_skill(a: int, b: str):
    """Complex skill that concatenates values"""
    return f"a={a}, b={b}"


def test_sync_skill_validation():
    # Test synchronous skill validation and output
    configured_call = sync_skill(x=3)
    # Await the configured skill call
    result = asyncio.run(configured_call())
    assert result == 6

    # Test that the skill definition has schema info from the input model
    skill_def = configured_call.skill_definition
    assert "function" in skill_def
    assert "parameters" in skill_def["function"]


@pytest.mark.asyncio
async def test_async_skill_validation():
    call = async_skill(msg="hello")
    output = await call()
    assert output == "Echo: hello"

    skill_def = call.skill_definition
    assert "function" in skill_def
    assert "parameters" in skill_def["function"]


def test_skill_configuration():
    # Test that calling the skill with configuration returns a configured skill call with defaults updated
    configured = sync_skill(x=10)
    # Ensure that we received a ConfiguredSkillCall
    assert isinstance(configured, ConfiguredSkillCall)

    # Verify that the default in the skill definition is updated to 10 for parameter 'x'
    skill_def = configured.skill_definition
    props = skill_def["function"]["parameters"]["properties"]
    assert props["x"].get("default") == 10


def test_complex_skill_configuration():
    # Call complex_skill with only 'a' provided, so that b defaults to 'default_b'
    configured = complex_skill(a=42)

    # Since our skills are wrapped, we call it asynchronously
    result = asyncio.run(configured())
    assert result == "a=42, b=default_b"

    # Check that the skill definition is updated: parameter 'a' should have default 42, and removed from required
    skill_def = configured.skill_definition
    props = skill_def["function"]["parameters"]["properties"]
    assert props["a"].get("default") == 42
    required = skill_def["function"]["parameters"].get("required", [])
    assert "a" not in required
