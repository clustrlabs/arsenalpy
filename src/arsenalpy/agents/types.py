"""
Type definitions for Arsenal agents.

This module contains all the type definitions used by the agent system.
"""

from typing import List, Optional

from pydantic import BaseModel


class ChatMessage(BaseModel):
    """Model representing a message in the chat conversation."""

    role: str
    content: Optional[str] = None


class Choice(BaseModel):
    """Model representing a choice in the chat completion response."""

    message: ChatMessage


class ChatCompletionResponse(BaseModel):
    """Model representing the complete response from the chat completion API."""

    choices: List[Choice] = []
    error: Optional[dict] = None


class Skill(BaseModel):
    """Model representing an executable skill that can be used by an agent."""

    type: Optional[str] = None
    function: dict
