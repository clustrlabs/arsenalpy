"""
Arsenal Skills Module

This module provides a framework for creating and managing skills that can be used by Arsenal agents.
Skills are specialized functions that agents can use to perform specific tasks, such as database operations,
web searches, or blockchain interactions.

Example:
    ```python
    from arsenal.skills import skill
    from pydantic import BaseModel, Field

    class MySkillInput(BaseModel):
        param: str = Field(..., description="Parameter description")

    @skill(MySkillInput)
    async def my_skill(param: str):
        '''My skill documentation'''
        return f"Processed: {param}"
    ```

Components:
    - skill: Decorator for creating new skills with input validation
    - Various built-in skills for common operations
"""

from .search_skill import search

# Export all core skills and the decorator for building new skills
# from .get_current_block import get_current_block
# from .serper_search import serper_search
from .skill_utils import skill

__all__ = ["skill", "search"]
