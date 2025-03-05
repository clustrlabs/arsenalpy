"""
Skill Utilities Module

This module provides core utilities for creating and managing skills in Arsenal.
The main component is the `skill` decorator which handles input validation and
skill definition for use with agents.

Example:
    ```python
    from arsenal.skills import skill
    from pydantic import BaseModel, Field

    class SearchInput(BaseModel):
        query: str = Field(..., description="Search query")

    @skill(SearchInput)
    async def search(query: str):
        '''Perform a search operation'''
        return f"Results for: {query}"
    ```
"""

import inspect
from typing import Any, Awaitable, Callable, Dict, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class ConfiguredSkillCall:
    def __init__(self, wrapper: "SkillWrapper", configured_args: Dict[str, Any]):
        self.wrapper = wrapper
        self.configured_args = configured_args

    @property
    def skill_definition(self) -> Dict[str, Any]:
        # Deep copy the parent's definition so we don't mutate it.
        import copy

        skill_def = copy.deepcopy(self.wrapper.skill_definition)
        # Update the defaults in the JSON schema using the configured arguments.
        parameters = skill_def.get("function", {}).get("parameters", {})
        if "properties" in parameters:
            for key, value in self.configured_args.items():
                if key in parameters["properties"]:
                    parameters["properties"][key]["default"] = value
        # If a default is provided, remove the parameter from the required list.
        if "required" in parameters and isinstance(parameters["required"], list):
            for key in self.configured_args.keys():
                if key in parameters["required"]:
                    parameters["required"].remove(key)
        return skill_def

    @property
    def __name__(self) -> str:
        return self.wrapper.__name__

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # Merge the run-time provided arguments with the pre-bound ones.
        # Here the pre-bound (configured) arguments override any arguments with the same name.
        merged = {**kwargs, **self.configured_args}
        return await self.wrapper.inner_func(**merged)

    def __await__(self) -> Any:
        return self.__call__().__await__()


class SkillWrapper:
    def __init__(
        self,
        func: Callable[..., Any],
        inner_func: Callable[..., Awaitable[Any]],
        skill_definition: Dict[str, Any],
    ):
        self.func = func
        self.inner_func = inner_func
        self._skill_definition = skill_definition

    @property
    def skill_definition(self) -> Dict[str, Any]:
        return self._skill_definition

    @property
    def __name__(self) -> str:
        return self.func.__name__

    def __call__(self, **kwargs: Any) -> ConfiguredSkillCall:
        # If someone "configures" the skill by calling it with parameters,
        # return a ConfiguredSkillCall.
        return ConfiguredSkillCall(self, kwargs)


def skill(input_model: Type[T]) -> Callable[[Callable[..., Any]], SkillWrapper]:
    """
    Decorator for creating Arsenal skills with automatic input validation.

    This decorator wraps a function to:
    1. Validate inputs using a Pydantic model.
    2. Attach the necessary skill definition for agent use.
    3. Handle both async and sync functions.
    4. Support configuration calls (so you can write e.g. search(llm_rewrite=False)).
    """

    def decorator(fn: Callable[..., Any]) -> SkillWrapper:
        async def _inner(**kwargs: Any) -> Any:
            instance = input_model(**kwargs)
            return (
                await fn(**instance.model_dump())
                if inspect.iscoroutinefunction(fn)
                else fn(**instance.model_dump())
            )

        skill_def: Dict[str, Any] = {
            "type": "function",
            "function": {
                "name": fn.__name__,
                "description": fn.__doc__ or "",
                "parameters": input_model.model_json_schema(),
            },
        }
        # Return a SkillWrapper that lets you optionally configure the skill.
        return SkillWrapper(fn, _inner, skill_def)

    return decorator
