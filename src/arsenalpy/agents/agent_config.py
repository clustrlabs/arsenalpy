from typing import Any, Dict, Literal, Optional, Union

from pydantic import BaseModel


class AgentConfig(BaseModel):
    """Configuration settings for an Agent."""

    temperature: Optional[float] = 0.5
    max_completion_tokens: Optional[int] = None
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    response_format: Optional[Dict[str, Any]] = None
    reasoning_effort: Optional[Literal["medium", "high", "low"]] = None
    store: Optional[bool] = False
    tool_choice: Optional[
        Union[Literal["none", "auto", "required"], Dict[str, Any]]
    ] = "auto"

    def to_api_params(self) -> dict:
        """Converts the configuration to API parameters."""
        params = self.model_dump(exclude_none=True)

        # Handle tool_choice parameter
        if "tool_choice" in params:
            if params["tool_choice"] not in ["none", "auto", "required"]:
                # If it's a dict specifying a function, ensure it has the right format
                if not isinstance(params["tool_choice"], dict):
                    raise ValueError(
                        "tool_choice must be 'none', 'auto', 'required' or a function specification dict"
                    )
                if "function" not in params["tool_choice"]:
                    params["tool_choice"] = {
                        "type": "function",
                        "function": params["tool_choice"],
                    }

        return params
