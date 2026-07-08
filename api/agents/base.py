"""Base class for all analytics agents."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

import re

from openai import AsyncOpenAI
from utils.logging import get_logger

logger = get_logger(__name__)


def _parse_json(raw: str, agent_name: str = "") -> dict[str, Any]:
    """Parse JSON from LLM output, handling markdown code blocks."""
    text = raw.strip()

    # Strip ```json ... ``` or ``` ... ``` wrappers (common with OpenRouter models)
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if match:
        text = match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Last resort: find the first {...} block in the response
        brace = text.find("{")
        if brace != -1:
            try:
                return json.loads(text[brace:])
            except json.JSONDecodeError:
                pass
        logger.error("Failed to parse LLM JSON", agent=agent_name, raw=raw[:300])
        return {}


class BaseAgent(ABC):
    """All agents share a common LLM client and call pattern."""

    name: str = "BaseAgent"

    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self.client = client
        self.model = model

    @abstractmethod
    async def run(self, state: dict) -> dict:
        """Execute the agent and return a partial state update."""

    async def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = True,
    ) -> dict[str, Any]:
        logger.info("LLM call", agent=self.name, model=self.model)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = await self.client.chat.completions.create(**kwargs)
        raw = response.choices[0].message.content or "{}"

        logger.info("LLM response received", agent=self.name, tokens=response.usage.total_tokens)

        return _parse_json(raw, self.name)
