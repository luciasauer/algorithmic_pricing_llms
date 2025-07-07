"""
Prompt Management for LLM Agent Decision-Making

This module handles dynamic prompt generation with context injection,
memory management, and market history formatting for LLM agents.
"""

import string
import logging
from typing import Type, List
from pydantic import BaseModel
from src.agents.base_agent import Agent


class PromptManager:
    """
    Manages dynamic prompt generation and context injection for LLM agents.

    Handles memory window management, market history formatting, and
    strategic context injection for agent decision-making.

    Args:
        logger: Logger instance for experiment tracking
    """

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger("experiment_logger")

    def generate_prompt(self, agent: Type[Agent], history: dict, round_num: int) -> str:
        """
        Generate contextualized prompt for agent decision-making.

        Creates prompt by injecting market history, benchmarks, and strategic
        context into the agent's prompt template.

        Args:
            agent: Agent instance requiring prompt
            history: Market history data
            round_num: Current round number

        Returns:
            Formatted prompt string with injected context
        """
        if not getattr(agent, "requires_prompt", True):  # for fake agents
            return ""
        prompt_body = agent.prompt_template
        prompt_fields = self._extract_prompt_fields(prompt_body)
        memory_fields = self._get_fields_to_remember(agent.response_model)

        past_data = self._get_memory_field_data(
            history, memory_fields, agent.memory_length
        )
        prompt_data = self._build_prompt_data(
            agent, history, prompt_fields, past_data, round_num
        )

        return prompt_body.format(**prompt_data)

    def _extract_prompt_fields(self, template: str) -> List[str]:
        formatter = string.Formatter()
        return [field for _, field, _, _ in formatter.parse(template) if field]

    def _get_fields_to_remember(self, model_cls: type[BaseModel]) -> List[str]:
        return [
            name
            for name, field in model_cls.model_fields.items()
            if field.json_schema_extra.get("keep_memory", True)
        ]

    def _get_memory_field_data(
        self, history: dict, memory_fields: List[str], memory_length: int
    ) -> dict:
        memory_data = {}
        for field in memory_fields:
            try:
                memory_val = [
                    (round_num, past_round[field])
                    for round_num, past_round in sorted(history.items())
                    if field in past_round
                ][-memory_length:]

                memory_data[field] = "\n".join(
                    f"Round {round_num}\n{value}"
                    for round_num, value in reversed(memory_val)
                )
            except KeyError as e:
                self.logger.error(f"Field '{field}' not found in history. {e}")
                raise
        return memory_data

    def _build_prompt_data(
        self,
        agent: Agent,
        history: dict,
        prompt_fields: List[str],
        memory_data: dict,
        round_num: int,
    ) -> dict:
        prompt_data = {}
        for field in prompt_fields:
            if field in memory_data:
                prompt_data[field] = memory_data[field]
            elif field == "marginal_cost":
                prompt_data["marginal_cost"] = agent.get_marginal_cost(round_num)
            elif history and field in history[round_num - 1]:
                prompt_data[field] = history[round_num - 1][field]
            elif field in agent.response_model.model_fields:
                prompt_data[field] = agent.response_model.model_fields[field].default
            else:
                prompt_data[field] = "No previous data available."

        return prompt_data
