import string
import logging
from typing import Type, List
from pydantic import BaseModel
from src.agents.base_agent import Agent

class PromptManager:
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger("experiment_logger")

    def generate_prompt(self, agent: Type[Agent], history: dict) -> str:
        prompt_body = agent.prompt_template
        prompt_fields = self._extract_prompt_fields(prompt_body)
        memory_fields = self._get_fields_to_remember(agent.response_model)

        past_data = self._get_memory_field_data(history, memory_fields, agent.memory_length)
        prompt_data = self._build_prompt_data(agent, history, prompt_fields, past_data)

        return prompt_body.format(**prompt_data)

    def _extract_prompt_fields(self, template: str) -> List[str]:
        formatter = string.Formatter()
        return [field for _, field, _, _ in formatter.parse(template) if field]

    def _get_fields_to_remember(self, model_cls: type[BaseModel]) -> List[str]:
        return [
            name for name, field in model_cls.model_fields.items()
            if field.json_schema_extra.get("keep_memory", True)
        ]

    def _get_memory_field_data(self, history: dict, memory_fields: List[str], memory_length: int) -> dict:
        memory_data = {}
        for field in memory_fields:
            try:
                memory_val = [
                    (round_num, past_round[field])
                    for round_num, past_round in sorted(history.items())
                    if field in past_round
                ][-memory_length:]

                memory_data[field] = "\n".join(
                    f"Round {round_num}\n{value}" for round_num, value in reversed(memory_val)
                )
            except KeyError as e:
                self.logger.error(f"Field '{field}' not found in history. {e}")
                raise
        return memory_data

    def _build_prompt_data(self, agent: Agent, history: dict, prompt_fields: List[str], memory_data: dict) -> dict:
        prompt_data = {}

        for field in prompt_fields:
            if field in memory_data:
                prompt_data[field] = memory_data[field]
            elif history and field in history:
                latest_round = list(history[field].keys())[-1]
                prompt_data[field] = history[field][latest_round]
            elif field in agent.response_model.model_fields:
                prompt_data[field] = agent.response_model.model_fields[field].default
            else:
                prompt_data[field] = "No previous data available."

        return prompt_data

