import string
import logging
from typing import Type, List
from pydantic import BaseModel

from src.agents.base_agent import Agent


class PromptManager:
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger("experiment_logger")


    def generate_prompt(self, agent: Type[Agent], history: dict) -> str:
        memory_fields = self._get_fields_to_remember(agent.response_model)
        past_rounds_data = self._get_past_rounds_data(history, memory_fields, agent.memory_length)

        prompt_body = agent.prompt_template
        prompt_fields = self._extract_prompt_fields(prompt_body)
        prompt_data = {field: None for field in prompt_fields}

        missing_fields = set(prompt_fields) - set(past_rounds_data.keys())
        #Generate fields data, to then add to the promtp
        for field in past_rounds_data.keys():
            prompt_data[field] = past_rounds_data[field]

        for field in missing_fields:
            if len(history) > 0:
                if field in history:
                    last_value = history[field][list(history[field].keys())[-1]]
                    prompt_data[field] = last_value
                elif field in agent.response_model.__fields__:
                    default_value = agent.response_model.__fields__[field].default
                    prompt_data[field] = default_value
            else:
                prompt_data[field] = "No previous data available."

        return prompt_body.format(**prompt_data)

    def _extract_prompt_fields(self, template: str) -> list:
        formatter = string.Formatter()
        fields = [field_name for _, field_name, _, _ in formatter.parse(template) if field_name]
        return list(fields)
    
    def _get_past_rounds_data(self, history: dict, memory_fields: List[str], memory_length: int) -> dict:
        try:
            previous_values = {}
            for field in memory_fields:
                memory_val = [
                    (round_num, past_round[field]) for round_num, past_round in sorted(history.items())
                    if field in past_round
                ][-memory_length:]

                concatenated_data = "\n".join([f"Round {round_num} \n {value}\n" 
                                               for round_num, value in memory_val[::-1]])
                previous_values[field] = concatenated_data

            return previous_values
        except KeyError as e:
            self.logger.error(f"KeyError: {e} - Check if the field exists in the history.")
            raise KeyError(f"Field '{field}' not found in history.") from e
        

    def _get_fields_to_remember(self, model_cls: type[BaseModel]) -> list[str]:
        return [
            name for name, field in model_cls.model_fields.items()
            if field.json_schema_extra.get("keep_memory", True)
        ]
    
    def _concat_past_data(self, past_data: List[str]) -> str:
        if not past_data:
            return "No past data available."
        return "\n".join(past_data)

    

