#src/agents/LLM_agent.py
import json
import logging
import asyncio

from mistralai import Mistral
from typing import Dict, Type
from pydantic import BaseModel, ValidationError, create_model

from src.agents.base_agent import Agent

MAX_RETRIES = 10
RETRY_DELAY_SECONDS = 2

class LLMAgent(Agent):
    def __init__(self, name: str, prefix: str, api_key: str, model_name: str,
                 response_model: Type[BaseModel], prompt_template = None, memory_length:int=100, 
                 env_params: dict = None, logger: logging.Logger = None):
        
        super().__init__(name=name, prefix=prefix, prompt_template=prompt_template, env_params=env_params, logger=logger)

        self.api_key = api_key
        self.model_name = model_name
        self.response_model = response_model
        self.reponse_model_in_answer = self.__filter_in_answer_fields(response_model)
        self.prompt_template = prompt_template
        self.memory_length = memory_length
        self.logger = logger or logging.getLogger("experiment_logger")
        self.reponse_model_str = (
                        "Respond only with a JSON object with this schema:\n{\n" +
                        "\n".join([f'  "{name}": {field.annotation.__name__}'
                            for name, field in response_model.model_fields.items()
                            if field.json_schema_extra.get("in_answer", True)
                        ]) + "\n}")

    async def act(self, prompt: str) -> Dict:
        async with Mistral(api_key=self.api_key) as client:
            for attempt in range(1, MAX_RETRIES + 1):
                self.logger.info(f"🔄 Attempt {attempt} for agent {self.name}")
                try:
                    response = await client.chat.complete_async(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": self.reponse_model_str},
                            {"role": "system", "content": self.prefix},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.7,
                        response_format={"type": "json_object"},
                    )
                    parsed_json = json.loads(response.choices[0].message.content)

                    ## Validate using Pydantic
                    validated = self.reponse_model_in_answer(**parsed_json)

                    return {'agent_name':self.name, 'content':validated.dict()}

                except (ValidationError, ValueError) as e:
                    if attempt == MAX_RETRIES:
                        self.logger.error(f"❌ Validation failed after {MAX_RETRIES} attempts for agent {self.name}: {str(e)}")
                        raise e
                except Exception as e:
                    if attempt == MAX_RETRIES:
                        self.logger.error(f"❌ Error after {MAX_RETRIES} attempts for agent {self.name}: {str(e)}")
                        raise e
                await asyncio.sleep(RETRY_DELAY_SECONDS * attempt)
    
    def __filter_in_answer_fields(self, model: Type[BaseModel]) -> Type[BaseModel]:
        # Filter the fields that have 'in_answer=True'
        in_answer_fields = {
            name: (field.annotation, field.default)
            for name, field in model.model_fields.items()
            if field.json_schema_extra.get("in_answer", False)
        }
        # Create a new model pydantic with only the filtered fields
        return create_model(model.__name__ + "Filtered", **in_answer_fields)
    
    @property
    def type(self) -> bool:
        return "LLM_agent"