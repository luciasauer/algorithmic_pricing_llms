import asyncio
import json
from typing import Dict
from mistralai import Mistral
from typing import Type
from pydantic import BaseModel
from pydantic import ValidationError

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 1

class Agent:
    def __init__(self, name: str, prefix: str, api_key: str, model_name: str, response_model: Type[BaseModel]):
        self.name = name
        self.prefix = prefix
        self.api_key = api_key
        self.model_name = model_name
        self.response_model = response_model
        self.reponse_model = (
                        "Respond only with a JSON object with this schema:\n{\n" +
                        "\n".join([f'  "{name}": {field.annotation.__name__}'
                            for name, field in response_model.model_fields.items()
                        ]) + "\n}")

    async def act(self, prompt: str) -> Dict:
        async with Mistral(api_key=self.api_key) as client:
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    response = await client.chat.complete_async(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": self.reponse_model},
                            {"role": "system", "content": self.prefix},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.7,
                        response_format={"type": "json_object"},
                    )
                    parsed_json = json.loads(response.choices[0].message.content)

                    ## Validate using Pydantic
                    validated = self.response_model(**parsed_json)

                    result = {
                        "agent": self.name,
                        "response": validated.dict()
                    }
                    return result

                except (ValidationError, ValueError) as e:
                    if attempt == MAX_RETRIES:
                        return {"firm": self.name, "error": f"Validation failed after {MAX_RETRIES} attempts: {str(e)}"}
                except Exception as e:
                    if attempt == MAX_RETRIES:
                        return {"firm": self.name, "error": f"Error after {MAX_RETRIES} attempts: {str(e)}"}
                await asyncio.sleep(RETRY_DELAY_SECONDS * attempt)