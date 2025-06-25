import os
import asyncio
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.agents.base_agent import Agent
from src.experiment.experiment import Experiment
from src.prompts.prompts import GENERAL_PROMPT, P1, P2
from src.prompts.prompts_models import PricingAgentResponse

load_dotenv()
API_KEY = os.getenv("MISTRAL_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")


async def main():
    # Load from config or pass manually
    agents = [
        Agent("Firm A", 
              prefix = P1,
              api_key = API_KEY, 
              model_name=MODEL_NAME, 
              response_model=PricingAgentResponse, 
              memory_length=2, 
              prompt_template=GENERAL_PROMPT
              ),
        Agent("Firm B", 
              prefix = P1,
              api_key = API_KEY, 
              model_name=MODEL_NAME, 
              response_model=PricingAgentResponse, 
              memory_length=2, 
              prompt_template=GENERAL_PROMPT
              ),
    ]

    experiment = Experiment(name="simple_market", agents=agents, num_rounds=5)
    await experiment.run()



if __name__ == "__main__":
    asyncio.run(main())