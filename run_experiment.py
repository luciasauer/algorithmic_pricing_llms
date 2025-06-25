import os
import asyncio
import numpy as np
from dotenv import load_dotenv

from src.agents.base_agent import Agent
from src.experiment.experiment import Experiment
from src.prompts.prompts import GENERAL_PROMPT, P1, P2
from src.prompts.prompts_models import PricingAgentResponse
from src.environment.environment import CalvanoDemandEnvironment

load_dotenv()
API_KEY = os.getenv("MISTRAL_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")


async def main():

    env_params = {
        "a_0": 0.0,
        "a": np.array([2.0, 2.0]),
        "mu": 0.25,
        "alpha": np.array([1.0, 1.0]),
        "beta": 100,
        "sigma": 0.0,
        "c": np.array([1.0, 1.0]),
        "group_idxs": (1,2),  # Assuming two agents, adjust as needed
    }

    env = CalvanoDemandEnvironment(
        name="Calvano Market",
        description="Simple duopoly environment with linear demand",
        nbr_agents=2,
        env_params=env_params,
    )
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
    
    experiment = Experiment(name="simple_market", agents=agents, num_rounds=5, environment=env)
    await experiment.run()



if __name__ == "__main__":
    asyncio.run(main())