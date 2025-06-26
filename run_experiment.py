import os
import asyncio
import numpy as np
from dotenv import load_dotenv

from src.agents.LLM_agent import LLMAgent
from src.agents.fake_agent import FakeAgent
from src.experiment.experiment import Experiment
from src.prompts.prompts import GENERAL_PROMPT, P1, P2
# from src.prompts.prompts_models import PricingAgentResponse
from src.environment.environment import CalvanoDemandEnvironment

load_dotenv()
API_KEY = os.getenv("MISTRAL_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

MEMORY_LENGTH = 100 
N_RUNS = 100

async def main():
    alpha = 1

    env_params = {
        "a_0": 0.0,
        "a": np.array([2.0, 2.0, 2.0]),
        "mu": 0.25,
        "alpha": np.array([alpha, alpha, alpha]),
        "beta": 100,
        "sigma": 0.0,
        "c": np.array([1.0, 1.0,1.0]),
        "group_idxs": (1,2,3),
    }

    from pydantic import BaseModel, Field

    class PricingAgentResponse(BaseModel):
        observations: str = Field(..., in_answer=True,keep_memory=False)
        plans: str = Field(..., in_answer=True, keep_memory=False)
        insights: str = Field(..., in_answer=True, keep_memory=False)
        chosen_price: float = Field(..., in_answer=True, keep_memory=False)
        market_data: str = Field(..., in_answer=False, keep_memory=True)
        marginal_cost: float = Field(1, in_answer=False, keep_memory=False)
        willigness_to_pay: float = Field(4.51*alpha, in_answer=False, keep_memory=False)

    env = CalvanoDemandEnvironment(
        name="Calvano Market",
        description="Duopoly environment with Calvano 2020 demand",
        nbr_agents=2,
        env_params=env_params,
    )
    # Load from config or pass manually
    agents = [
        LLMAgent("Firm A", 
              prefix=P1,
              api_key=API_KEY, 
              model_name=MODEL_NAME, 
              response_model=PricingAgentResponse, 
              memory_length=MEMORY_LENGTH, 
              prompt_template=GENERAL_PROMPT
              ),
        LLMAgent("Firm B", 
              prefix=P1,
              api_key=API_KEY, 
              model_name=MODEL_NAME, 
              response_model=PricingAgentResponse, 
              memory_length=MEMORY_LENGTH, 
              prompt_template=GENERAL_PROMPT
              ),
        FakeAgent("Firm C", time_series_data=np.array([2]*(N_RUNS)), nbr_rounds=N_RUNS,),
    ]

    experiment = Experiment(name="oligopoly_setting", agents=agents, num_rounds=300, environment=env)
    await experiment.run()



if __name__ == "__main__":
    asyncio.run(main())
