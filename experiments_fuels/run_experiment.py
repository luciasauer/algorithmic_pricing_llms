#experiments_fuels/run_experiment.py
import os
import sys
import asyncio
import numpy as np
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.LLM_agent import LLMAgent
from src.agents.fake_agent import FakeAgent
from src.experiment.experiment import Experiment
from src.prompts.prompts import GENERAL_PROMPT, P1, P2
from src.prompts.prompts_models import create_pricing_response_model
from src.environment.environment import CalvanoDemandEnvironment
from pathlib import Path

current_file_path = Path(__file__).resolve()

load_dotenv()
API_KEY = os.getenv("MISTRAL_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

MEMORY_LENGTH = 100 
N_ROUNDS = 300
N_RUNS = 7
ALPHAS_TO_TRY = [1, 3.2, 10]


async def main(alpha=1):

    PricingAgentResponse = create_pricing_response_model(include_wtp=True, wtp_value=4.51 * alpha)
    cost_series = np.array([
        np.linspace(1.0, 2.0, N_RUNS),     # Firm A
        np.linspace(1.2, 1.8, N_RUNS),     # Firm B
        np.ones(N_RUNS) * 1.5,             # Firm C
    ])

    # Load from config or pass manually
    agents = [
        LLMAgent("Firm A", 
              prefix=P1,
              api_key=API_KEY, 
              model_name=MODEL_NAME, 
              response_model=PricingAgentResponse, 
              memory_length=MEMORY_LENGTH, 
              prompt_template=GENERAL_PROMPT,
              env_params={"a": 2.0, "alpha": 1.0, "c": 1.0},
              ),
        LLMAgent("Firm B", 
              prefix=P1,
              api_key=API_KEY, 
              model_name=MODEL_NAME, 
              response_model=PricingAgentResponse, 
              memory_length=MEMORY_LENGTH, 
              prompt_template=GENERAL_PROMPT,
              env_params={"a": 2.0, "alpha": 1.0, "c": 1.0},
              ),
        FakeAgent("Firm C", time_series_data=np.array([2]*(N_RUNS)), nbr_rounds=N_RUNS, env_params={"a": 1.0, "alpha": 1.0, "c": 1.0},),
    ]

    env = CalvanoDemandEnvironment(
        name="Calvano Market",
        description="Oligopoly environment with Calvano 2020 demand",
    )

    experiment = Experiment(name="oligopoly_setting", agents=agents, num_rounds=N_RUNS, environment=env,
                            cost_series=cost_series,
                            experiment_dir=current_file_path.parent / "experiments_runs",
                            )
    await experiment.run()



if __name__ == "__main__":
    for _ in range(N_RUNS):
        for alpha in ALPHAS_TO_TRY:
            print(f"Running experiment with alpha={alpha}")
            asyncio.run(main(alpha))
            print(f"Experiment with alpha={alpha} completed.\n")
