# experiments_fuels/only_one_agent.py
import os
import sys
import json
import asyncio
import numpy as np
import polars as pl
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.LLM_agent import LLMAgent
from src.experiment.experiment import Experiment
from src.prompts.prompts import GENERAL_PROMPT, P1
from src.prompts.prompts_models import create_pricing_response_model
from src.environment.calvano import CalvanoDemandEnvironment
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
current_file_path = Path(__file__).resolve()

load_dotenv()
API_KEY = os.getenv("MISTRAL_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
DATA_DIR = PROJECT_ROOT / "data/processed"

marginal_costs = (
    pl.read_parquet(DATA_DIR / "marginal_costs_tgp.parquet")["tgpmin"]
    .to_numpy()
    .flatten()
)

MEMORY_LENGTH = 100
N_ROUNDS = len(marginal_costs)
N_RUNS = 1
ALPHAS_TO_TRY = [1]

with open(DATA_DIR / "initial_real_data_to_inject_as_history.json", "r") as f:
    initial_real_data = json.load(f)


async def main(alpha=1):
    PricingAgentResponse = create_pricing_response_model(
        include_wtp=True, wtp_value=2 * alpha
    )
    cost_series = np.tile(
        marginal_costs, (4, 1)
    )  # NOTE! SHOULD BE IN THE SAME ORDER AS AGENTS

    # NOTE! BRAND EFFECTS!
    # (2.45, 2.13, 2.13, 2.0)
    # Load from config or pass manually
    agents = [
        LLMAgent(
            "BP",
            prefix=P1,
            api_key=API_KEY,
            model_name=MODEL_NAME,
            response_model=PricingAgentResponse,
            memory_length=MEMORY_LENGTH,
            prompt_template=GENERAL_PROMPT,
            env_params={"a": 1.0, "alpha": 1.0, "c": 1.0},
        ),
        LLMAgent(
            "Caltex",
            prefix=P1,
            api_key=API_KEY,
            model_name=MODEL_NAME,
            response_model=PricingAgentResponse,
            memory_length=MEMORY_LENGTH,
            prompt_template=GENERAL_PROMPT,
            env_params={"a": 1.0, "alpha": 1.0, "c": 1.0},
        ),
        LLMAgent(
            "Woolworths",
            prefix=P1,
            api_key=API_KEY,
            model_name=MODEL_NAME,
            response_model=PricingAgentResponse,
            memory_length=MEMORY_LENGTH,
            prompt_template=GENERAL_PROMPT,
            env_params={"a": 1.0, "alpha": 1.0, "c": 1.0},
        ),
        LLMAgent(
            "Coles",
            prefix=P1,
            api_key=API_KEY,
            model_name=MODEL_NAME,
            response_model=PricingAgentResponse,
            memory_length=MEMORY_LENGTH,
            prompt_template=GENERAL_PROMPT,
            env_params={"a": 1.0, "alpha": 1.0, "c": 1.0},
        ),
    ]

    env = CalvanoDemandEnvironment(
        name="Calvano Market",
        description="Oligopoly environment with Calvano 2020 demand",
    )

    experiment = Experiment(
        name="oligopoly_setting_all_agents_P1_all_ai_1",
        agents=agents,
        num_rounds=N_ROUNDS,
        environment=env,
        cost_series=cost_series,
        initial_real_data=initial_real_data,
        experiment_dir=current_file_path.parent / "experiments_runs",
        experiment_plot=False,
    )
    await experiment.run()


if __name__ == "__main__":
    for _ in range(N_RUNS):
        for alpha in ALPHAS_TO_TRY:
            print(f"Running experiment with alpha={alpha}")
            asyncio.run(main(alpha))
            print(f"Experiment with alpha={alpha} completed.\n")
