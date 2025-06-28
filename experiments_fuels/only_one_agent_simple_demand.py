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
from src.agents.fake_agent import FakeAgent
from src.experiment.experiment import Experiment
from src.prompts.prompts import GENERAL_PROMPT, P1C
from src.prompts.prompts_models import create_pricing_response_model
from src.environment.penalty_demand_environment import PenaltyDemandEnvironment
from pathlib import Path

current_file_path = Path(__file__).resolve()

load_dotenv()
API_KEY = os.getenv("MISTRAL_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")


marginal_costs = (
    pl.read_parquet("experiments_fuels/data/marginal_costs.parquet")["tgpmin"]
    .to_numpy()
    .flatten()
    / 100
)
bp_prices = (
    pl.read_parquet("experiments_fuels/data/bp_prices.parquet")["avg_price"]
    .to_numpy()
    .flatten()
    / 100
)
caltex_prices = (
    pl.read_parquet("experiments_fuels/data/caltex_prices.parquet")["avg_price"]
    .to_numpy()
    .flatten()
    / 100
)
coles_prices = (
    pl.read_parquet("experiments_fuels/data/coles_prices.parquet")["avg_price"]
    .to_numpy()
    .flatten()
    / 100
)
woolworths_prices = (
    pl.read_parquet("experiments_fuels/data/woolworths_prices.parquet")["avg_price"]
    .to_numpy()
    .flatten()
    / 100
)
gull_prices = (
    pl.read_parquet("experiments_fuels/data/gull_prices.parquet")["avg_price"]
    .to_numpy()
    .flatten()
    / 100
)


MEMORY_LENGTH = 100
N_ROUNDS = len(marginal_costs)
N_RUNS = 1
ALPHAS_TO_TRY = [1]

with open("experiments_fuels/data/initial_real_data.json", "r") as f:
    initial_real_data = json.load(f)


async def main(alpha=1):
    PricingAgentResponse = create_pricing_response_model(
        include_wtp=True, wtp_value=2 * alpha
    )
    cost_series = np.tile(
        marginal_costs, (4, 1)
    )  # NOTE! SHOULD BE IN THE SAME ORDER AS AGENTS

    print("marginal_costs.shape:", getattr(cost_series, "shape", type(marginal_costs)))
    print("cost_series.shape:", cost_series.shape)
    # print("Expected shape:", (len(agents), N_ROUNDS))

    # NOTE! BRAND EFFECTS!
    # (2.45, 2.13, 2.13, 2.0)
    # MARKET SHARES NORMALIZED TO 1
    # (0.22, 0.16, 0.16, 0.14)
    # (0.323, 0.235, 0.235, 0.207)

    # Load from config or pass manually
    agents = [
        FakeAgent(
            "BP",
            time_series_data=bp_prices,
            nbr_rounds=N_RUNS,
            env_params={"a": 2.0, "alpha": 1.0, "c": 1.0, "market_share": 0.323},
        ),
        FakeAgent(
            "Caltex",
            time_series_data=caltex_prices,
            nbr_rounds=N_RUNS,
            env_params={"a": 2.0, "alpha": 1.0, "c": 1.0, "market_share": 0.235},
        ),
        FakeAgent(
            "Woolworths",
            time_series_data=woolworths_prices,
            nbr_rounds=N_RUNS,
            env_params={"a": 2.0, "alpha": 1.0, "c": 1.0, "market_share": 0.235},
        ),
        LLMAgent(
            "Coles",
            prefix=P1C,
            api_key=API_KEY,
            model_name=MODEL_NAME,
            response_model=PricingAgentResponse,
            memory_length=MEMORY_LENGTH,
            prompt_template=GENERAL_PROMPT,
            env_params={"a": 2.0, "alpha": 1.0, "c": 1.0, "market_share": 0.207},
        ),
    ]

    env = PenaltyDemandEnvironment(
        name="Penalty Market",
        description="Oligopoly with penalty on price deviation from competitor average",
        penalty_lambda=0.0622,
    )

    experiment = Experiment(
        name="oligopoly_experiment_one_agent",
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
