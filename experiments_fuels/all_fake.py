# experiments_fuels/run_experiment.py
import os
import sys
import asyncio
import numpy as np
import polars as pl
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.fake_agent import FakeAgent
from src.experiment.experiment import Experiment
from src.environment.calvano import CalvanoDemandEnvironment
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


async def main(alpha=1):
    cost_series = np.tile(
        marginal_costs, (4, 1)
    )  # NOTE! SHOULD BE IN THE SAME ORDER AS AGENTS

    print("marginal_costs.shape:", getattr(cost_series, "shape", type(marginal_costs)))
    print("cost_series.shape:", cost_series.shape)
    # print("Expected shape:", (len(agents), N_ROUNDS))

    # Load from config or pass manually
    agents = [
        FakeAgent(
            "BP",
            time_series_data=bp_prices,
            nbr_rounds=N_RUNS,
            env_params={"a": 1.0, "alpha": 1.0, "c": 1.0},
        ),
        FakeAgent(
            "Caltex",
            time_series_data=caltex_prices,
            nbr_rounds=N_RUNS,
            env_params={"a": 1.0, "alpha": 1.0, "c": 1.0},
        ),
        FakeAgent(
            "Coles",
            time_series_data=coles_prices,
            nbr_rounds=N_RUNS,
            env_params={"a": 1.0, "alpha": 1.0, "c": 1.0},
        ),
        FakeAgent(
            "Woolworths",
            time_series_data=woolworths_prices,
            nbr_rounds=N_RUNS,
            env_params={"a": 1.0, "alpha": 1.0, "c": 1.0},
        ),
        # FakeAgent("Gull", time_series_data=gull_prices, nbr_rounds=N_RUNS, env_params={"a": 1.0, "alpha": 1.0, "c": 1.0},),
    ]

    env = CalvanoDemandEnvironment(
        name="Calvano Market",
        description="Oligopoly environment with Calvano 2020 demand",
    )

    experiment = Experiment(
        name="oligopoly_setting",
        agents=agents,
        num_rounds=N_ROUNDS,
        environment=env,
        cost_series=cost_series,
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
