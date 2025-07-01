# experiments_synthetic/run_experiment.py
import asyncio
import os
import sys

from dotenv import load_dotenv

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path

from src.agents.LLM_agent import LLMAgent
from src.environment.calvano import CalvanoDemandEnvironment
from src.experiment.experiment import Experiment
from src.prompts.prompts import GENERAL_PROMPT, P1
from src.prompts.prompts_models import create_pricing_response_model
from src.utils.cost_generators import create_step_shock_series

current_file_path = Path(__file__).resolve()

load_dotenv()
API_KEY = os.getenv("MISTRAL_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

MEMORY_LENGTH = 100
N_ROUNDS = 300
N_RUNS = 7
ALPHAS_TO_TRY = [1]  # , 3.2, 10]


async def main(alpha=1):
    PricingAgentResponse = create_pricing_response_model(
        include_wtp=True, wtp_value=4.51 * alpha
    )
    agents = [
        LLMAgent(
            "Firm A",
            prefix=P1,
            api_key=API_KEY,
            model_name=MODEL_NAME,
            response_model=PricingAgentResponse,
            memory_length=MEMORY_LENGTH,
            prompt_template=GENERAL_PROMPT,
            env_params={"a": 2.0, "alpha": alpha, "c": 1.0},
        ),
    ]

    env = CalvanoDemandEnvironment(
        name="Calvano Market",
        description="Monopoly environment with Calvano 2020 demand",
    )

    # Create cost series with shock
    cost_series = create_step_shock_series(
        n_agents=len(agents),
        n_rounds=N_ROUNDS,
        shock_round=50,  # After first third (convergence normally afer 60-80 rounds)
        base_cost=1.0,
        shock_magnitude=2,
    )

    experiment = Experiment(
        name="monopoly_shock_experiment",
        agents=agents,
        num_rounds=N_ROUNDS,
        environment=env,
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
