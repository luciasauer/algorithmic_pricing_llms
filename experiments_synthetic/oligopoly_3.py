# experiments_synthetic/oligopoly_3.py
"""
3-agent oligopoly algorithmic pricing experiment implementation.

This script runs a 3-agent algorithmic pricing experiment testing Folk Theorem
predictions about coordination breakdown as market concentration decreases.
Three LLM agents compete over multiple rounds using the Calvano et al. (2020)
demand specification.

The experiment tests the first step in coordination breakdown: moving from
duopoly (n=2) to 3-agent competition. According to Folk Theorem predictions,
coordination should become more difficult with additional competitors.

Research Question:
    How does coordination change when moving from 2 to 3 competing agents?
    Do we observe the predicted price reduction as market concentration decreases?

Experimental Design:
    - 3 competing LLM agents (Firm A, Firm B, Firm C)
    - 300 rounds per experiment for convergence analysis
    - Calvano demand environment with nested logit specification
    - 100-period rolling memory for strategic learning
    - Multiple alpha parameter values to test robustness

Folk Theorem Prediction:
    Prices should be systematically lower than in duopoly experiments due to:
    - Increased coordination complexity
    - Greater incentive for individual deviation
    - Lower per-firm collusive profits (monopoly profit divided by more firms)

Expected Outcomes:
    - Pricing below duopoly levels but above Nash equilibrium
    - Statistically significant group size effect (-3.7% per additional competitor)
    - Maintained evidence of supracompetitive pricing despite increased competition

Usage:
    python experiments_synthetic/oligopoly_3.py

Requirements:
    - MISTRAL_API_KEY environment variable set
    - MODEL_NAME environment variable (default: mistral-large-2411)
    - Sufficient API credits for ~3150 API calls per full experiment run

Output:
    Results saved to data/results/ in Parquet format for statistical analysis.
    Critical for Folk Theorem testing and group size effect estimation.
"""

import os
import sys
import asyncio
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.LLM_agent import LLMAgent
from src.experiment.experiment import Experiment
from src.prompts.prompts import GENERAL_PROMPT, P1
from src.prompts.prompts_models import create_pricing_response_model

from src.environment.calvano import CalvanoDemandEnvironment
from pathlib import Path

current_file_path = Path(__file__).resolve()

load_dotenv()
API_KEY = os.getenv("MISTRAL_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

# Experimental Configuration Parameters
MEMORY_LENGTH = 100  # Number of past rounds agents remember for decision-making
N_ROUNDS = 300  # Length of each pricing game (allows for convergence analysis)
N_RUNS = 7  # Number of repetitions for statistical significance
ALPHAS_TO_TRY = [1, 3.2, 10]  # Demand parameter variations for robustness testing


async def main(prompt_prefix, alpha=1, experiment_name="oligopoly_setting_3_firms"):
    PricingAgentResponse = create_pricing_response_model(
        include_wtp=True, wtp_value=4.51 * alpha
    )
    agents = [
        LLMAgent(
            "Firm A",
            prefix=prompt_prefix,
            api_key=API_KEY,
            model_name=MODEL_NAME,
            response_model=PricingAgentResponse,
            memory_length=MEMORY_LENGTH,
            prompt_template=GENERAL_PROMPT,
            env_params={"a": 2.0, "alpha": alpha, "c": 1.0},
        ),
        LLMAgent(
            "Firm B",
            prefix=prompt_prefix,
            api_key=API_KEY,
            model_name=MODEL_NAME,
            response_model=PricingAgentResponse,
            memory_length=MEMORY_LENGTH,
            prompt_template=GENERAL_PROMPT,
            env_params={"a": 2.0, "alpha": alpha, "c": 1.0},
        ),
        LLMAgent(
            "Firm C",
            prefix=prompt_prefix,
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
        description="Oligopoly environment with Calvano 2020 demand",
    )

    experiment = Experiment(
        name=experiment_name,
        agents=agents,
        num_rounds=N_ROUNDS,
        environment=env,
        experiment_dir=current_file_path.parent / "experiments_runs",
    )
    await experiment.run()


if __name__ == "__main__":
    for _ in range(N_RUNS):
        for alpha in ALPHAS_TO_TRY:
            print(f"Running experiment with alpha={alpha}")
            asyncio.run(
                main(
                    prompt_prefix=P1,
                    alpha=alpha,
                    experiment_name="oligopoly_setting_3_firms_P1",
                )
            )
            print(f"Experiment with alpha={alpha} completed.\n")
