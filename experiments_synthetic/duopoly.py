# experiments_synthetic/duopoly.py
"""
Duopoly algorithmic pricing experiment implementation.

This script runs a 2-agent algorithmic pricing experiment testing Folk Theorem
predictions about coordination in repeated pricing games. Two LLM agents compete
over multiple rounds using the Calvano et al. (2020) demand specification.

The experiment tests whether algorithmic coordination emerges in duopoly settings
and serves as a baseline for comparison with higher-order oligopoly experiments.
This is a direct extension of Fish et al. (2025) methodology to test coordination
sustainability.

Research Question:
    Do LLM agents exhibit sustained supracompetitive pricing in duopoly markets?
    How does this compare to theoretical Nash equilibrium and monopoly benchmarks?

Experimental Design:
    - 2 competing LLM agents (Firm A, Firm B)
    - 300 rounds per experiment for convergence analysis
    - Calvano demand environment with nested logit specification
    - 100-period rolling memory for strategic learning
    - Multiple alpha parameter values to test robustness

Expected Outcomes:
    - Sustained pricing above Nash equilibrium
    - Evidence of tacit coordination without explicit communication
    - Baseline for Folk Theorem testing across different market structures

Usage:
    python experiments_synthetic/duopoly.py

Requirements:
    - MISTRAL_API_KEY environment variable set
    - MODEL_NAME environment variable (default: mistral-large-2411)
    - Sufficient API credits for ~2100 API calls per full experiment run

Output:
    Results saved to data/results/ in Parquet format for statistical analysis.
    Includes round-by-round prices, profits, and agent reasoning text.
"""

import os
import sys
import asyncio
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.LLM_agent import LLMAgent
from src.experiment.experiment import Experiment
from src.prompts.prompts import GENERAL_PROMPT, P1, P2
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


async def main(prompt_prefix, alpha=1, experiment_name="duopoly_setting"):
    """
    Run a single duopoly pricing experiment.

    Creates two LLM agents and runs a 300-round pricing game using the Calvano
    demand environment. Agents make simultaneous pricing decisions with access
    to rolling window of market history.

    Args:
        prompt_prefix: Prompt specification to use (P1 or P2) - affects coordination propensity
        alpha: Demand parameter controlling product quality/markup (default: 1)
        experiment_name: Identifier for this experimental run (for data storage)

    Returns:
        None (results are saved to disk automatically)

    Note:
        This function represents one experimental run. The main execution loop
        runs multiple instances with different parameters for statistical analysis.
        Each run requires approximately 600 API calls (300 rounds × 2 agents).
    """
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
    ]

    env = CalvanoDemandEnvironment(
        name="Calvano Market",
        description="Duopoly environment with Calvano 2020 demand",
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
            for n, prompt in enumerate([P1, P2], start=1):
                print(f"Running experiment with alpha={alpha}")
                asyncio.run(
                    main(
                        prompt_prefix=prompt,
                        alpha=alpha,
                        experiment_name=f"duopoly_setting_P{n}",
                    )
                )
                print(f"Experiment with alpha={alpha} completed.\n")
