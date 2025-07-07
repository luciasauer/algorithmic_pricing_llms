# experiments_synthetic/oligopoly_5.py
"""
5-agent oligopoly algorithmic pricing experiment implementation.

This script runs a 5-agent algorithmic pricing experiment representing the most
competitive market structure in the Folk Theorem test suite. Five LLM agents
compete over multiple rounds using the Calvano et al. (2020) demand specification.

This experiment provides the strongest test of Folk Theorem predictions:
coordination should be most difficult to sustain with 5 competitors, representing
the endpoint of the coordination breakdown analysis.

Research Question:
    Can algorithmic coordination survive in highly competitive 5-agent markets?
    What is the cumulative effect of moving from duopoly to 5-agent competition?

Experimental Design:
    - 5 competing LLM agents (Firm A, B, C, D, E)
    - 300 rounds per experiment for convergence analysis
    - Calvano demand environment with nested logit specification
    - 100-period rolling memory for strategic learning
    - Multiple alpha parameter values to test robustness

Folk Theorem Prediction:
    Strongest coordination breakdown expected:
    - Most complex coordination problem (5-way communication/coordination)
    - Highest individual deviation incentives
    - Lowest per-firm collusive profits (monopoly profit divided by 5)
    - Required discount factor approaches 1 for sustainability

Expected Outcomes:
    - Prices closest to Nash equilibrium among all market structures
    - Cumulative -10.6% price reduction from duopoly to 5-agent competition
    - Still some evidence of supracompetitive pricing (incomplete breakdown)
    - Strongest statistical evidence for Folk Theorem predictions

Usage:
    python experiments_synthetic/oligopoly_5.py

Requirements:
    - MISTRAL_API_KEY environment variable set
    - MODEL_NAME environment variable (default: mistral-large-2411)
    - Sufficient API credits for ~3000 API calls per full experiment run

Output:
    Results saved to data/results/ in Parquet format for statistical analysis.
    Provides endpoint for Folk Theorem coordination breakdown analysis.

Note:
    N_RUNS = 2 (reduced from 7) due to high computational cost of 5-agent experiments.
    This still provides sufficient data for statistical analysis when combined with
    other market structures.
"""

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
from src.prompts.prompts import GENERAL_PROMPT, P1, P2
from src.prompts.prompts_models import create_pricing_response_model

current_file_path = Path(__file__).resolve()

load_dotenv()
API_KEY = os.getenv("MISTRAL_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

# Experimental Configuration Parameters
MEMORY_LENGTH = 100  # Number of past rounds agents remember for decision-making
N_ROUNDS = 300  # Length of each pricing game (allows for convergence analysis)
N_RUNS = 2  # Reduced runs due to computational cost (5 agents × 300 rounds)
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
        LLMAgent(
            "Firm D",
            prefix=prompt_prefix,
            api_key=API_KEY,
            model_name=MODEL_NAME,
            response_model=PricingAgentResponse,
            memory_length=MEMORY_LENGTH,
            prompt_template=GENERAL_PROMPT,
            env_params={"a": 2.0, "alpha": alpha, "c": 1.0},
        ),
        LLMAgent(
            "Firm E",
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
            for n, prompt in enumerate([P1, P2], start=1):
                print(f"Running experiment with alpha={alpha}")
                asyncio.run(
                    main(
                        prompt_prefix=prompt,
                        alpha=alpha,
                        experiment_name=f"oligopoly_setting_5_firms_P{n}",
                    )
                )
                print(f"Experiment with alpha={alpha} completed.\n")
