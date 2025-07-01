# experiments_synthetic/duopoly.py
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
from src.prompts.prompts import GENERAL_PROMPT, P2
from src.prompts.prompts_models import create_pricing_response_model

current_file_path = Path(__file__).resolve()

load_dotenv()
API_KEY = os.getenv("MISTRAL_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

MEMORY_LENGTH = 100
N_ROUNDS = 300
N_RUNS = 7
ALPHAS_TO_TRY = [1, 3.2, 10]


async def main(prompt_prefix, alpha=1, experiment_name="oligopoly_setting_8_firms"):
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
        LLMAgent(
            "Firm F",
            prefix=prompt_prefix,
            api_key=API_KEY,
            model_name=MODEL_NAME,
            response_model=PricingAgentResponse,
            memory_length=MEMORY_LENGTH,
            prompt_template=GENERAL_PROMPT,
            env_params={"a": 2.0, "alpha": alpha, "c": 1.0},
        ),
        LLMAgent(
            "Firm G",
            prefix=prompt_prefix,
            api_key=API_KEY,
            model_name=MODEL_NAME,
            response_model=PricingAgentResponse,
            memory_length=MEMORY_LENGTH,
            prompt_template=GENERAL_PROMPT,
            env_params={"a": 2.0, "alpha": alpha, "c": 1.0},
        ),
        LLMAgent(
            "Firm H",
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
            for n, prompt in enumerate([P2], start=1):
                print(f"Running experiment with alpha={alpha}")
                asyncio.run(
                    main(
                        prompt_prefix=prompt,
                        alpha=alpha,
                        experiment_name=f"oligopoly_setting_8_firms_P{n}",
                    )
                )
                print(f"Experiment with alpha={alpha} completed.\n")
