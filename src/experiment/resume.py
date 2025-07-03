# src/experiment/resume.py
# This script finishes unresumed experiments, just pass a folder name in the CLI, e.g.: python src/experiment/resume.py experiments_synthetic\experiments_runs\8_agents\1751510415_oligopoly_setting_8_firms_P2
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.agents.LLM_agent import LLMAgent
from src.environment.calvano import CalvanoDemandEnvironment
from src.experiment.experiment import Experiment
from src.prompts.prompts import GENERAL_PROMPT
from src.prompts.prompts_models import create_pricing_response_model


def get_last_completed_round(results: Dict, metadata: Dict) -> int:
    """Find the last round completed by ALL agents."""
    if not results:
        return 0

    agent_names = list(metadata.get("agents_types", {}).keys())
    if not agent_names:
        return 0

    # Find minimum round across all agents (last round where ALL agents have data)
    agent_rounds = []
    for agent_name in agent_names:
        agent_data = results.get(agent_name, {})
        if agent_data:
            max_round = max([int(r) for r in agent_data.keys()], default=0)
            agent_rounds.append(max_round)
        else:
            agent_rounds.append(0)

    return min(agent_rounds) if agent_rounds else 0


def restore_agent_memory(agent, agent_history: Dict):
    """Restore agent memory from history."""
    agent.memory = []

    for round_num in sorted(agent_history.keys(), key=int):
        round_data = agent_history[str(round_num)]
        memory_entry = {
            "round": int(round_num),
            "price": round_data.get("chosen_price", 0),
            "profit": round_data.get("profit", 0),
            "reasoning": round_data.get("pricing_reasoning", ""),
            "market_data": round_data.get("market_data", ""),
        }
        agent.memory.append(memory_entry)

    # Trim to memory length
    if len(agent.memory) > agent.memory_length:
        agent.memory = agent.memory[-agent.memory_length :]


async def resume_experiment(experiment_dir: str):
    """
    Resume an experiment from the given directory.

    Args:
        experiment_dir: Path to the experiment directory
    """
    experiment_path = Path(experiment_dir)

    # Load metadata and results
    with open(experiment_path / "metadata.json", "r") as f:
        metadata = json.load(f)

    with open(experiment_path / "results.json", "r") as f:
        results = json.load(f)

    # Check if already complete
    if metadata.get("end_time"):
        print("Experiment already completed!")
        return

    # Find where to resume
    last_round = get_last_completed_round(results, metadata)
    start_round = last_round + 1
    total_rounds = metadata.get("num_rounds", 300)

    if start_round > total_rounds:
        print("Experiment already completed!")
        return

    print(f"Resuming from round {start_round} (last completed: {last_round})")

    # Extract experiment parameters
    agent_env_mapping = metadata.get("agent_environment_mapping", {})
    agent_prefixes = metadata.get("agents_prefixes", {})
    agents_models = metadata.get("agents_models", {})

    # Get parameters from first agent
    first_agent_data = next(iter(agent_env_mapping.values()))
    alpha = first_agent_data.get("alpha", 1.0)
    prompt_prefix = next(iter(agent_prefixes.values()))

    # Load environment variables
    load_dotenv()
    API_KEY = os.getenv("MISTRAL_API_KEY")
    MODEL_NAME = next(iter(agents_models.values()))

    # Recreate agents
    PricingAgentResponse = create_pricing_response_model(
        include_wtp=True, wtp_value=4.51 * alpha
    )

    agents = []
    for agent_name in sorted(agent_env_mapping.keys()):
        agent_data = agent_env_mapping[agent_name]
        agent = LLMAgent(
            agent_name,
            prefix=prompt_prefix,
            api_key=API_KEY,
            model_name=MODEL_NAME,
            response_model=PricingAgentResponse,
            memory_length=metadata.get("agents_memory_length", {}).get(agent_name, 100),
            prompt_template=GENERAL_PROMPT,
            env_params={
                "a": agent_data.get("a", 2.0),
                "alpha": agent_data.get("alpha", alpha),
                "c": agent_data.get("c", 1.0),
            },
        )

        # Restore memory if agent has history
        if agent_name in results:
            restore_agent_memory(agent, results[agent_name])
            print(f"Restored {len(agent.memory)} memory entries for {agent_name}")

        agents.append(agent)

    # Create environment
    env = CalvanoDemandEnvironment(
        name="Calvano Market",
        description="Oligopoly environment with Calvano 2020 demand",
    )

    # Create experiment without automatic storage setup
    experiment = Experiment(
        name=metadata.get("name"),
        agents=agents,
        num_rounds=total_rounds,
        environment=env,
    )

    # Manually set up storage and logging for existing experiment
    from src.experiment.storage import StorageManager
    from src.utils.logger import setup_logger

    # Create storage manager and point it to existing directory
    experiment.storage = StorageManager(n_agents=len(agents))
    experiment.storage.experiment_path = experiment_path

    # Setup logger
    experiment.logger = setup_logger(log_file=experiment_path / "log.txt")
    for agent in experiment.agents:
        agent.logger = experiment.logger

    # Setup prompt manager
    from src.prompts.prompt_manager import PromptManager

    experiment.prompt_manager = PromptManager(logger=experiment.logger)

    # Restore history
    experiment.history = results

    # Convert string keys to integers in history (JSON saves them as strings)
    for agent_name in experiment.history:
        agent_history = experiment.history[agent_name]
        # Convert string round numbers to integers
        converted_history = {}
        for round_str, round_data in agent_history.items():
            converted_history[int(round_str)] = round_data
        experiment.history[agent_name] = converted_history

    # Run remaining rounds
    print(f"Running rounds {start_round} to {total_rounds}")
    for round_num in range(start_round, total_rounds + 1):
        experiment.logger.info(f"--- Round {round_num} ---")
        await experiment._run_round(round_num)

    # Finalize
    experiment._finalize_experiment()
    print("✅ Experiment completed!")


# Simple script usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python resume.py <experiment_directory>")
        print(
            "Example: python resume.py experiments_runs/8_agents/1234567890_oligopoly_setting_8_firms"
        )
        sys.exit(1)

    experiment_dir = sys.argv[1]

    if not Path(experiment_dir).exists():
        print(f"Directory does not exist: {experiment_dir}")
        sys.exit(1)

    asyncio.run(resume_experiment(experiment_dir))


# ===============================================================================
# How to add resume to your existing oligopoly_8.py
# ===============================================================================

# Add this at the top of your oligopoly_8.py:
# from src.experiment.resume import resume_experiment

# Add this to your main function or argument parser:
"""
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, help="Resume experiment from directory")
    # ... your other arguments
    
    args = parser.parse_args()
    
    if args.resume:
        asyncio.run(resume_experiment(args.resume))
    else:
        # Your normal experiment code
        asyncio.run(main())
"""
