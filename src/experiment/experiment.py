"""
Experiment Orchestration for Multi-Agent Pricing Games

This module coordinates repeated pricing games between LLM agents,
managing rate limiting, data collection, and experiment lifecycle.
"""

import asyncio
import datetime
import logging
import time
from typing import Any, Dict, List

import numpy as np
import polars as pl

from src.agents.base_agent import Agent
from src.experiment.storage import StorageManager
from src.plotting.plotting import plot_experiment_svg, plot_real_data_svg
from src.prompts.prompt_manager import PromptManager
from src.utils.logger import setup_logger


class RateLimiter:
    """Thread-safe rate limiter that ensures requests are spaced appropriately."""

    def __init__(self, rate_limit_seconds: float = 1.0):
        self.rate_limit_seconds = rate_limit_seconds
        self.last_request_time = 0
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Acquire the rate limiter, waiting if necessary to respect the rate limit."""
        async with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time

            if time_since_last < self.rate_limit_seconds:
                wait_time = self.rate_limit_seconds - time_since_last
                await asyncio.sleep(wait_time)

            self.last_request_time = time.time()


class RateLimitedAgentWrapper:
    """Wrapper for agents that enforces rate limiting on their act method."""

    def __init__(self, agent, rate_limiter: RateLimiter, max_retries: int = 20):
        self.agent = agent
        self.rate_limiter = rate_limiter
        self.max_retries = max_retries
        self.logger = getattr(agent, "logger", logging.getLogger())

    async def act_with_rate_limit(self, prompt: str) -> Dict[str, Any]:
        """Execute agent.act() with rate limiting and retry logic."""
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                # Acquire rate limiter before making request
                await self.rate_limiter.acquire()

                # Make the actual request
                result = await self.agent.act(prompt)

                if attempt > 0:
                    self.logger.info(
                        f"Agent {self.agent.name} succeeded on attempt {attempt + 1}"
                    )

                return result

            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    wait_time = min(2**attempt, 30)  # Exponential backoff, max 30s
                    self.logger.warning(
                        f"Agent {self.agent.name} failed on attempt {attempt + 1}/{self.max_retries + 1}: {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error(
                        f"Agent {self.agent.name} failed after {self.max_retries + 1} attempts: {e}"
                    )

        # If we get here, all retries failed
        raise last_exception


async def execute_agents_rate_limited(
    agents: List,
    prompts: Dict[str, str],
    rate_limit_seconds: float = 1.1,
    max_retries: int = 20,
    logger=None,
) -> List[Dict[str, Any]]:
    """
    Execute all agents with rate limiting and async response collection.

    Args:
        agents: List of agent objects
        prompts: Dict mapping agent names to their prompts
        rate_limit_seconds: Minimum seconds between API requests
        max_retries: Maximum number of retries per agent
        logger: Logger instance for debugging

    Returns:
        List of results from all agents
    """
    if logger:
        logger.info(f"Starting rate-limited execution for {len(agents)} agents")

    # Create shared rate limiter
    rate_limiter = RateLimiter(rate_limit_seconds)

    # Wrap agents with rate limiting
    wrapped_agents = [
        RateLimitedAgentWrapper(agent, rate_limiter, max_retries) for agent in agents
    ]

    # Create tasks for all agents
    tasks = [
        asyncio.create_task(
            wrapped_agent.act_with_rate_limit(prompts[wrapped_agent.agent.name]),
            name=f"agent_{wrapped_agent.agent.name}",
        )
        for wrapped_agent in wrapped_agents
    ]

    if logger:
        logger.info(f"Created {len(tasks)} tasks, collecting results...")

    # Wait for all agents to complete (with retries)
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle any remaining exceptions
    final_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            # Log the final failure and create a fallback result
            agent_name = agents[i].name
            if logger:
                logger.error(f"Agent {agent_name} ultimately failed: {result}")
            raise result  # Or handle this differently based on your needs
        else:
            final_results.append(result)

    if logger:
        logger.info(f"Successfully collected {len(final_results)} results")

    return final_results


class Experiment:
    def __init__(
        self,
        name: str,
        agents: List[Agent],
        num_rounds: int,
        environment,
        cost_series: np.ndarray = None,
        experiment_dir: str = None,
        initial_real_data: dict[str, list[dict]] = None,
        experiment_plot: bool = True,
        include_date_in_prompt: bool = False,  # NEW PARAMETER
        start_date: datetime.date = datetime.date(2023, 1, 1),  # NEW PARAMETER
        rate_limit_seconds: float = 1.1,  # NEW: API rate limit
        max_retries: int = 20,  # NEW: Max retries per agent
    ):
        self.name = name
        self.agents = agents
        self.num_rounds = num_rounds
        self.cost_series = cost_series
        self.environment = environment
        self.experiment_plot = experiment_plot
        self.initial_real_data = initial_real_data or {}
        self.include_date_in_prompt = include_date_in_prompt
        self.start_date = start_date
        self.rate_limit_seconds = rate_limit_seconds
        self.max_retries = max_retries

        for idx, agent in enumerate(self.agents):
            agent.env_index = idx
            agent.environment = self.environment
        assert all(agent.env_index is not None for agent in self.agents), (
            "Some agents are missing env_index"
        )

        sorted_agents = sorted(self.agents, key=lambda ag: ag.env_index)
        env_params = {
            "a_0": 0.0,
            "a": np.array([agent.a for agent in sorted_agents]),
            "alpha": np.array([agent.alpha for agent in sorted_agents]),
            "c": np.array([agent.c for agent in sorted_agents]),
            "mu": 0.25,  # 0.046,
            "beta": 100,
            "sigma": 0.0,
            "group_idxs": [1 for _ in sorted_agents],
            "market_share": np.array(
                [
                    getattr(agent, "market_share", agent.env_params.get("market_share"))
                    for agent in sorted_agents
                ]
            ),
        }

        # ✅ Overwrite the attributes in the existing environment object
        self.environment.a_0 = env_params["a_0"]
        self.environment.a = env_params["a"]
        self.environment.alpha = env_params["alpha"]
        self.environment.c = env_params["c"]
        self.environment.mu = env_params["mu"]
        self.environment.beta = env_params["beta"]
        self.environment.sigma = env_params["sigma"]
        self.environment.group_idxs = env_params["group_idxs"]
        self.environment.nbr_agents = len(sorted_agents)
        self.environment.market_share = env_params["market_share"]
        self.environment._compute_benchmarks()  # Compute benchmarks for the environment

        self.logger = setup_logger()
        self.storage = StorageManager(
            n_agents=len(agents), logger=self.logger, experiment_dir=experiment_dir
        )
        self.prompt_manager = PromptManager(logger=self.logger)
        self.history = {agent.name: {} for agent in agents}

        if self.cost_series is not None:
            self.logger.info(
                f"agents {(len(agents), num_rounds)} cost series shape: {cost_series.shape}"
            )
            assert cost_series.shape == (len(agents), num_rounds), (
                "Cost series shape mismatch"
            )
            environment.register_time_series(c_series=cost_series)

    async def run(self):
        self._setup_experiment()
        if self.initial_real_data:
            self._inject_initial_real_data()
        for round_num in range(1, self.num_rounds + 1):
            self.logger.info(f"--- Round {round_num} ---")
            await self._run_round(round_num)
        self._finalize_experiment()

    def _setup_experiment(self):
        self.storage.create_experiment_dir(self.name)
        self.logger = setup_logger(log_file=self.storage.get_log_file_path())

        for agent in self.agents:
            agent.logger = self.logger

        metadata = {
            "name": self.name,
            "num_agents": len(self.agents),
            "agents_types": {agent.name: agent.type for agent in self.agents},
            "agents_prefixes": {agent.name: agent.prefix for agent in self.agents},
            "agents_prompts": {
                agent.name: agent.prompt_template for agent in self.agents
            },
            "agents_memory_length": {
                agent.name: agent.memory_length for agent in self.agents
            },
            "agents_models": {agent.name: agent.model_name for agent in self.agents},
            "agent_environment_mapping": {
                agent.name: {
                    "env_index": agent.env_index,
                    "a": agent.a,
                    "alpha": agent.alpha,
                    "c": agent.c,
                }
                for agent in self.agents
            },
            "num_rounds": self.num_rounds,
            "rate_limit_seconds": self.rate_limit_seconds,
            "max_retries": self.max_retries,
            "start_time": datetime.datetime.now().isoformat(),
        }

        if self.environment:
            metadata["environment"] = {
                "name": self.environment.name,
                "description": self.environment.description,
                "environment_params": self.environment.get_environment_params(),
            }

        self.storage.save_metadata(metadata)

    async def _run_round(self, round_num: int):
        """Optimized round execution with rate limiting."""
        self.environment.set_round(round_num)
        prompts = {
            agent.name: self.prompt_manager.generate_prompt(
                agent, self.history[agent.name], round_num
            )
            for agent in self.agents
        }

        # Use rate-limited execution
        self.logger.info(
            f"Executing {len(self.agents)} agents with {self.rate_limit_seconds}s rate limit"
        )
        start_time = time.time()

        results = await execute_agents_rate_limited(
            agents=self.agents,
            prompts=prompts,
            rate_limit_seconds=self.rate_limit_seconds,
            max_retries=self.max_retries,
            logger=self.logger,
        )

        execution_time = time.time() - start_time
        self.logger.info(f"Agent execution completed in {execution_time:.2f} seconds")

        prices = self._store_agent_outputs(results, round_num)
        agent_order = [(agent.name, agent.env_index) for agent in self.agents]
        quantities, profits = self.environment.compute_quantities_and_profits(
            agent_order, prices
        )
        self._store_environment_outputs(round_num, prices, quantities, profits)
        df_history = self._create_environment_dataframe()
        self.storage.save_environment_parquet(df_history)
        metadata = self.storage.load_metadata()
        svg_path = self.storage.experiment_path / "results_plot.svg"
        if self.experiment_plot:
            plot_experiment_svg(
                df=df_history,
                metadata=metadata,
                save_path=svg_path,
                show_quantities=True,
                show_profits=True,
                plot_references=True,
            )
        else:
            plot_real_data_svg(
                df=df_history,
                metadata=metadata,
                save_path=svg_path,
            )

        self.storage.save_round_data(self.history)

    def _store_agent_outputs(self, results, round_num) -> dict:
        prices = {}
        for result in results:
            name = result["agent_name"]
            output = result["content"]
            self.history[name][round_num] = output
            prices[name] = output["chosen_price"]
        return prices

    def _store_environment_outputs(self, round_num, prices, quantities, profits):
        for idx, agent in enumerate(self.agents):
            name = agent.name
            competitors_prices = {
                a: round(p, 2) for a, p in prices.items() if a != name
            }

            market_data = ""
            if self.include_date_in_prompt:
                current_date = self.start_date + datetime.timedelta(days=round_num - 1)
                day_of_week = current_date.strftime("%A")
                market_data += (
                    f"- Date: {current_date.strftime('%Y-%m-%d')} ({day_of_week})\n"
                )

            market_data += f"- My price: {round(prices[name], 2)}\n"
            if competitors_prices:
                market_data += f"- Competitor's prices: {competitors_prices}\n"
            market_data += f"- My quantity sold: {round(quantities[name], 2)}\n"
            market_data += f"- My profit earned: {round(profits[name], 2)}\n"

            marginal_cost = agent.get_marginal_cost(round_num)
            if self.cost_series is not None:
                market_data += f"- Marginal cost: {round(marginal_cost, 2)}\n"

            self.history[name][round_num]["marginal_cost"] = round(marginal_cost, 2)
            self.history[name][round_num]["quantity"] = round(quantities[name], 2)
            self.history[name][round_num]["profit"] = round(profits[name], 2)
            self.history[name][round_num]["market_data"] = market_data

    def _finalize_experiment(self):
        metadata = self.storage.load_metadata()
        metadata["end_time"] = datetime.datetime.now().isoformat()
        self.storage.save_metadata(metadata)
        self.logger.info("✅ Experiment completed.\n")
        for handler in self.logger.handlers[:]:
            handler.flush()
            if isinstance(handler, logging.FileHandler):
                handler.close()
                self.logger.removeHandler(handler)

    def _create_environment_dataframe(self) -> pl.DataFrame:
        records = []

        for agent in self.agents:
            agent_name = agent.name
            for round_num, data in self.history[agent_name].items():
                # Use marginal cost from history if it's already injected
                marginal_cost = data.get(
                    "marginal_cost", agent.get_marginal_cost(round_num)
                )

                record = {
                    "round": round_num,
                    "agent": agent_name,
                    "agent_type": agent.type,
                    "price": data.get("chosen_price"),
                    "marginal_cost": marginal_cost,
                    "quantity": data.get("quantity"),
                    "profit": data.get("profit"),
                    "initial_history": data.get("is_initial_history", False),
                }
                records.append(record)
        return pl.DataFrame(records)

    def _inject_initial_real_data(self):
        self.logger.info("📜 Injecting real data as initial memory.")
        rounds_to_use = sorted(
            set(
                r["round"]
                for agent_data in self.initial_real_data.values()
                for r in agent_data
            )
        )

        for round_num in rounds_to_use:
            # Gather prices
            prices = {
                agent_name: next(
                    (
                        entry["chosen_price"]
                        for entry in agent_data
                        if entry["round"] == round_num
                    ),
                    None,
                )
                for agent_name, agent_data in self.initial_real_data.items()
            }

            if any(price is None for price in prices.values()):
                self.logger.error(f"Skipping round {round_num}: incomplete data")
                raise ValueError(f"Skipping round {round_num}: incomplete data")

            # Create c_override vector in the correct env_index order
            c_override = np.array(
                [
                    next(
                        (
                            entry.get(
                                "marginal_cost", agent.get_marginal_cost(round_num)
                            )
                            for entry in self.initial_real_data[agent.name]
                            if entry["round"] == round_num
                        ),
                        agent.get_marginal_cost(round_num),
                    )
                    for agent in sorted(self.agents, key=lambda ag: ag.env_index)
                ]
            )

            agent_order = [(agent.name, agent.env_index) for agent in self.agents]
            quantities, profits = self.environment.compute_quantities_and_profits(
                agent_order, prices, c_override=c_override
            )

            for agent in self.agents:
                name = agent.name
                entry = next(
                    (
                        e
                        for e in self.initial_real_data.get(name, [])
                        if e["round"] == round_num
                    ),
                    None,
                )
                if entry is None:
                    continue

                price = entry["chosen_price"]
                quantity = quantities[name]
                profit = profits[name]
                marginal_cost = entry.get(
                    "marginal_cost", agent.get_marginal_cost(round_num)
                )

                # Inject date if enabled
                market_data = ""

                if self.include_date_in_prompt:
                    current_date = self.start_date + datetime.timedelta(
                        days=round_num - 1
                    )
                    day_of_week = current_date.strftime("%A")
                    market_data += (
                        f"- Date: {current_date.strftime('%Y-%m-%d')} ({day_of_week})\n"
                    )

                market_data += (
                    f"- My price: {round(price, 2)}\n"
                    f"- Competitor's prices: { {k: round(v, 2) for k, v in prices.items() if k != name} }\n"
                    f"- My quantity sold: {round(quantity, 2)}\n"
                    f"- My profit earned: {round(profit, 2)}\n"
                    f"- Marginal cost: {round(marginal_cost, 2)}\n"
                )

                self.history[name][round_num] = {
                    "chosen_price": price,
                    "quantity": quantity,
                    "profit": profit,
                    "marginal_cost": marginal_cost,
                    "market_data": market_data,
                    "is_initial_history": True,
                }

        self.logger.info("✅ Finished injecting real data.\n")
