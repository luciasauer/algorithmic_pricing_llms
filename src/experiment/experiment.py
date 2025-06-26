import datetime
import asyncio
import polars as pl
from typing import List
from src.utils.logger import setup_logger
from src.agents.base_agent import Agent
from src.experiment.storage import StorageManager
from src.prompts.prompt_manager import PromptManager
from src.plotting.plotting import plot_experiment_svg


class Experiment:
    def __init__(self, name: str, agents: List[Agent], num_rounds: int, environment=None):
        self.name = name
        self.agents = agents
        self.num_rounds = num_rounds
        self.environment = environment

        self.logger = setup_logger()
        self.storage = StorageManager(n_agents=len(agents), logger=self.logger)
        self.prompt_manager = PromptManager(logger=self.logger)
        self.history = {agent.name: {} for agent in agents}

    async def run(self):
        self._setup_experiment()
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
            "agents_prompts": {agent.name: agent.prompt_template for agent in self.agents},
            "num_rounds": self.num_rounds,
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
        prompts = {
            agent.name: self.prompt_manager.generate_prompt(agent, self.history[agent.name])
            for agent in self.agents
        }

        tasks = [agent.act(prompts[agent.name]) for agent in self.agents]
        results = await asyncio.gather(*tasks)

        prices = self._store_agent_outputs(results, round_num)
        quantities, profits = self.environment.compute_quantities_and_profits(prices)
        self._store_environment_outputs(round_num, prices, quantities, profits)
        df_history = self._create_environment_dataframe()
        self.storage.save_environment_parquet(df_history)
        metadata = self.storage.load_metadata()
        svg_path = self.storage.experiment_path / "results_plot.svg"
        plot_experiment_svg(
                df=df_history,
                metadata=metadata,
                save_path=svg_path,
                show_quantities=True,
                show_profits=True,
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
        for agent in self.agents:
            name = agent.name
            competitors_prices = {a: p for a, p in prices.items() if a != name}

            market_data = f'- My price: {prices[name]}\n'
            if competitors_prices:
                market_data += f"- Competitor's prices: {competitors_prices}\n"
            market_data += f'- My quantity sold: {quantities[name]}\n'
            market_data += f'- My profit earned: {profits[name]}\n'

            self.history[name][round_num]['quantity'] = quantities[name]
            self.history[name][round_num]['profit'] = profits[name]
            self.history[name][round_num]['market_data'] = market_data

    def _finalize_experiment(self):
        metadata = self.storage.load_metadata()
        metadata["end_time"] = datetime.datetime.now().isoformat()
        self.storage.save_metadata(metadata)
        self.logger.info("✅ Experiment completed.\n")
    
    def _create_environment_dataframe(self) -> pl.DataFrame:
        records = []

        for agent in self.agents:
            agent_name = agent.name
            for round_num, data in self.history[agent_name].items():
                records.append({
                    "round": round_num,
                    "agent": agent_name,
                    "agent_type": agent.type,
                    "price": data.get("chosen_price"),
                    "quantity": data.get("quantity"),
                    "profit": data.get("profit"),
                })

        return pl.DataFrame(records)
