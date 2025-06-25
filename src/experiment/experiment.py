import datetime
import asyncio
from typing import List
from src.utils.logger import setup_logger
from src.agents.base_agent import Agent
from src.experiment.storage import StorageManager
from src.prompts.prompt_manager import PromptManager

class Experiment:
    def __init__(self, name: str, agents: List[Agent], num_rounds: int):
        self.name = name
        self.agents = agents
        self.num_rounds = num_rounds
        self.logger = setup_logger()
        self.storage = StorageManager(n_agents=len(self.agents), logger=self.logger)
        self.prompt_manager = PromptManager(logger=self.logger)
        self.history = {agent.name: {} for agent in agents}

    async def run(self):
        self.storage.create_experiment_dir(self.name)

        #Set up the logger
        log_file = self.storage.get_log_file_path()
        self.logger = setup_logger(log_file=log_file)
        self.logger.info(f"🔬 Starting experiment: {self.name}")

        for agent in self.agents:
            agent.logger = self.logger

        metadata = {
            "name": self.name,
            "num_agents": len(self.agents),
            "num_rounds": self.num_rounds,
            "start_time": datetime.datetime.now().isoformat(),
            #NOTE!remember to add parameters here about the demand function!
        }
        self.storage.save_metadata(metadata)


        for round_num in range(1, self.num_rounds + 1):
            self.logger.info(f"--- Round {round_num} ---")

            tasks = []

            # Generate and pass prompts to agents
            for agent in self.agents:
                prompt = self.prompt_manager.generate_prompt(agent, self.history[agent.name])
                tasks.append(agent.act(prompt)) 

            results = await asyncio.gather(*tasks)
            
            for result in results:
                self.history[result['agent_name']][round_num] = result['content']

            self.storage.save_round_data(self.history)

        metadata["end_time"] = datetime.datetime.now().isoformat()
        self.storage.save_metadata(metadata)
        self.logger.info("✅ Experiment completed.\n")
