#src/agents/fake_agent.py
import numpy as np
from src.agents.base_agent import Agent

class FakeAgent(Agent):
    def __init__(self, name: str, time_series_data: np.ndarray, nbr_rounds:int, **kwargs):
        # Pass `env_params` to the parent (Agent) constructor
        super().__init__(name=name, **kwargs)
        
        assert len(time_series_data) >= nbr_rounds, "Time series can't be smaller than the number of rounds"
        self.time_series_data = time_series_data
        self.current_index = 0

    async def act(self, prompt: str) -> dict:
        chosen_price = self.time_series_data[self.current_index]
        self.current_index += 1
        return {'agent_name': self.name, 'content': {'chosen_price': float(chosen_price)}}
    
    @property
    def requires_prompt(self) -> bool:
        return False
    
    @property
    def type(self) -> str:
        return "fake_agent"