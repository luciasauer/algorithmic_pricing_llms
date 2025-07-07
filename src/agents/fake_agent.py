"""
Fake Agent for Testing and Simulation

This module provides a deterministic agent that follows pre-defined price
sequences, useful for testing and baseline comparisons.
"""

import numpy as np
from src.agents.base_agent import Agent


class FakeAgent(Agent):
    """
    Deterministic agent that follows pre-defined price sequences.

    Used for testing market mechanisms and providing baseline comparisons
    with deterministic pricing behavior.

    Args:
        name: Unique identifier for the agent
        time_series_data: Array of predetermined prices to follow
        nbr_rounds: Number of rounds the agent will participate in
        **kwargs: Additional arguments passed to parent Agent class
    """

    def __init__(
        self, name: str, time_series_data: np.ndarray, nbr_rounds: int, **kwargs
    ):
        # Pass `env_params` to the parent (Agent) constructor
        super().__init__(name=name, **kwargs)

        assert len(time_series_data) >= nbr_rounds, (
            "Time series can't be smaller than the number of rounds"
        )
        self.model_name = "fake_agent"
        self.memory_length = -1
        self.time_series_data = time_series_data
        self.current_index = 0

    async def act(self, prompt: str) -> dict:
        """
        Return the next predetermined price in the sequence.

        Args:
            prompt: Ignored, as fake agent doesn't use prompts

        Returns:
            Dict containing agent name and next price in sequence
        """
        chosen_price = self.time_series_data[self.current_index]
        self.current_index += 1
        return {
            "agent_name": self.name,
            "content": {"chosen_price": float(chosen_price)},
        }

    @property
    def requires_prompt(self) -> bool:
        """Fake agents don't require prompts."""
        return False

    @property
    def type(self) -> str:
        """Return agent type identifier."""
        return "fake_agent"
