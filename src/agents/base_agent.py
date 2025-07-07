"""
Base Agent Interface for Market Simulation

This module defines the abstract base class for all market agents,
providing common functionality for pricing decisions and market interaction.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class Agent(ABC):
    """
    Abstract base class for market agents in oligopoly experiments.

    Defines the interface for agents that make pricing decisions based on
    market context and strategic parameters.

    Args:
        name: Unique identifier for the agent
        prefix: Strategic context prefix for decision-making
        prompt_template: Template for formatting decision prompts
        env_index: Agent's index in the market environment
        env_params: Market environment parameters (a, alpha, c)
        logger: Logger instance for experiment tracking
    """

    def __init__(
        self,
        name: str,
        prefix: str = "",
        prompt_template: str = "",
        env_index: int = None,
        env_params: dict = None,
        logger=None,
    ):
        self.name = name
        self.prefix = prefix
        self.prompt_template = prompt_template
        self.logger = logger
        # Env values
        self.env_index = env_index
        # Initialize parameters from env_params
        env_params = env_params or {}
        self.env_params = env_params
        self.a = env_params.get("a")
        self.alpha = env_params.get("alpha")
        self.c = env_params.get("c")

        # Assert that these parameters are not None
        assert self.a is not None, f"Missing 'a' parameter for agent {self.name}"
        assert self.alpha is not None, (
            f"Missing 'alpha' parameter for agent {self.name}"
        )
        assert self.c is not None, f"Missing 'c' parameter for agent {self.name}"

    @property
    def requires_prompt(self) -> bool:
        """Whether agent requires prompt for decision-making."""
        return True  # default for LLM-based agents

    @property
    def type(self) -> str:
        """Return agent type identifier."""
        return None

    @abstractmethod
    async def act(self, prompt: str) -> Dict[str, Any]:
        """
        Execute pricing decision based on market context.

        Args:
            prompt: Market context and decision prompt

        Returns:
            Dict containing agent response and metadata
        """
        pass

    def get_marginal_cost(self, round_num: int) -> float:
        """
        Return marginal cost for a given round.
        Falls back to static cost if no dynamic series is present.
        """
        if (
            hasattr(self, "environment")
            and hasattr(self.environment, "c_series")
            and self.environment.c_series is not None
        ):
            marginal_cost = self.environment.c_series[self.env_index, round_num - 1]
        else:
            marginal_cost = self.c
        return marginal_cost
