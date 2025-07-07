# src/agents/base_agent.py
"""
Abstract base class for market agents.

This module defines the interface that all agents must implement to participate
in the market simulation. Agents represent firms or players that make pricing
decisions in the repeated pricing game.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class Agent(ABC):
    """
    Abstract base class for all market agents.

    Defines the interface that all agents must implement to participate in the
    market simulation. Agents can be LLM-based, rule-based, or any other type
    that implements the required methods.

    Attributes:
        name: Unique identifier for the agent (e.g., "Firm A")
        prefix: Prompt prefix for LLM agents (e.g., P1, P2)
        prompt_template: Template string for generating prompts
        env_index: Index of this agent in the environment arrays
        env_params: Dictionary of environment parameters specific to this agent
        a: Demand intercept parameter (product quality)
        alpha: Quality/markup parameter for profit calculations
        c: Base marginal cost for this agent
        logger: Logger instance for debugging and monitoring
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
        """
        Initialize a new agent.

        Args:
            name: Unique identifier for the agent
            prefix: Prompt prefix for LLM agents (e.g., "P1", "P2")
            prompt_template: Template string for generating prompts
            env_index: Index of this agent in environment parameter arrays
            env_params: Dictionary containing economic parameters for this agent.
                       Must include 'a' (demand intercept), 'alpha' (quality parameter),
                       and 'c' (marginal cost)
            logger: Logger instance for debugging and monitoring

        Raises:
            AssertionError: If required parameters 'a', 'alpha', or 'c' are missing
                           from env_params
        """
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
        """
        Whether this agent requires a prompt to make decisions.

        Returns:
            True for LLM-based agents that need prompts, False for rule-based agents
        """
        return True  # default for LLM-based agents

    @property
    def type(self) -> str:
        """
        Get the type identifier for this agent.

        Returns:
            String identifier for the agent type (e.g., "LLM", "rule-based")
        """
        return None

    @abstractmethod
    async def act(self, prompt: str) -> Dict[str, Any]:
        """
        Make a pricing decision based on the given prompt.

        This is the core method that each agent must implement to participate
        in the market simulation. The agent should analyze the prompt (which
        contains market history and context) and return a pricing decision.

        Args:
            prompt: String containing market context, history, and instructions
                   for the pricing decision

        Returns:
            Dictionary containing the agent's response. Must include at minimum
            a 'price' field with the chosen price. May include additional fields
            like 'reasoning' for analysis purposes.

        Raises:
            NotImplementedError: If not implemented by subclass
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
