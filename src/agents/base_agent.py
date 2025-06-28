# src/agents/base_agent.py
from abc import ABC, abstractmethod
from typing import Dict, Any


class Agent(ABC):
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
        return True  # default for LLM-based agents

    @property
    def type(self) -> str:
        return None

    @abstractmethod
    async def act(self, prompt: str) -> Dict[str, Any]:
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
