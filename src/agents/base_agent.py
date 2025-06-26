from abc import ABC, abstractmethod
from typing import Dict, Any

class Agent(ABC):
    def __init__(self, name: str, prefix: str = "", prompt_template: str = "", logger=None):
        self.name = name
        self.prefix = prefix
        self.prompt_template = prompt_template
        self.logger = logger

    @property
    def requires_prompt(self) -> bool:
        return True  # default for LLM-based agents

    @property
    def type(self) -> str:
        return None
    
    @abstractmethod
    async def act(self, prompt: str) -> Dict[str, Any]:
        pass