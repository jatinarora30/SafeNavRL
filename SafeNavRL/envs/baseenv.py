from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

class BaseEnv(ABC):

    def __init__(self, renderMode: str = "human") -> None:
        print("Initiate Base Env Class")
        
        self.env = None
        self.envName = ""
        self.renderMode = renderMode
        self.numEnvs = 0

    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, float, bool, bool, Dict]:
        raise NotImplementedError("Subclasses must implement step().")

    @abstractmethod
    def reset(self) -> Tuple[Any, Dict]:
        raise NotImplementedError("Subclasses must implement reset().")

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError("Subclasses must implement close().")
