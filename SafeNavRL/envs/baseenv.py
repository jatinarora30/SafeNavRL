from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

class BaseEnv(ABC):
    """Abstract base class for SafeNavRL environments."""

    def __init__(self, renderMode: str = "human") -> None:
        self.env = None
        self.envName = ""
        self.renderMode = renderMode
        self.numEnvs = 0

    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, float, bool, bool, Dict]:
        """Perform one environment step."""
        raise NotImplementedError("Subclasses must implement step().")

    @abstractmethod
    def reset(self) -> Tuple[Any, Dict]:
        """Reset the environment."""
        raise NotImplementedError("Subclasses must implement reset().")

    @abstractmethod
    def close(self) -> None:
        """Close environment resources."""
        raise NotImplementedError("Subclasses must implement close().")
