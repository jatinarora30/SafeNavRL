from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any,Tuple

class BaseAlgo(ABC):

    def __init__(self):
        print("Initiate Base Algo Class")
        
    @abstractmethod
    def saveModel(self,path): 
        raise NotImplementedError("Subclasses must implement saveModel().")

    @abstractmethod
    def loadModel(self,path):
        raise NotImplementedError("Subclasses must implement loadModel().")

    @abstractmethod
    def train(self) -> Tuple[Any, Any]:
        raise NotImplementedError("Subclasses must implement train().")
    
    def test(self) -> Tuple[Any, Any]:
        raise NotImplementedError("Subclasses must implement test().")


