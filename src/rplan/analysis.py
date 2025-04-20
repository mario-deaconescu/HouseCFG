from abc import ABC, abstractmethod
from typing import TypeVar, Iterable, Optional

from attr import dataclass
from traitlets import Callable

from src.rplan.types import Plan

T = TypeVar('T')
U = TypeVar('U')

class RPlanAnalysisProcessor(ABC):

    @abstractmethod
    def process(self, plan: Plan) -> T:
        pass

    @abstractmethod
    def aggregate(self, plans: Iterable[T]) -> U:
        pass


class RPlanAnalysisVisualizer(RPlanAnalysisProcessor):

    def process(self, plan: Plan) -> T:
        pass

    def aggregate(self, plans: Iterable[T]) -> U:
        pass

    @abstractmethod
    def visualize(self, data: U):
        pass
