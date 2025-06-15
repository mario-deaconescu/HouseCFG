from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional

OPENAPI_T = TypeVar("OPENAPI_T")
PYTHON_T = TypeVar("PYTHON_T")

class Serializer(ABC, Generic[OPENAPI_T, PYTHON_T]):

    def __init__(self, serialized: OPENAPI_T):
        self.serialized = serialized

    @abstractmethod
    def validate(self) -> Optional[str]:
        """
        Validate the serialized data.
        """
        pass

    @abstractmethod
    def _to_object(self, serialized: OPENAPI_T) -> PYTHON_T:
        """
        Convert the serialized data to a Python object.
        """
        pass

    def to_object(self) -> PYTHON_T:
        exception = self.validate()
        if exception:
            raise ValueError(exception)

        return self._to_object(self.serialized)

