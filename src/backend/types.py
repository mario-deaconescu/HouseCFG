from typing import TypeVar, Generic, TypedDict

import numpy as np
import numpy.typing as npt

from src.rplan.types import Plan

T = TypeVar("T", float, int)

NumpyArray1D = list[T]
NumpyArray2D = list[list[T]]
NumpyArray3D = list[list[list[T]]]


class RoomModel(TypedDict):
    room_type: int
    corners: NumpyArray2D[float]


class PlanModel(TypedDict):
    rooms: list[RoomModel]


class NumpyArray1DSerializer(Generic[T]):

    @staticmethod
    def deserialize(serialized: NumpyArray1D[T]) -> np.ndarray:
        arr = np.array(serialized)
        if arr.ndim != 1:
            raise ValueError("Input must be a 1D array.")
        return arr

    @staticmethod
    def serialize(array: np.ndarray) -> NumpyArray1D[T]:
        return array.tolist()  # type: ignore


class NumpyArray2DSerializer(Generic[T]):

    @staticmethod
    def deserialize(serialized: NumpyArray2D[T]) -> np.ndarray:
        arr = np.array(serialized)
        if arr.ndim != 2:
            raise ValueError("Input must be a 2D array.")
        return arr

    @staticmethod
    def serialize(array: np.ndarray) -> NumpyArray2D[T]:
        return array.tolist()  # type: ignore


class NumpyArray3DSerializer(Generic[T]):
    @staticmethod
    def deserialize(serialized: NumpyArray3D[T]) -> np.ndarray:
        arr = np.array(serialized)
        if arr.ndim != 3:
            raise ValueError("Input must be a 3D array.")
        return arr

    @staticmethod
    def serialize(array: np.ndarray) -> NumpyArray3D[T]:
        return array.tolist()  # type: ignore


class PlanSerializer:

    @staticmethod
    def deserialize(serialized: Plan) -> Plan:
        raise NotImplementedError("Deserialization not implemented.")

    @staticmethod
    def serialize(plan: Plan) -> PlanModel:
        return {
            "rooms": [{
                "room_type": room.room_type.value,
                "corners": NumpyArray2DSerializer.serialize(room.corners)
            } for room in plan.rooms]
        }
