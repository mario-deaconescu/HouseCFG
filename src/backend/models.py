from typing import Optional

from pydantic import BaseModel

from src.backend.types import NumpyArray1D, NumpyArray3D, NumpyArray2D


class BaseInputParameters(BaseModel):
    condition_scale: float = 1.0
    rescaled_phi: float = 0.0
    num_samples: int = 1
    num_steps: int = 100
    ddim: bool = True
    mask: NumpyArray3D[int]
    as_image: bool = True
    skeletonize: bool = False
    simplify: bool = False
    felzenszwalb: bool = False

class BubblesInputParameters(BaseInputParameters):
    bubbles: Optional[NumpyArray3D[float]] = None

class RoomTypeInputParameters(BaseInputParameters):
    room_types: Optional[NumpyArray1D[int]] = None

class CombinedInputParameters(BubblesInputParameters, RoomTypeInputParameters):
    pass
