import json
import os
from collections.abc import Callable
from copy import copy
from functools import partial
from multiprocessing.shared_memory import SharedMemory
from typing import Optional, TypeVar, Iterable, Sized

from multiprocessing import Value, Manager

import numpy as np
import torch
from attr import dataclass
from torch.utils.data import Dataset
from tqdm import tqdm

from src.constants import MAX_POSSIBLE_CORNERS, FILTER_CACHE_PATH, RPLAN_CACHE_PATH
from src.rplan.analysis import RPlanAnalysisProcessor
from src.rplan.types import RawPlan, Plan, TorchTransformerPlan, PlanGraph, RoomType, MaskPlan, ImagePlan

from tqdm.contrib.concurrent import process_map

T = TypeVar('T')


def max_corner_filter(max_room_corners: int, max_total_corners: int, plan: Plan) -> bool:
    return max(
        [room.corners.shape[0] for room in plan.rooms]) <= max_room_corners and plan.total_points <= max_total_corners


def max_room_corners_filter(max_room_corners: int, plan: Plan) -> bool:
    return all([room.corners.shape[0] <= max_room_corners for room in plan.rooms]) and all(
        [door is None or door.corners.shape[0] == 4 for _, _, door in plan.edges])


FilterArgs = tuple[Callable[[Plan], bool], str, str]


class RawRPlanDataset(Dataset):

    def __init__(self, data_path: str, filter_args: Optional[FilterArgs] = None, load_all: bool = False,
                 max_workers: int = 8):
        self.load_all = load_all
        if load_all:
            print("Processing raw RPlan dataset")
            self.data = process_map(RawRPlanDataset.read_raw_plan,
                                    [os.path.join(data_path, f) for f in os.listdir(data_path)],
                                    max_workers=max_workers, chunksize=1000)
        else:
            self.data = [os.path.join(data_path, f) for f in os.listdir(data_path)]

        self.index_mapper = None
        if filter_args is not None:
            filter_fn, cache_path, cache_name = filter_args
            if load_all:
                self.data = list(RPlanFilter(data_path, filter_fn, cache_path, cache_name).filter(self.data))
            else:
                self.index_mapper = RPlanFilter(data_path, filter_fn, cache_path, cache_name).index_mapper()

    @staticmethod
    def read_raw_plan(data_path: str):
        with open(data_path, 'r') as f:
            return RawPlan(json.load(f))

    def __len__(self):
        if self.index_mapper is not None:
            return len(self.index_mapper)
        return len(self.data)

    def __getitem__(self, idx) -> RawPlan:
        if self.load_all:
            return self.data[idx]

        if self.index_mapper is not None:
            idx = self.index_mapper[idx]
        return self.read_raw_plan(self.data[idx])


class RPlanDatasetMeta(type):
    def __del__(cls):
        if not isinstance(cls, RPlanDataset):
            return
        if cls._cache is not None:
            cls._cache.cleanup()
            cls._cache = None


class RPlanDataset(Dataset, metaclass=RPlanDatasetMeta):
    @dataclass
    class Cache:
        _plan_memory: SharedMemory
        _offset_memory: SharedMemory
        _length: int
        _filter_memory: Optional[SharedMemory] = None
        _filter_length: Optional[int] = None

        @staticmethod
        def from_data(data: list[Plan]):
            pickled_plans = [plan.pickle() for plan in data]
            plan_bytes = b''.join(pickled_plans)
            offsets = np.zeros(len(data), dtype=np.int64)
            offset_size = 0
            for i, plan in enumerate(pickled_plans):
                offset_size += len(plan)
                offsets[i] = offset_size

            plan_memory = SharedMemory(create=True, size=len(plan_bytes))
            offset_memory = SharedMemory(create=True, size=offsets.nbytes)
            plan_memory.buf[:len(plan_bytes)] = plan_bytes
            offset_memory.buf[:offsets.nbytes] = offsets.tobytes()
            return RPlanDataset.Cache(plan_memory, offset_memory, len(data))

        @staticmethod
        def from_shared_memory(plan_memory_name: str, offset_memory_name: str, length: int):
            plan_memory = SharedMemory(plan_memory_name)
            offset_memory = SharedMemory(offset_memory_name)
            return RPlanDataset.Cache(plan_memory, offset_memory, length)

        def set_filter(self, filter: np.ndarray):
            self._filter_memory = SharedMemory(create=True, size=filter.nbytes)
            self._filter_memory.buf[:filter.nbytes] = filter.tobytes()
            self._filter_length = len(filter)

        def __len__(self):
            if self._filter_memory is not None:
                return self._filter_length
            return self._length

        def __getitem__(self, item) -> Plan:
            if self._filter_memory is not None:
                item = np.frombuffer(self._filter_memory.buf, dtype=np.int64, count=len(self))[item]
            offset_end = np.frombuffer(self._offset_memory.buf, dtype=np.int64, count=self._length)[item]
            offset_start = 0 if item == 0 else \
                np.frombuffer(self._offset_memory.buf, dtype=np.int64, count=self._length)[item - 1]
            plan_bytes = bytes(self._plan_memory.buf[offset_start:offset_end])
            return Plan.unpickle(plan_bytes)

        def cleanup(self):
            self._plan_memory.close()
            self._offset_memory.close()
            if self._filter_memory is not None:
                self._filter_memory.close()

    _cache: Optional[Cache] = None

    def __init__(self, data_path: str, load_all: bool = False, canvas_size: tuple[int, int] = (256, 256),
                 filter_args: Optional[FilterArgs] = None, random_translate: bool = False,
                 random_rotate_90: bool = False, random_horizontal_flip: bool = False, centralize: bool = False,
                 random_scale: Optional[tuple[float, float]] = None,
                 random_vertical_flip: bool = False, expand: bool = True,
                 max_workers: int = 8):
        # Load raw data
        self.random_rotate_90 = random_rotate_90
        self.random_horizontal_flip = random_horizontal_flip
        self.random_vertical_flip = random_vertical_flip
        self.random_translate = random_translate
        self.random_scale = random_scale
        self.centralize = centralize
        self.canvas_size = canvas_size
        self.load_all = load_all
        self.expand = expand
        if load_all:
            if RPlanDataset._cache is None:
                RPlanDataset.cache(data_path, RPLAN_CACHE_PATH, max_workers=max_workers)
            if filter_args is not None:
                RPlanDataset._cache.set_filter(
                    RPlanFilter(data_path, filter_args[0], filter_args[1], filter_args[2]).index_map)
            self._cache = RPlanDataset._cache
        else:
            self.raw_data = RawRPlanDataset(data_path, load_all=load_all, filter_args=filter_args,
                                            max_workers=max_workers)

    @staticmethod
    def cache(data_path: str, path: str, max_workers: int = 8):
        if os.path.exists(path):
            print("Loading cached RPlan dataset")
            data = np.load(path, allow_pickle=True)
        else:
            data = process_map(Plan.from_raw, RawRPlanDataset(data_path, load_all=True, max_workers=max_workers),
                               max_workers=max_workers, chunksize=1000)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.save(RPLAN_CACHE_PATH, data, allow_pickle=True)

        RPlanDataset._cache = RPlanDataset.Cache.from_data(data)

        return RPlanDataset._cache

    def __len__(self):
        if not self.load_all:
            return len(self.raw_data)
        return len(self._cache)

    def __getitem__(self, idx) -> Plan:
        if self.load_all:
            plan = self._cache[idx]
        else:
            raw = self.raw_data[idx]
            plan = Plan.from_raw(raw)
        if self.centralize:
            plan.centralize(self.canvas_size)
        if self.random_scale:
            plan.scale(np.random.uniform(self.random_scale[0], self.random_scale[1]), self.canvas_size)
        if self.random_rotate_90:
            num_rotations = np.random.randint(0, 4)
            plan.rotate_90(num_rotations, self.canvas_size)
        if self.random_horizontal_flip or self.random_vertical_flip:
            hflip, vflip = np.random.choice([True, False], 2)
            hflip = hflip if self.random_horizontal_flip else False
            vflip = vflip if self.random_vertical_flip else False
            plan.flip(self.canvas_size, hflip, vflip)
        if self.random_translate:
            plan.random_translate(self.canvas_size)
        return plan

    def analyze(self, processor: RPlanAnalysisProcessor, max_workers: int = 8) -> T:
        return processor.aggregate([processor.process(plan) for plan in self])


class RPlanFilter:
    class IndexMapper:

        def __init__(self, index_map: np.ndarray):
            self.index_map = index_map

        def __getitem__(self, idx):
            return self.index_map[idx]

        def __len__(self):
            return len(self.index_map)

    def __init__(self, data_path: str, filter_generator: Callable[[Plan], bool], cache_path: str,
                 cache_name: str = 'filter_array.npy', max_workers: int = 8):
        self.data_path = data_path
        self.cache_path = cache_path
        cache_path_matrix = os.path.join(cache_path, cache_name)
        if os.path.exists(cache_path_matrix):
            self.filter_matrix = np.load(cache_path_matrix)
        else:
            self.filter_matrix = self._create_filter(data_path, filter_generator, max_workers)
            if not os.path.exists(os.path.dirname(cache_path_matrix)):
                os.makedirs(os.path.dirname(cache_path_matrix))
            np.save(cache_path_matrix, self.filter_matrix)

        self.index_map = (np.where(self.filter_matrix)[0]).astype(np.int64)

        print(f"Initalized filter with {len(self.index_map)} valid entries out of {len(self.filter_matrix)}")

    @staticmethod
    def _create_filter(data_path: str, filter_generator: Callable[[Plan], bool], max_workers: int = 8):
        dataset = RPlanDataset(data_path, load_all=True, max_workers=max_workers)
        print("Creating filter")
        filters = process_map(filter_generator, dataset,
                              max_workers=max_workers, chunksize=1000)
        return np.array(filters)

    def filter(self, data: Iterable[T]) -> Iterable[T]:
        return [entry for i, entry in enumerate(data) if self.filter_matrix[i]]

    def index_mapper(self) -> IndexMapper:
        return self.IndexMapper(self.index_map)


class RPlanTorchDataset(Dataset):

    def __init__(self, data_path: str, max_room_points: int, max_total_points: int, cache_path: str,
                 load_base_rplan: bool = False, random_flip: bool = False, random_translate: bool = False,
                 random_rotate: bool = False, shuffle_rooms: bool = False, front_door_at_end: bool = False,
                 no_doors: bool = False,
                 canvas_size: tuple[int, int] = (256, 256), max_workers: int = 8, device=torch.device('cpu')):
        self.device = device
        self.canvas_size = canvas_size
        self.max_room_points = max_room_points
        self.max_total_points = max_total_points
        self.no_doors = no_doors
        self.shuffle_rooms = shuffle_rooms
        self.front_door_at_end = front_door_at_end
        filter_args = (
            partial(max_corner_filter, max_room_points, max_total_points), cache_path, 'total_points_filter.npy')
        self.dataset = RPlanDataset(data_path, filter_args=filter_args, random_rotate_90=random_rotate,
                                    random_horizontal_flip=random_flip, random_vertical_flip=random_flip,
                                    random_translate=random_translate,
                                    load_all=load_base_rplan,
                                    max_workers=max_workers)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> TorchTransformerPlan:
        return TorchTransformerPlan.from_plan(self.dataset[idx], self.max_room_points, self.max_total_points,
                                              randomize_rooms=self.shuffle_rooms, no_doors=self.no_doors,
                                              front_door_at_end=self.front_door_at_end,
                                              device=self.device)

    @property
    def condition_channels(self):
        return TorchTransformerPlan.condition_channels(self.max_room_points)


class RPlanGraphRepresentationDataset(Dataset):

    def __init__(self, data_path: str, max_room_points: int, cache_path: str,
                 load_base_rplan: bool = False, random_flip: bool = False, random_translate: bool = False,
                 include_edge_data: bool = True, canvas_size: tuple[int, int] = (256, 256),
                 random_rotate: bool = False, shuffle_rooms: bool = False, max_workers: int = 8):
        self.max_room_points = max_room_points
        self.shuffle_rooms = shuffle_rooms
        self.include_edge_data = include_edge_data
        self.canvas_size = canvas_size
        filter_args = (
            partial(max_room_corners_filter, max_room_points), cache_path, 'total_room_points_filter.npy')
        self.dataset = RPlanDataset(data_path, filter_args=filter_args, random_rotate_90=random_rotate,
                                    random_horizontal_flip=random_flip, random_vertical_flip=random_flip,
                                    random_translate=random_translate,
                                    load_all=load_base_rplan,
                                    max_workers=max_workers)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> PlanGraph:
        plan = self.dataset[idx]
        plan = copy(plan)
        plan.normalize(self.canvas_size)
        if self.shuffle_rooms:
            new_indices = np.random.permutation(len(plan.rooms))
            new_rooms = [None] * len(plan.rooms)
            for i, room in enumerate(plan.rooms):
                new_rooms[new_indices[i]] = room
            plan.rooms = new_rooms
            plan.edges = [(new_indices[i], new_indices[j], door) for i, j, door in plan.edges]

        if self.include_edge_data:
            plan.edges = [(u, v, d) for u, v, d in plan.edges if d is not None]
        return PlanGraph.from_plan(plan, self.max_room_points, include_edge_data=self.include_edge_data)

class RPlanMasksDataset(Dataset):

    def __init__(self, data_path: str,
                 load_base_rplan: bool = False, random_flip: bool = False, random_translate: bool = False,
                 canvas_size: tuple[int, int] = (256, 256), mask_size: tuple[int, int] = (64, 64),
                 no_doors: bool = False, no_front_door: bool = False,
                 random_rotate: bool = False, shuffle_rooms: bool = False, max_workers: int = 8):
        self.shuffle_rooms = shuffle_rooms
        self.canvas_size = canvas_size
        self.mask_size = mask_size
        self.no_doors = no_doors
        self.no_front_door = no_front_door
        self.dataset = RPlanDataset(data_path, random_rotate_90=random_rotate,
                                    random_horizontal_flip=random_flip, random_vertical_flip=random_flip,
                                    random_translate=random_translate,
                                    load_all=load_base_rplan,
                                    max_workers=max_workers)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> MaskPlan:
        plan = self.dataset[idx]
        plan = copy(plan)
        if self.shuffle_rooms:
            new_indices = np.random.permutation(len(plan.rooms))
            new_rooms: list[Plan.Room] = [None] * len(plan.rooms)
            for i, room in enumerate(plan.rooms):
                new_rooms[new_indices[i]] = room
            plan.rooms = new_rooms
            plan.edges = [(new_indices[i], new_indices[j], door) for i, j, door in plan.edges]

        if self.no_doors:
            plan.edges = [(u, v, None) for u, v, _ in plan.edges]

        if self.no_front_door:
            try:
                front_door_index = next(i for i, room in enumerate(plan.rooms) if room.room_type == RoomType.FRONT_DOOR)
            except StopIteration:
                front_door_index = None
            if front_door_index is not None:
                plan.rooms.pop(front_door_index)
                def new_index(index):
                    return index - 1 if index > front_door_index else index
                plan.edges = [(new_index(u), new_index(v), door) for u, v, door in plan.edges if u != front_door_index and v != front_door_index]

        return MaskPlan.from_plan(plan, self.canvas_size, self.mask_size)

class RPlanImageDataset(Dataset):

    def __init__(self, data_path: str,
                 load_base_rplan: bool = False, random_flip: bool = False, random_scale: Optional[float] = None,
                 canvas_size: tuple[int, int] = (256, 256), mask_size: tuple[int, int] = (64, 64),
                 random_translate: bool = False,
                 no_doors: bool = False, no_front_door: bool = False,
                 random_rotate: bool = False, shuffle_rooms: bool = False, max_workers: int = 8):
        self.shuffle_rooms = shuffle_rooms
        self.canvas_size = canvas_size
        self.mask_size = mask_size
        self.no_doors = no_doors
        self.no_front_door = no_front_door
        if random_scale is not None:
            random_scale = (random_scale, 1.0)
        self.dataset = RPlanDataset(data_path, random_rotate_90=random_rotate,
                                    random_horizontal_flip=random_flip, random_vertical_flip=random_flip,
                                    random_translate=random_translate,
                                    centralize=True, random_scale=random_scale,
                                    load_all=load_base_rplan,
                                    max_workers=max_workers)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> ImagePlan:
        plan = self.dataset[idx]

        if self.no_doors:
            plan.edges = [(u, v, None) for u, v, _ in plan.edges]

        if self.no_front_door:
            try:
                front_door_index = next(i for i, room in enumerate(plan.rooms) if room.room_type == RoomType.FRONT_DOOR)
            except StopIteration:
                front_door_index = None
            if front_door_index is not None:
                plan.rooms.pop(front_door_index)
                def new_index(index):
                    return index - 1 if index > front_door_index else index
                plan.edges = [(new_index(u), new_index(v), door) for u, v, door in plan.edges if u != front_door_index and v != front_door_index]

        return ImagePlan.from_plan(plan, self.canvas_size[0], self.mask_size[0])