from __future__ import annotations

import enum
import itertools
import math
import pickle
from copy import copy
from dataclasses import dataclass
from typing import Any, Optional, Callable

import cv2
import networkx as nx
import numpy as np
import torch
from scipy.ndimage import label
from scipy.stats import truncnorm
from shapely import set_precision
from shapely.affinity import rotate
from shapely.geometry import Polygon, CAP_STYLE, JOIN_STYLE
from skimage.morphology import skeletonize
from skimage.segmentation import felzenszwalb, find_boundaries

from src.utils.linked_list import LinkedList


class RoomType(enum.Enum):
    LIVING_ROOM = 1
    KITCHEN = 2
    BEDROOM = 3
    BATHROOM = 4
    BALCONY = 5
    ENTRANCE = 6
    DINING_ROOM = 7
    STUDY_ROOM = 8
    STORAGE = 10
    FRONT_DOOR = 15
    UNKNOWN = 16
    INTERIOR_DOOR = 17

    @staticmethod
    def frequencies():
        return {
            RoomType.BEDROOM: 177097,
            RoomType.BATHROOM: 92778,
            RoomType.BALCONY: 82840,
            RoomType.LIVING_ROOM: 77287,
            RoomType.FRONT_DOOR: 76743,
            RoomType.KITCHEN: 74393,
            RoomType.STUDY_ROOM: 14175,
            RoomType.STORAGE: 3158,
            RoomType.DINING_ROOM: 1212,
            RoomType.UNKNOWN: 981,
            RoomType.ENTRANCE: 267
        }

    def to_str(self) -> str:
        return self.name.lower().replace("_", " ")

    def __str__(self) -> str:
        return self.to_str()

    def __repr__(self) -> str:
        return self.to_str()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RoomType):
            return False
        return self.value == other.value

    def __hash__(self) -> int:
        return hash(self.value)

    def one_hot(self, size: Optional[int] = None, device: torch.device = torch.device('cpu')) -> torch.Tensor:
        index_map = {room_type: i for i, room_type in enumerate(RoomType)}
        vector = torch.nn.functional.one_hot(torch.tensor(index_map[self], device=device), num_classes=len(RoomType))
        if size is not None:
            assert size >= vector.shape[0]
            vector = torch.cat([vector, torch.zeros(size - vector.shape[0], device=device)], dim=0)
        return vector

    def index(self) -> int:
        index_map = {room_type: i for i, room_type in enumerate(RoomType)}
        return index_map[self]

    def index_restricted(self) -> int:
        enums = list(RoomType)
        enums.remove(RoomType.INTERIOR_DOOR)
        enums.remove(RoomType.FRONT_DOOR)
        index_map = {room_type: i for i, room_type in enumerate(enums)}
        return index_map[self]

    @staticmethod
    def from_index_restricted(index: int) -> RoomType:
        enums = list(RoomType)
        enums.remove(RoomType.INTERIOR_DOOR)
        enums.remove(RoomType.FRONT_DOOR)
        index_map = {i: room_type for i, room_type in enumerate(enums)}
        return index_map[index]

    @staticmethod
    def restricted_length() -> int:
        return len(RoomType) - 2  # Exclude INTERIOR_DOOR and FRONT_DOOR

    @staticmethod
    def from_one_hot(one_hot: torch.Tensor) -> RoomType:
        index = torch.argmax(one_hot)
        return list(RoomType)[index]


class RawPlan:
    room_type: list[int]
    boxes: list[tuple[int, int, int, int]]
    edges: list[tuple[int, int, int, int, int, int]]
    ed_rm: list[list[int]]
    filename: str

    def __init__(self, json_obj: Any, filename: Optional[str] = None):
        self.room_type = json_obj['room_type']
        self.boxes = [tuple(x) for x in json_obj['boxes']]
        self.edges = [tuple(x) for x in json_obj['edges']]
        self.ed_rm = json_obj['ed_rm']
        self.filename = filename


class Plan:
    @dataclass
    class Room:
        room_type: RoomType
        corners: np.ndarray

        def normalize(self, canvas_size: tuple[int, int]):
            self.corners = self.corners / np.array(canvas_size) * 2 - 1

        def unnorm(self, canvas_size: tuple[int, int]):
            self.corners = (self.corners + 1) / 2 * np.array(canvas_size)

        def area(self) -> float:
            return Polygon(self.corners).area

        def centroid(self) -> np.ndarray:
            return np.array(Polygon(self.corners).centroid.coords[0])

        def __copy__(self):
            return Plan.Room(room_type=self.room_type, corners=self.corners.copy())

        def rotate(self, angle: float, canvas_size: tuple[int, int]):
            center = np.array(canvas_size) / 2
            self.corners -= center
            rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            self.corners = np.dot(self.corners, rotation_matrix)
            self.corners += center

        def rotate_90(self, num_rotations: int, canvas_size: tuple[int, int]):
            num_rotations %= 4
            if num_rotations == 0:
                return
            self.rotate(num_rotations * np.pi / 2, canvas_size)

        def flip(self, canvas_size: tuple[int, int], horizontal: bool = False, vertical: bool = False):
            center = np.array(canvas_size) / 2
            self.corners -= center
            if horizontal:
                self.corners[:, 0] = -self.corners[:, 0]
            if vertical:
                self.corners[:, 1] = -self.corners[:, 1]
            if (horizontal or vertical) and not (horizontal and vertical):
                # Reverse the order of the corners
                self.corners = self.corners[::-1]
            self.corners += center

        def translate(self, delta: np.ndarray):
            assert delta.shape == (2,)
            self.corners += delta

        def expand(self, factor: float, round_corners: bool = True):
            try:
                polygon = Polygon(self.corners)
                expanded = polygon.buffer(factor, cap_style=CAP_STYLE.square, join_style=JOIN_STYLE.mitre)
                self.corners = np.array(expanded.exterior.coords[:-1])
            except:
                pass
            finally:
                if round_corners:
                    self.corners = self.corners.round()

        def align(self):
            # Make the edges of the room parallel to the x and y axis
            for i in range(len(self.corners)):
                x1, y1 = self.corners[i]
                x2, y2 = self.corners[(i + 1) % len(self.corners)]

                if abs(x1 - x2) > abs(y1 - y2):  # Horizontal edge
                    # Align y-coordinates to the midpoint of y1 and y2
                    mid = (y1 + y2) / 2
                    self.corners[i] = (x1, mid)
                    self.corners[(i + 1) % len(self.corners)] = (x2, mid)
                else:  # Vertical edge
                    # Align x-coordinates to the midpoint of x1 and x2
                    mid = (x1 + x2) / 2
                    self.corners[i] = (mid, y1)
                    self.corners[(i + 1) % len(self.corners)] = (mid, y2)

        def simplify(self, tolerance: float = 1e-2):
            self.corners = np.array(
                Polygon(self.corners).simplify(tolerance, preserve_topology=True).exterior.coords[:-1])

            def collinear(p1, p2, p3):
                angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) - np.arctan2(p3[1] - p2[1], p3[0] - p2[
                    0])  # Angle between the two lines in radians
                angle = np.abs((angle + np.pi) % np.pi - np.pi)  # Absolute value of the angle in the range [0, pi]
                angle_deg = np.degrees(angle)
                return angle_deg < 15

            new_corners = [self.corners[0]]
            for i in range(1, len(self.corners) - 1):
                if not collinear(self.corners[i - 1], self.corners[i], self.corners[i + 1]):
                    new_corners.append(self.corners[i])
            new_corners.append(self.corners[-1])

            self.corners = np.array(new_corners)

        def straighten(self):
            corner_list = []

        def to_mask(self, canvas_size: tuple[int, int], mask_size: tuple[int, int]) -> np.ndarray:
            mask = np.zeros(mask_size, dtype=np.uint8)
            points = self.corners * np.array(mask_size) / np.array(canvas_size)
            mask = cv2.fillPoly(mask, [points.astype(np.int32)], 1)
            # mask = cv2.resize(mask, mask_size, interpolation=cv2.INTER_AREA)
            return mask.astype(bool)

    rooms: list[Room]
    edges: list[tuple[int, int, Optional[Room]]]

    def pickle(self) -> bytes:
        return pickle.dumps(self)

    @staticmethod
    def unpickle(data: bytes) -> Plan:
        return pickle.loads(data)

    def __copy__(self):
        return Plan(rooms=[copy(room) for room in self.rooms],
                    edges=[(u, v, copy(edge_data)) for u, v, edge_data in self.edges])

    def __init__(self, rooms: list[Room], edges: Optional[list[tuple[int, int, Optional[Room]]]] = None):
        self.rooms = rooms
        if edges is None:
            edges = []
        self.edges = edges

    def normalize(self, canvas_size: tuple[int, int]):
        for room in self.rooms:
            room.normalize(canvas_size)
        for _, _, edge_data in self.edges:
            if edge_data is not None:
                edge_data.normalize(canvas_size)

    def unnorm(self, canvas_size: tuple[int, int]):
        for room in self.rooms:
            room.unnorm(canvas_size)
        for _, _, edge_data in self.edges:
            if edge_data is not None:
                edge_data.unnorm(canvas_size)

    def _room_fn(self, fn: Callable[[Plan.Room], None]):
        for room in self.rooms:
            fn(room)
        for _, _, edge_data in self.edges:
            if edge_data is not None:
                fn(edge_data)

    def rotate(self, angle: float, canvas_size: tuple[int, int]):
        self._room_fn(lambda room: room.rotate(angle, canvas_size))

    def rotate_90(self, num_rotations: int, canvas_size: tuple[int, int]):
        self._room_fn(lambda room: room.rotate_90(num_rotations, canvas_size))

    def flip(self, canvas_size: tuple[int, int], horizontal: bool = False, vertical: bool = False):
        self._room_fn(lambda room: room.flip(canvas_size, horizontal, vertical))

    def translate(self, delta: np.ndarray):
        self._room_fn(lambda room: room.translate(delta))

    def align(self):
        self._room_fn(lambda room: room.align())

    def random_translate(self, canvas_size: tuple[int, int]):
        min_x = min([min(room.corners[:, 0]) for room in self.rooms])
        max_x = max([max(room.corners[:, 0]) for room in self.rooms])
        min_y = min([min(room.corners[:, 1]) for room in self.rooms])
        max_y = max([max(room.corners[:, 1]) for room in self.rooms])
        delta = np.random.uniform(-min_x, canvas_size[0] - max_x), np.random.uniform(-min_y, canvas_size[1] - max_y)
        delta = np.array(delta)
        self.translate(delta)

    @property
    def total_points(self) -> int:
        return sum([room.corners.shape[0] for room in self.rooms]) + sum(
            [edge_data.corners.shape[0] if edge_data is not None else 0 for _, _, edge_data in self.edges])

    @staticmethod
    def from_raw(raw_plan: RawPlan, expand: Optional[float] = 3) -> Plan:
        corners = [[] for _ in range(len(raw_plan.room_type))]
        graph_matrix = np.zeros((len(raw_plan.room_type), len(raw_plan.room_type)), dtype=bool)

        for i, edge in enumerate(raw_plan.edges):
            # Make graph
            if len(raw_plan.ed_rm[i]) == 2:
                room1, room2 = raw_plan.ed_rm[i]
                graph_matrix[room1, room2] = True
                graph_matrix[room2, room1] = True

            # Make corners
            for room_index in raw_plan.ed_rm[i]:
                corner1 = (edge[0], edge[1])
                corner2 = (edge[2], edge[3])

                try:
                    index1 = next(i for i, node in enumerate(corners[room_index]) if node.data == corner1)
                except StopIteration:
                    index1 = None

                try:
                    index2 = next(i for i, node in enumerate(corners[room_index]) if node.data == corner2)
                except StopIteration:
                    index2 = None

                if index1 is not None and index2 is not None:
                    node1 = corners[room_index][index1]
                    node2 = corners[room_index][index2]
                    if node1.next is None and node2.prev is None:
                        node1.next = node2
                        node2.prev = node1
                    elif node1.prev is None and node2.next is None:
                        node1.prev = node2
                        node2.next = node1
                    elif node1.next is None and node2.prev is not None:
                        node1.next = node2
                        node2.prev = node1
                    elif node1.prev is not None and node2.next is None:
                        node1.prev = node2
                        node2.next = node1
                    else:
                        continue
                elif index1 is None and index2 is None:
                    node1 = LinkedList.Node(corner1)
                    node2 = LinkedList.Node(corner2)
                    node1.next = node2
                    node2.prev = node1
                    corners[room_index].append(node1)
                    corners[room_index].append(node2)
                elif index1 is not None:
                    node1 = corners[room_index][index1]
                    node2 = LinkedList.Node(corner2)
                    if node1.next is None:
                        node1.next = node2
                        node2.prev = node1
                    elif node1.prev is None:
                        node1.prev = node2
                        node2.next = node1
                    else:
                        # raise ValueError('Invalid node')
                        continue
                    corners[room_index].append(node2)
                elif index2 is not None:
                    node1 = LinkedList.Node(corner1)
                    node2 = corners[room_index][index2]
                    if node2.next is None:
                        node2.next = node1
                        node1.prev = node2
                    elif node2.prev is None:
                        node2.prev = node1
                        node1.next = node2
                    else:
                        # raise ValueError('Invalid node')
                        continue
                    corners[room_index].append(node1)

        # Convert corners to list

        def find_chain(points: list[LinkedList.Node]) -> list[tuple[int, int]]:
            visited = [False] * len(points)
            for c_i, corner in enumerate(points):
                if visited[c_i] or corner.prev is None:
                    continue
                current = corner
                chain = []
                c_j = c_i
                while current is not None:
                    visited[c_j] = True
                    c_j += 1
                    chain.append(current.data)
                    if current.next is not None and current.next == corner:
                        return chain
                    current = current.next

            return []

        corner_list = list(map(find_chain, corners))

        rooms = [Plan.Room(room_type=RoomType(room_type), corners=np.array(list(chain))) for room_type, chain in
                 zip(raw_plan.room_type, corner_list)]

        # Remove doors
        graph = nx.Graph(graph_matrix)
        doors = set()
        edge_rooms = []
        for i, node in enumerate(graph.nodes):
            if raw_plan.room_type[i] != RoomType.INTERIOR_DOOR.value:
                continue
            doors.add(i)
            neighbors = list(graph.neighbors(node))
            for neighbor1_idx in range(len(neighbors)):
                neighbor1 = neighbors[neighbor1_idx]
                for neighbor2_idx in range(neighbor1_idx + 1, len(neighbors)):
                    neighbor2 = neighbors[neighbor2_idx]
                    if neighbor1 != neighbor2:
                        graph.add_edge(neighbor1, neighbor2, door=rooms[i])

        graph.remove_nodes_from(doors)
        # Remove nodes with no edges
        graph = nx.convert_node_labels_to_integers(graph)

        rooms = [room for room in rooms if room.room_type != RoomType.INTERIOR_DOOR and len(room.corners) > 0]

        edges = [(u, v, data.get('door', None)) for u, v, data in graph.edges(data=True)]

        edges = list(sorted(edges, key=lambda x: x[2] is None))

        plan = Plan(rooms=rooms, edges=edges)

        if expand is not None:
            plan.expand(expand)

        return plan

    def remove_overlaps(self):
        for i in range(len(self.rooms)):
            room1 = self.rooms[i]
            if room1.room_type == RoomType.FRONT_DOOR:
                continue
            for j in range(i + 1, len(self.rooms)):
                room2 = self.rooms[j]
                if room2.room_type == RoomType.FRONT_DOOR:
                    continue
                try:
                    ok = Polygon(room1.corners).intersects(Polygon(room2.corners))
                except Exception as e:
                    print(e)
                    continue
                if ok:
                    # Check if one room is inside the other
                    if Polygon(room1.corners).contains(Polygon(room2.corners)):
                        # Cut the room with the largest area
                        room1_polygon = Polygon(room1.corners).difference(Polygon(room2.corners))
                        # Check if the room was cut into multiple parts
                        if room1_polygon.geom_type == 'MultiPolygon':
                            # Keep the largest part
                            room1_polygon = max(room1_polygon.geoms, key=lambda x: x.area)
                        room1.corners = np.array(room1_polygon.exterior.coords[:-1])
                    elif Polygon(room2.corners).contains(Polygon(room1.corners)):
                        # Cut the room with the largest area
                        room2_polygon = Polygon(room2.corners).difference(Polygon(room1.corners))
                        # Check if the room was cut into multiple parts
                        if room2_polygon.geom_type == 'MultiPolygon':
                            # Keep the largest part
                            room2_polygon = max(room2_polygon.geoms, key=lambda x: x.area)
                        room2.corners = np.array(room2_polygon.exterior.coords[:-1])
                    else:
                        area1 = room1.area()
                        area2 = room2.area()

                        # Cut the room with the smallest area
                        if area1 < area2:
                            room1_polygon = Polygon(room1.corners).difference(Polygon(room2.corners))
                            # Check if the room was cut into multiple parts
                            if room1_polygon.geom_type == 'MultiPolygon':
                                # Keep the largest part
                                room1_polygon = max(room1_polygon.geoms, key=lambda x: x.area)
                            room1.corners = np.array(room1_polygon.exterior.coords[:-1])
                        else:
                            room2_polygon = Polygon(room2.corners).difference(Polygon(room1.corners))
                            # Check if the room was cut into multiple parts
                            if room2_polygon.geom_type == 'MultiPolygon':
                                # Keep the largest part
                                room2_polygon = max(room2_polygon.geoms, key=lambda x: x.area)
                            room2.corners = np.array(room2_polygon.exterior.coords[:-1])

    def expand_neighbours2(self, max_factor: float = 30, min_room_area: float = 70):

        neighbours = [[] for _ in range(len(self.rooms))]
        for u, v, _ in self.edges:
            neighbours[u].append(v)
            neighbours[v].append(u)

        def has_untouched_neighbour(room_idx: int) -> bool:
            for neighbour in neighbours[room_idx]:
                if self.rooms[neighbour].room_type == RoomType.FRONT_DOOR:
                    continue
                if not Polygon(self.rooms[room_idx].corners).intersects(Polygon(self.rooms[neighbour].corners)):
                    return True
            return False

        # Sort rooms by area (only the indices)
        room_indices = [idx for idx in sorted(range(len(self.rooms)), key=lambda x: self.rooms[x].area()) if
                        self.rooms[idx].room_type != RoomType.FRONT_DOOR and has_untouched_neighbour(idx)]

        tries = 0
        while room_indices and tries < max_factor:
            for room_idx in room_indices:
                room = self.rooms[room_idx]
                room.expand(1)

                # Remove overlap with other rooms
                for neighbour in range(len(self.rooms)):
                    if neighbour == room_idx:
                        continue
                    polygon = Polygon(self.rooms[room_idx].corners).difference(
                        Polygon(self.rooms[neighbour].corners))
                    if polygon.geom_type == 'MultiPolygon':
                        polygon = max(polygon.geoms, key=lambda x: x.area)
                    room.corners = np.array(polygon.exterior.coords[:-1])

            room_indices = [idx for idx in room_indices if has_untouched_neighbour(idx)]
            tries += 1

        for room in self.rooms:
            if room.room_type == RoomType.FRONT_DOOR:
                continue

            tries = 0
            while room.area() < min_room_area and tries < 10:
                room.expand(5)
                self.remove_overlaps()
                tries += 1

    def expand_neighbours(self, max_factor: float = 30, min_room_area: float = 70):

        neighbours = [[] for _ in range(len(self.rooms))]
        for u, v, _ in self.edges:
            neighbours[u].append(v)
            neighbours[v].append(u)

        def has_untouched_neighbour(room_idx: int) -> bool:
            for neighbour in neighbours[room_idx]:
                if self.rooms[neighbour].room_type == RoomType.FRONT_DOOR:
                    continue
                if not Polygon(self.rooms[room_idx].corners).intersects(Polygon(self.rooms[neighbour].corners)):
                    return True
            return False

        # Sort rooms by area (only the indices)
        room_indices = [idx for idx in sorted(range(len(self.rooms)), key=lambda x: self.rooms[x].area()) if
                        self.rooms[idx].room_type != RoomType.FRONT_DOOR and has_untouched_neighbour(idx)]

        for room_idx in room_indices:
            room = self.rooms[room_idx]
            centroid = room.centroid()
            for neighbour in neighbours[room_idx]:
                if self.rooms[neighbour].room_type == RoomType.FRONT_DOOR:
                    continue
                other_centroid = self.rooms[neighbour].centroid()
                # Find direction to the neighbour (north, south, east, west)
                dx, dy = other_centroid - centroid
                if abs(dx) > abs(dy):
                    # Move along x-axis
                    dx = np.sign(dx)
                    dy = 0
                    dx2 = 0
                    dy2 = np.sign(dy)
                else:
                    # Move along y-axis
                    dx = 0
                    dy = np.sign(dy)
                    dx2 = np.sign(dx)
                    dy2 = 0

                new_room = room.corners.copy()

                def good_intersection():
                    nonlocal new_room
                    if not Polygon(new_room).intersects(Polygon(self.rooms[neighbour].corners)):
                        return False

                    # Remove overlap with other rooms
                    for other_idx in range(len(self.rooms)):
                        if other_idx == room_idx or self.rooms[other_idx].room_type == RoomType.FRONT_DOOR:
                            continue
                        polygon = Polygon(new_room).difference(
                            Polygon(self.rooms[other_idx].corners))
                        if polygon.geom_type == 'MultiPolygon':
                            polygon = max(polygon.geoms, key=lambda x: x.area)
                        # polygon = set_precision(polygon, 1)
                        new_room = np.array(polygon.exterior.coords[:-1])

                    intersection = Polygon(new_room).intersection(Polygon(self.rooms[neighbour].corners))
                    if intersection.length < 5:
                        return False

                    return True

                tt = 0
                while not good_intersection() and tt < max_factor:
                    room_copy = new_room.copy()
                    room_copy[:, 0] += dx
                    room_copy[:, 1] += dy
                    room_copy = Polygon(room_copy)
                    # Union with the original room
                    new_room = np.array(room_copy.union(Polygon(new_room)).exterior.coords[:-1])
                    tt += 1

                # Remove overlap with other rooms
                for other_idx in range(len(self.rooms)):
                    if other_idx == room_idx or self.rooms[other_idx].room_type == RoomType.FRONT_DOOR:
                        continue
                    polygon = Polygon(new_room).difference(
                        Polygon(self.rooms[other_idx].corners))
                    if polygon.geom_type == 'MultiPolygon':
                        polygon = max(polygon.geoms, key=lambda x: x.area)
                    # polygon = set_precision(polygon, 1)
                    new_room = np.array(polygon.exterior.coords[:-1])

                # Try the other direction
                tt = 0
                while not good_intersection() and tt < max_factor:
                    room_copy = new_room.copy()
                    room_copy[:, 0] += dx2
                    room_copy[:, 1] += dy2
                    room_copy = Polygon(room_copy)
                    # Union with the original room
                    new_room = np.array(room_copy.union(Polygon(new_room)).exterior.coords[:-1])
                    tt += 1

                # Remove overlap with other rooms
                for other_idx in range(len(self.rooms)):
                    if other_idx == room_idx or self.rooms[other_idx].room_type == RoomType.FRONT_DOOR:
                        continue
                    polygon = Polygon(new_room).difference(
                        Polygon(self.rooms[other_idx].corners))
                    if polygon.geom_type == 'MultiPolygon':
                        polygon = max(polygon.geoms, key=lambda x: x.area)
                    # polygon = set_precision(polygon, 1)
                    new_room = np.array(polygon.exterior.coords[:-1])

                room.corners = new_room

        for room in self.rooms:
            if room.room_type == RoomType.FRONT_DOOR:
                continue

            tries = 0
            while room.area() < min_room_area and tries < 10:
                room.expand(5)
                self.remove_overlaps()
                tries += 1

            room_polygon = Polygon(room.corners)
            room_polygon = set_precision(room_polygon, 1)
            room.corners = np.array(room_polygon.exterior.coords[:-1])

    def add_doors(self, door_width: float, door_height: float):
        for edge_idx, (room1_idx, room2_idx, door) in enumerate(self.edges):
            try:
                if door is not None:
                    continue

                room1 = copy(self.rooms[room1_idx])
                room2 = copy(self.rooms[room2_idx])

                if room1.room_type == RoomType.FRONT_DOOR or room2.room_type == RoomType.FRONT_DOOR:
                    continue

                room1_polygon = Polygon(room1.corners)
                room2_polygon = Polygon(room2.corners)

                expand_factor = 1
                max_tries = 10

                while not room1_polygon.intersects(room2_polygon) and max_tries > 0:
                    room1.expand(expand_factor)
                    room2.expand(expand_factor)
                    room1_polygon = Polygon(room1.corners)
                    room2_polygon = Polygon(room2.corners)
                    expand_factor *= 2
                    max_tries -= 1

                if max_tries == 0:
                    raise ValueError('Could not find a room that touches another room')

                room1_area = room1.area()
                room2_area = room2.area()
                # Cut the room with the smallest area
                if room1_area < room2_area:
                    room1_polygon = room1_polygon.difference(room2_polygon)
                else:
                    room2_polygon = room2_polygon.difference(room1_polygon)

                # Find the intersections lines
                intersections = room1_polygon.intersection(room2_polygon)

                # intersections.simplify(5, preserve_topology=True)

                if not intersections.geom_type == 'MultiLineString' and not intersections.geom_type == 'LineString':
                    raise ValueError('Invalid intersection type')

                if intersections.geom_type == 'LineString':
                    intersections = [intersections]
                else:
                    intersections = intersections.geoms

                # Remove walls that are too short
                valid_intersections = [intersection for intersection in intersections if
                                       intersection.length >= door_width]
                if len(valid_intersections) == 0:
                    # raise ValueError('No valid intersection found')
                    intersection = max(intersections, key=lambda x: x.length)
                    current_door_width = intersection.length
                else:
                    current_door_width = door_width
                    # Choose a random intersection
                    intersection = intersections[np.random.randint(0, len(intersections))]

                margin = current_door_width / 2 / intersection.length

                # Choose a random point along the line (between one end and the other)
                low, high = margin, 1 - margin
                std = 1
                point = truncnorm(a=(low - 0.5) / std, b=(high - 0.5) / std, loc=0.5, scale=std).rvs()
                x = intersection.xy[0][0] + point * (intersection.xy[0][1] - intersection.xy[0][0])
                y = intersection.xy[1][0] + point * (intersection.xy[1][1] - intersection.xy[1][0])

                # Draw a wide rectangle with the center at the intersection point and the same orientation as the intersection line
                # The rectangle will be the door
                door_polygon = Polygon([(x - current_door_width / 2, y - door_height / 2),
                                        (x + current_door_width / 2, y - door_height / 2),
                                        (x + current_door_width / 2, y + door_height / 2),
                                        (x - current_door_width / 2, y + door_height / 2)])

                # Align the door with the line
                door_polygon = rotate(door_polygon, np.arctan2(intersection.xy[1][1] - intersection.xy[1][0],
                                                               intersection.xy[0][1] - intersection.xy[0][0]),
                                      origin=(x, y), use_radians=True)
                door = Plan.Room(room_type=RoomType.INTERIOR_DOOR, corners=np.array(door_polygon.exterior.coords[:-1]))

                self.edges[edge_idx] = (room1_idx, room2_idx, door)
            except Exception as e:
                print(e)

    def expand(self, factor: float, round_corners: bool = True):
        self._room_fn(lambda room: room.expand(factor, round_corners))

        # Remove overlap between rooms
        self.remove_overlaps()

    def visualize(self, view_graph: bool = True, view_door_index: bool = False, view_room_type: bool = False, ax=None):
        from matplotlib import pyplot as plt
        if ax is None:
            fig, ax = plt.subplots()
            # Set limits
            ax.set_xlim(0, 256)
            ax.set_ylim(0, 256)

        # Color for each room type
        spectrum = np.linspace(0, 1, max([room_type.value for room_type in RoomType]) + 1)
        colors = [plt.cm.hsv(spectrum[room.room_type.value]) for room in self.rooms]
        for room_idx, room in enumerate(self.rooms):
            for i in range(len(room.corners)):
                ax.plot([room.corners[i - 1][0], room.corners[i][0]], [room.corners[i - 1][1], room.corners[i][1]],
                        color=colors[room_idx], linewidth=2)

            if view_room_type:
                centroid = room.centroid()
                ax.text(centroid[0], centroid[1], str(room.room_type), fontsize=6, color='black',
                        horizontalalignment='center',
                        verticalalignment='center')

            if view_graph and room.room_type != RoomType.INTERIOR_DOOR:
                centroid = room.centroid()
                ax.plot(centroid[0], centroid[1], 'o', color=colors[room_idx], markersize=5)
                # Plot the room index
                if not view_room_type:
                    ax.text(centroid[0], centroid[1], str(room_idx), fontsize=12, color='black',
                            horizontalalignment='center',
                            verticalalignment='center')

        for door_idx, (room1_idx, room2_idx, door) in enumerate(self.edges):
            # Draw door
            if door is not None:
                for i in range(len(door.corners)):
                    ax.plot([door.corners[i - 1][0], door.corners[i][0]], [door.corners[i - 1][1], door.corners[i][1]],
                            color='black', linewidth=1)
                if view_door_index:
                    centroid = door.centroid()
                    ax.text(centroid[0], centroid[1], str(door_idx + len(self.rooms)), fontsize=12, color='black')
            if view_graph:
                # Draw line between rooms
                room1 = self.rooms[room1_idx]
                room2 = self.rooms[room2_idx]
                ax.plot([room1.centroid()[0], room2.centroid()[0]], [room1.centroid()[1], room2.centroid()[1]],
                        color='black', linewidth=1)

    def scale(self, factor: float, canvas_size: tuple[int, int]):
        def scale_room(room: Plan.Room):
            room.corners -= np.array(canvas_size) / 2
            room.corners *= factor
            room.corners += np.array(canvas_size) / 2

        self._room_fn(scale_room)

    def centralize(self, canvas_size: tuple[int, int]):
        # Centralize the plan and fill canvas
        room_points = [room.corners for room in self.rooms]
        door_points = [edge_data.corners for _, _, edge_data in self.edges if edge_data is not None]
        all_points = np.concatenate(room_points + door_points, axis=0)
        min_x = np.min(all_points[:, 0])
        max_x = np.max(all_points[:, 0])
        min_y = np.min(all_points[:, 1])
        max_y = np.max(all_points[:, 1])
        delta_x = (canvas_size[0] - (max_x - min_x)) / 2 - min_x
        delta_y = (canvas_size[1] - (max_y - min_y)) / 2 - min_y
        self.translate(np.array([delta_x, delta_y]))

        horizontal_scale = canvas_size[0] / (max_x - min_x)
        vertical_scale = canvas_size[1] / (max_y - min_y)

        scale = min(horizontal_scale, vertical_scale)

        self.scale(scale, canvas_size)


@dataclass
class TorchTransformerPlan:
    ROOM_ONE_HOT_SIZE = 32

    @dataclass
    class Conditions:
        door_mask: torch.Tensor
        self_mask: torch.Tensor
        gen_mask: torch.Tensor
        room_types: torch.Tensor
        corner_indices: torch.Tensor
        room_indices: torch.Tensor
        src_key_padding_mask: torch.Tensor
        connections: torch.Tensor  # The next corner in the room
        room_graph: Optional[list[list[tuple[int, int, Optional[Plan.Room]]]]] = None

        def to(self, device: torch.device) -> TorchTransformerPlan.Conditions:
            return TorchTransformerPlan.Conditions(door_mask=self.door_mask.to(device),
                                                   self_mask=self.self_mask.to(device),
                                                   gen_mask=self.gen_mask.to(device),
                                                   room_types=self.room_types.to(device),
                                                   corner_indices=self.corner_indices.to(device),
                                                   room_indices=self.room_indices.to(device),
                                                   src_key_padding_mask=self.src_key_padding_mask.to(device),
                                                   connections=self.connections.to(device),
                                                   room_graph=self.room_graph)

        def to_dict(self) -> dict[str, torch.Tensor]:
            return {
                'door_mask': self.door_mask,
                'self_mask': self.self_mask,
                'gen_mask': self.gen_mask,
                'room_types': self.room_types,
                'corner_indices': self.corner_indices,
                'room_indices': self.room_indices,
                'src_key_padding_mask': self.src_key_padding_mask,
                'connections': self.connections,
                'room_graph': self.room_graph
            }

    coordinates: torch.Tensor
    conditions: Conditions

    def to(self, device: torch.device) -> TorchTransformerPlan:
        return TorchTransformerPlan(coordinates=self.coordinates.to(device), conditions=self.conditions.to(device))

    @staticmethod
    def condition_channels(max_room_points: int) -> int:
        return len(RoomType) + max_room_points + TorchTransformerPlan.ROOM_ONE_HOT_SIZE

    @staticmethod
    def collate(plans: list[TorchTransformerPlan], keep_graphs: bool = False) -> TorchTransformerPlan:
        coordinates = torch.stack([plan.coordinates for plan in plans], dim=0)
        door_mask = torch.stack([plan.conditions.door_mask for plan in plans], dim=0)
        self_mask = torch.stack([plan.conditions.self_mask for plan in plans], dim=0)
        gen_mask = torch.stack([plan.conditions.gen_mask for plan in plans], dim=0)
        room_types = torch.stack([plan.conditions.room_types for plan in plans], dim=0)
        corner_indices = torch.stack([plan.conditions.corner_indices for plan in plans], dim=0)
        room_indices = torch.stack([plan.conditions.room_indices for plan in plans], dim=0)
        src_key_padding_mask = torch.stack([plan.conditions.src_key_padding_mask for plan in plans], dim=0)
        connections = torch.stack([plan.conditions.connections for plan in plans], dim=0)
        if not keep_graphs or any(plan.conditions.room_graph is None for plan in plans):
            room_graph = None
        else:
            room_graph = itertools.chain(*[plan.conditions.room_graph for plan in plans])

        return TorchTransformerPlan(coordinates=coordinates,
                                    conditions=TorchTransformerPlan.Conditions(door_mask=door_mask, self_mask=self_mask,
                                                                               gen_mask=gen_mask, room_types=room_types,
                                                                               corner_indices=corner_indices,
                                                                               room_indices=room_indices,
                                                                               src_key_padding_mask=src_key_padding_mask,
                                                                               connections=connections,
                                                                               room_graph=room_graph))

    @staticmethod
    def uncollate(plans: TorchTransformerPlan) -> list[TorchTransformerPlan]:
        coordinates = plans.coordinates
        door_mask = plans.conditions.door_mask
        self_mask = plans.conditions.self_mask
        gen_mask = plans.conditions.gen_mask
        room_types = plans.conditions.room_types
        corner_indices = plans.conditions.corner_indices
        room_indices = plans.conditions.room_indices
        src_key_padding_mask = plans.conditions.src_key_padding_mask
        connections = plans.conditions.connections

        return [TorchTransformerPlan(coordinates=coordinates[i],
                                     conditions=TorchTransformerPlan.Conditions(door_mask=door_mask[i],
                                                                                self_mask=self_mask[i],
                                                                                gen_mask=gen_mask[i],
                                                                                room_types=room_types[i],
                                                                                corner_indices=corner_indices[i],
                                                                                room_indices=room_indices[i],
                                                                                src_key_padding_mask=
                                                                                src_key_padding_mask[i],
                                                                                connections=connections[i]))
                for i in range(len(coordinates))]

    @staticmethod
    def from_plan(source: Plan, max_room_points: int, max_total_points: int, randomize_rooms: bool = False,
                  front_door_at_end: bool = False, no_doors: bool = False,
                  device: torch.device = torch.device('cpu')) -> TorchTransformerPlan:
        plan = copy(source)
        plan.normalize((256, 256))
        if no_doors:
            assert front_door_at_end is False
            plan.edges = [(u, v, None) for u, v, _ in plan.edges]
            front_door_index = None
            for i, room in enumerate(plan.rooms):
                if room.room_type == RoomType.FRONT_DOOR:
                    front_door_index = i
                    break
            if front_door_index is not None:
                plan.edges = [(u, v, edge_data) for u, v, edge_data in plan.edges if
                              plan.rooms[u].room_type != RoomType.FRONT_DOOR and plan.rooms[
                                  v].room_type != RoomType.FRONT_DOOR]
                plan.edges = [(u - 1 if u > front_door_index else u, v - 1 if v > front_door_index else v, edge_data)
                              for u, v, edge_data in plan.edges]
                plan.rooms = plan.rooms[:front_door_index] + plan.rooms[front_door_index + 1:]

        total_points = plan.total_points
        # print("Checking plan")
        for i, (_, _, edge_data) in enumerate(plan.edges):
            if edge_data is not None:
                continue
            for j in range(i + 1, len(plan.edges)):
                if plan.edges[j][2] is not None:
                    raise ValueError('Invalid plan')
        door_edges = [(u, v, edge_data) for u, v, edge_data in plan.edges if edge_data is not None]
        all_rooms = plan.rooms + [edge_data for _, _, edge_data in door_edges]
        door_indices = len(plan.rooms) + np.arange(len(door_edges))
        if randomize_rooms:
            random_indices = np.random.permutation(len(all_rooms))
            random_rooms = [all_rooms[i] for i in random_indices]
            inverse_indices = np.argsort(random_indices)
            plan.edges = [(inverse_indices[u], inverse_indices[v], edge_data) for u, v, edge_data in plan.edges]
            door_edges = [(u, v, edge_data) for u, v, edge_data in plan.edges if edge_data is not None]
            door_indices = inverse_indices[door_indices]
            all_rooms = random_rooms
        elif front_door_at_end:
            front_door_index = None
            for i, room in enumerate(all_rooms):
                if room.room_type == RoomType.FRONT_DOOR:
                    front_door_index = i
                    break
            if front_door_index is not None:
                all_rooms = all_rooms[:front_door_index] + all_rooms[front_door_index + 1:] + [
                    all_rooms[front_door_index]]
                door_indices -= 1

                def convert_index(index: int) -> int:
                    if index < front_door_index:
                        return index
                    elif index == front_door_index:
                        return len(all_rooms) - 1
                    else:
                        return index - 1

                plan.edges = [(convert_index(u), convert_index(v), edge_data) for u, v, edge_data in plan.edges]
                door_edges = [(u, v, edge_data) for u, v, edge_data in plan.edges if edge_data is not None]

        door_mask = torch.ones(max_total_points, max_total_points, dtype=torch.bool, device=device)
        self_mask = torch.ones(max_total_points, max_total_points, dtype=torch.bool, device=device)
        gen_mask = torch.ones(max_total_points, max_total_points, dtype=torch.bool, device=device)

        # Gen mask
        gen_mask[:total_points, :total_points] = False

        room_indices = torch.zeros(max_total_points, dtype=torch.long, device=device)
        room_to_points = []

        passed_points = 0
        connections1 = None
        connections2 = None
        for i, room in enumerate(all_rooms):
            room_indices[passed_points:passed_points + room.corners.shape[0]] = i + 1
            room_to_points.append((passed_points, passed_points + room.corners.shape[0]))
            connections1_vec = np.arange(room.corners.shape[0]) + passed_points
            connections2_vec = np.roll(connections1_vec, -1)
            connections1 = np.concatenate(
                [connections1, connections1_vec]) if connections1 is not None else connections1_vec
            connections2 = np.concatenate(
                [connections2, connections2_vec]) if connections2 is not None else connections2_vec
            passed_points += room.corners.shape[0]

        # Self mask (points in the same room)
        # Door mask (points in adjacent rooms)
        room_adjacency = np.zeros((len(all_rooms), len(all_rooms)), dtype=bool)
        for door_idx, (u, v, door) in zip(door_indices, door_edges):
            assert door is not None
            room_adjacency[u, door_idx] = True
            room_adjacency[door_idx, u] = True
            room_adjacency[v, door_idx] = True
            room_adjacency[door_idx, v] = True

        for u, v, _ in [(u, v, door) for u, v, door in plan.edges]:
            room_adjacency[u, v] = True
            room_adjacency[v, u] = True

        for i in range(len(room_to_points)):
            start, end = room_to_points[i]
            self_mask[start:end, start:end] = False
            for j in range(i + 1, len(room_to_points)):
                start2, end2 = room_to_points[j]
                if room_adjacency[i, j]:
                    door_mask[start:end, start2:end2] = False
                    door_mask[start2:end2, start:end] = False

        corner_indices = torch.tensor([i for room in all_rooms for i in range(room.corners.shape[0])], dtype=torch.long,
                                      device=device)
        corner_indices = torch.cat(
            [corner_indices, torch.zeros(max_total_points - total_points, dtype=torch.long, device=device)])
        room_types = torch.stack(
            [room.room_type.one_hot(25, device) for room in all_rooms for _ in range(room.corners.shape[0])])
        room_types = torch.cat([room_types,
                                torch.zeros((max_total_points - total_points, 25), dtype=torch.float32,
                                            device=device)])
        src_key_padding_mask = torch.cat(
            [torch.zeros(room.corners.shape[0], dtype=torch.bool, device=device) for room in all_rooms] + [
                torch.ones(max_total_points - total_points, dtype=torch.bool, device=device)])

        corner_indices = torch.nn.functional.one_hot(corner_indices, num_classes=max_room_points)

        coordinates = torch.tensor(np.concatenate([room.corners for room in all_rooms], axis=0), dtype=torch.float32,
                                   device=device)
        coordinates = torch.cat(
            [coordinates, torch.zeros((max_total_points - total_points, 2), dtype=torch.float32, device=device)])
        connections = torch.tensor(np.stack([connections1, connections2], axis=1), dtype=torch.long, device=device)
        connections = torch.cat(
            [connections, torch.zeros((max_total_points - total_points, 2), dtype=torch.long, device=device)])
        room_indices = torch.nn.functional.one_hot(room_indices, num_classes=max_room_points)

        return TorchTransformerPlan(coordinates=coordinates,
                                    conditions=TorchTransformerPlan.Conditions(door_mask=door_mask, self_mask=self_mask,
                                                                               gen_mask=gen_mask, room_types=room_types,
                                                                               corner_indices=corner_indices,
                                                                               room_indices=room_indices,
                                                                               src_key_padding_mask=src_key_padding_mask,
                                                                               connections=connections,
                                                                               room_graph=[plan.edges]))

    def to_plan(self) -> Plan:
        num_points = (~self.conditions.src_key_padding_mask).sum().item()
        room_types = [RoomType.from_one_hot(one_hot) for one_hot in self.conditions.room_types[:num_points]]
        room_indices = self.conditions.room_indices[:num_points]
        room_indices = torch.argmax(room_indices, dim=1) - 1
        room_corners = []

        rooms = []
        result_room_types = []

        for i in range(num_points):
            room_idx = room_indices[i].item()
            if len(room_corners) <= room_idx:
                room_corners.extend([[] for _ in range(room_idx - len(room_corners) + 1)])
                result_room_types.extend([None for _ in range(room_idx - len(result_room_types) + 1)])
            room_corners[room_idx].append(self.coordinates[i].cpu().numpy())
            result_room_types[room_idx] = room_types[i]

        for room_type, corners in zip(result_room_types, room_corners):
            rooms.append(Plan.Room(room_type=room_type, corners=np.array(corners)))

        # Find edges based on door mask
        adjacency_matrix = np.zeros((len(rooms), len(rooms)))
        for i in range(num_points):
            room_1 = room_indices[i].item()
            for j in range(i + 1, num_points):
                room_2 = room_indices[j].item()
                if self.conditions.door_mask[i, j] == 0:
                    adjacency_matrix[room_1, room_2] = True
                    adjacency_matrix[room_2, room_1] = True

        graph = nx.Graph(adjacency_matrix)
        # Remove doors
        edges = []
        for node in graph.nodes:
            # if result_room_types[node] in [RoomType.FRONT_DOOR, RoomType.BALCONY]:
            #     for neighbor in graph.neighbors(node):
            #         edges.append((node, neighbor, None))
            #     continue
            if result_room_types[node] != RoomType.INTERIOR_DOOR:
                continue
            neighbors = list(graph.neighbors(node))
            for neighbor1_idx in range(len(neighbors)):
                neighbor1 = neighbors[neighbor1_idx]
                for neighbor2_idx in range(neighbor1_idx + 1, len(neighbors)):
                    neighbor2 = neighbors[neighbor2_idx]
                    if neighbor1 != neighbor2:
                        graph.add_edge(neighbor1, neighbor2, door=rooms[node])
                        # edges.append((neighbor1, neighbor2, rooms[node]))

        graph.remove_nodes_from([node for node in graph.nodes if result_room_types[node] == RoomType.INTERIOR_DOOR])

        pos = np.full(max(graph.nodes) + 1, -1)
        pos[list(graph.nodes)] = np.arange(len(graph.nodes))

        edges = [(pos[u].item(), pos[v].item(), edge_data.get('door', None)) for u, v, edge_data in
                 graph.edges(data=True)]

        plan = Plan(rooms=[room for room in rooms if room.room_type != RoomType.INTERIOR_DOOR], edges=edges)
        plan.unnorm((256, 256))
        return plan


@dataclass
class PlanGraph:
    nodes: np.ndarray
    src_key_padding_mask: np.ndarray
    edges: np.ndarray
    edge_data: Optional[np.ndarray]

    @staticmethod
    def from_plan(plan: Plan, max_points: int, include_edge_data: bool = False) -> PlanGraph:

        def expand_points(points: np.ndarray, factor: int) -> np.ndarray:
            # points = [[x1, y1], [x2, y2], ...]
            x = np.linspace(points[:, 0], np.roll(points[:, 0], -1), factor + 2)[:-1].T
            y = np.linspace(points[:, 1], np.roll(points[:, 1], -1), factor + 2)[:-1].T
            return np.stack([x.flatten(), y.flatten()], axis=1)

        def room_encoding(room: Plan.Room) -> tuple[np.ndarray, np.ndarray]:
            # points = np_dec2bin(room.corners.round().astype(int), 8).reshape(-1).astype(int)
            # points[points == 0] = -1
            assert room.corners.shape[0] <= max_points
            remaining_points = max_points - room.corners.shape[0]
            expand_factor = remaining_points // room.corners.shape[0]
            points = expand_points(room.corners, expand_factor)
            remaining_points = max_points - points.shape[0]
            # Duplicate last point to fill the remaining points
            points = np.concatenate([points, np.tile(points[-1], (remaining_points, 1))])
            points_flat = np.concatenate([points.flatten(), np.zeros((max_points - points.shape[0], 2)).flatten()])
            encoding = np.concatenate([room.room_type.one_hot().numpy(), points_flat])
            mask = np.concatenate(
                [np.zeros(points.shape[0] * 2, dtype=bool), np.ones((max_points - points.shape[0]) * 2, dtype=bool)])
            mask = np.concatenate([np.zeros(len(RoomType), dtype=bool), mask])
            return encoding, mask

        def edge_encoding(edge: tuple[int, int, Optional[Plan.Room]]) -> tuple[np.ndarray, Optional[np.ndarray]]:
            points = np.array([edge[0], edge[1]])
            if include_edge_data:
                if edge[2] is None:
                    raise ValueError('Edge data is None')
                # edge_bin = np_dec2bin(edge[2].corners.round().astype(int), 8).astype(int)
                # edge_bin[edge_bin == 0] = -1
                edge_bin = edge[2].corners
                return points, edge_bin

            return points, None

        nodes, src_key_padding_mask = zip(*[room_encoding(room) for room in plan.rooms])
        nodes = np.stack(nodes, axis=0)
        src_key_padding_mask = np.stack(src_key_padding_mask, axis=0)
        edges, edge_data = zip(*[edge_encoding(edge) for edge in plan.edges])
        edges = np.stack(edges, axis=0)
        if include_edge_data:
            edge_data = np.stack(edge_data, axis=0)
        else:
            edge_data = None

        return PlanGraph(nodes=nodes, src_key_padding_mask=src_key_padding_mask, edges=edges, edge_data=edge_data)

    @staticmethod
    def collate(graphs: list[PlanGraph]) -> tuple[PlanGraph, np.ndarray, np.ndarray]:
        nodes = np.concatenate([graph.nodes for graph in graphs])
        src_key_padding_mask = np.concatenate([graph.src_key_padding_mask for graph in graphs])
        edges = []
        attention_mask = np.ones((nodes.shape[0], nodes.shape[0]), dtype=bool)
        passed_nodes = 0
        nodes_to_graph = []
        for i, graph in enumerate(graphs):
            edges.append(graph.edges + passed_nodes)
            new_passed_nodes = passed_nodes + graph.nodes.shape[0]
            attention_mask[passed_nodes:new_passed_nodes, passed_nodes:new_passed_nodes] = 0
            passed_nodes = new_passed_nodes
            nodes_to_graph.append(np.full(graph.nodes.shape[0], i))

        edges = np.concatenate(edges, axis=0)

        if any(graph.edge_data is None for graph in graphs):
            edge_data = None
        else:
            edge_data = np.concatenate([graph.edge_data for graph in graphs], axis=0)

        return PlanGraph(nodes=nodes, src_key_padding_mask=src_key_padding_mask, edges=edges,
                         edge_data=edge_data), attention_mask, np.concatenate(nodes_to_graph)

    @staticmethod
    def node_channels(max_points: int) -> int:
        return len(RoomType) + max_points * 2

    def to_plan(self) -> Plan:
        assert len(self.nodes.shape) == 2
        assert len(self.src_key_padding_mask.shape) == 2
        assert len(self.edges.shape) == 2
        if self.edge_data is not None:
            assert len(self.edge_data.shape) == 3

        num_points = ((~self.src_key_padding_mask).sum(axis=1) - len(RoomType)) // 2
        room_types = [RoomType.from_one_hot(torch.tensor(node[:len(RoomType)])) for node in self.nodes]
        room_corners = [node[len(RoomType):].reshape(-1, 2)[:num_points] for num_points, node in
                        zip(num_points, self.nodes)]
        rooms = [Plan.Room(room_type=room_type, corners=corners) for room_type, corners in
                 zip(room_types, room_corners)]
        if self.edge_data is None:
            edges = [(u, v, None)
                     for u, v in self.edges]
        else:
            edges = [
                (u, v, Plan.Room(room_type=RoomType.INTERIOR_DOOR, corners=edge_data)) if edge_data is not None else (
                    u, v, None) for (u, v), edge_data in zip(self.edges, self.edge_data)]
        return Plan(rooms=rooms, edges=edges)


@dataclass
class MaskPlan:
    room_masks: np.ndarray
    room_types: list[RoomType]
    edges: np.ndarray

    @staticmethod
    def from_plan(plan: Plan, canvas_size: tuple[int, int], mask_size: tuple[int, int]) -> MaskPlan:
        plan = copy(plan)
        plan.centralize(canvas_size)
        plan.expand(-2, round_corners=False)
        room_masks = []
        room_types = []
        edges = []
        for room in plan.rooms:
            room_masks.append(room.to_mask(canvas_size, mask_size))
            room_types.append(room.room_type)
        for u, v, door in plan.edges:
            edges.append((u, v))
            if door is None:
                continue
            door_mask = door.to_mask(canvas_size, mask_size)
            room_masks.append(door_mask)
            room_types.append(RoomType.INTERIOR_DOOR)
            edges.append((len(room_masks) - 1, u))
            edges.append((len(room_masks) - 1, v))
        return MaskPlan(room_masks=np.stack(room_masks, axis=0),
                        room_types=room_types,
                        edges=np.array(edges, dtype=int))

    @staticmethod
    def collate(plans: list[MaskPlan]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        room_masks = torch.tensor(np.concatenate([plan.room_masks for plan in plans], axis=0))
        room_types = list(itertools.chain(*[plan.room_types for plan in plans]))
        room_types = torch.stack([room_type.one_hot() for room_type in room_types], dim=0)
        passed_rooms = 0
        edges = []
        nodes_to_graph = []
        for i, plan in enumerate(plans):
            edges.append(plan.edges + passed_rooms)
            nodes_to_graph.append(np.full(plan.room_masks.shape[0], i))
            passed_rooms += plan.room_masks.shape[0]
        edges = np.concatenate(edges, axis=0).T
        edges = torch.tensor(edges, dtype=torch.long)
        nodes_to_graph = np.concatenate(nodes_to_graph)
        nodes_to_graph = torch.tensor(nodes_to_graph, dtype=torch.long)
        return room_masks, room_types, edges, nodes_to_graph

    @staticmethod
    def collate_image(plans: list[MaskPlan]):
        MAX_ROOMS = 10

        padding = [MAX_ROOMS - len(plan.room_masks) for plan in plans]
        room_masks = [np.concatenate([plan.room_masks, np.zeros((padding[i], *plan.room_masks.shape[1:]))], axis=0) for
                      i, plan in enumerate(plans)]
        room_masks = np.stack(room_masks, axis=0)
        room_masks = torch.tensor(room_masks, dtype=torch.float32)
        room_types = [torch.cat([torch.stack([room_type.one_hot() for room_type in plan.room_types]),
                                 torch.zeros(padding[i], len(RoomType))], dim=0) for i, plan in enumerate(plans)]
        room_types = torch.stack(room_types, dim=0)
        src_key_padding_mask = [np.concatenate([np.zeros(MAX_ROOMS - padding[i], dtype=bool),
                                                np.ones(padding[i], dtype=bool)]) for i in range(len(plans))]
        src_key_padding_mask = np.stack(src_key_padding_mask, axis=0)
        src_key_padding_mask = torch.tensor(src_key_padding_mask, dtype=torch.bool)

        return room_masks, room_types, src_key_padding_mask

    def to_plan(self, canvas_size: tuple[int, int], mask_size: tuple[int, int]) -> Plan:

        def find_largest_contour(contours):
            max_area = 0
            max_contour = None
            for i, contour in enumerate(contours):
                contour = contour.astype(np.float32).squeeze()
                points = contour.reshape(-1, 2)
                polygon = Polygon(points)
                area = polygon.area
                if area > max_area:
                    max_area = area
                    max_contour = points
            return max_contour

        rooms = []
        for room_mask, room_type in zip(self.room_masks, self.room_types):
            contours, _ = cv2.findContours(room_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            points = find_largest_contour(contours)
            # Rescale from mask size to canvas size
            points = points * np.array(canvas_size) / np.array(mask_size)
            rooms.append(Plan.Room(room_type=room_type, corners=points))
        edges = []
        graph = nx.Graph()
        for u, v in self.edges:
            if u == v or rooms[u].room_type == RoomType.INTERIOR_DOOR or rooms[v].room_type == RoomType.INTERIOR_DOOR:
                continue
            graph.add_edge(u, v)

        # Remove doors
        for node in graph.nodes:
            if self.room_types[node] == RoomType.INTERIOR_DOOR:
                assert len(graph.neighbors(node)) == 2
                neighbor1, neighbor2 = graph.neighbors(node)
                edge_index = edges.index((neighbor1, neighbor2))
                edges[edge_index] = (neighbor1, neighbor2, rooms[node])

        return Plan(rooms=[room for room in rooms if room.room_type != RoomType.INTERIOR_DOOR], edges=edges)


ImagePlanCollated = tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]


@dataclass
class ImagePlan:
    image: np.ndarray
    walls: np.ndarray
    door_image: np.ndarray
    room_types: Optional[np.ndarray] = None
    bubbles: Optional[np.ndarray] = None

    def mask(self) -> np.ndarray:
        return self.image > -1

    @staticmethod
    def from_plan(plan: Plan, canvas_size: int = 256, mask_size: int = 64, with_bubbles: bool = True) -> ImagePlan:
        image = -np.ones((mask_size, mask_size), dtype=np.float32)
        door_image = -np.ones((mask_size, mask_size), dtype=np.float32)
        walls = -np.ones((mask_size, mask_size), dtype=np.float32)
        bubbles = -np.ones((mask_size, mask_size), dtype=np.float32) if with_bubbles else None
        bubble_edges = -np.ones((mask_size, mask_size), dtype=np.float32) if with_bubbles else None
        room_types = np.zeros(RoomType.restricted_length())
        scale = mask_size / canvas_size
        for room in plan.rooms:
            mask = room.to_mask((canvas_size, canvas_size), (mask_size, mask_size))
            if room.room_type == RoomType.FRONT_DOOR:
                door_image[mask > 0] = 0
            else:
                room_types[room.room_type.index_restricted()] += 1
                room_type_value = ImagePlan.room_type_to_value(room.room_type)
                image[mask > 0] = room_type_value
                corners = room.corners * mask_size / canvas_size
                walls = cv2.drawContours(walls, [corners.astype(np.int32)], -1, 1, 1)
                if with_bubbles:
                    centroid = room.centroid() * scale
                    radius = room.area() ** 0.5 / 3 * scale
                    bubbles = cv2.circle(bubbles, (int(centroid[0]), int(centroid[1])), int(radius), room_type_value,
                                         -1)

        for room1_idx, room2_idx, door in plan.edges:
            if door is None:
                continue
            room1, room2 = plan.rooms[room1_idx], plan.rooms[room2_idx]
            mask = door.to_mask((canvas_size, canvas_size), (mask_size, mask_size))
            door_image[mask > 0] = 1
            if with_bubbles:
                centroid1 = room1.centroid() * scale
                centroid2 = room2.centroid() * scale
                bubble_edges = cv2.line(bubble_edges, (int(centroid1[0]), int(centroid1[1])),
                                        (int(centroid2[0]), int(centroid2[1])), 1, 1)

        if with_bubbles:
            bubbles = np.stack([bubbles, bubble_edges], axis=0)

        return ImagePlan(image=image, walls=walls, door_image=door_image, room_types=room_types, bubbles=bubbles)

    @staticmethod
    def collate(plans: list[ImagePlan]) -> ImagePlanCollated:
        images = torch.tensor(np.stack([plan.image for plan in plans], axis=0), dtype=torch.float32).unsqueeze(1)
        walls = torch.tensor(np.stack([plan.walls for plan in plans], axis=0), dtype=torch.float32).unsqueeze(1)
        doors = torch.tensor(np.stack([plan.door_image for plan in plans], axis=0), dtype=torch.float32).unsqueeze(1)
        room_types = torch.tensor(np.stack([plan.room_types for plan in plans], axis=0), dtype=torch.float32) if plans[
                                                                                                                     0].room_types is not None else None
        bubbles = torch.tensor(np.stack([plan.bubbles for plan in plans], axis=0), dtype=torch.float32) if plans[
                                                                                                               0].bubbles is not None else None
        masks = images > -1
        return images, walls, doors, room_types, bubbles, masks

    @staticmethod
    def room_type_to_value(room_type: Optional[RoomType]) -> float:
        if room_type is None:
            return -1
        room_type_values = np.linspace(-1, 1, RoomType.restricted_length() + 1)
        return room_type_values[room_type.index_restricted() + 1].item()

    @staticmethod
    def value_to_room_type(value: float) -> Optional[RoomType]:
        room_type_values = np.linspace(-1, 1, RoomType.restricted_length() + 1)
        diffs = np.abs(room_type_values - value)
        index = diffs.argmin()
        if index == 0:
            return None
        return RoomType.from_index_restricted(index - 1)

    @staticmethod
    def value_to_room_type_np(values: np.ndarray) -> np.ndarray:
        room_type_values = np.linspace(-1, 1, RoomType.restricted_length() + 1)
        while len(room_type_values.shape) - 1 < len(values.shape):
            room_type_values = room_type_values[..., None]
        return np.abs(np.tile(room_type_values, (1, *values.shape)) - values).argmin(axis=0)

    @staticmethod
    def room_types_generator(room_types: dict[RoomType, int]) -> np.ndarray:
        room_type_vector = np.zeros(RoomType.restricted_length(), dtype=int)
        for room_type, count in room_types.items():
            room_type_vector[room_type.index_restricted()] = count

        return room_type_vector

    def to_image(self, alpha: bool = False) -> np.ndarray:
        def rescale(array: np.ndarray):
            return ((array + 1) / 2 * 255).astype(np.uint8)

        rooms = rescale(self.image)
        walls = rescale(self.walls)
        doors = rescale(self.door_image)
        if alpha:
            alpha_channel = (rooms + walls + doors) > 0
            alpha_channel = (alpha_channel * 255).astype(np.uint8)
            return np.stack([rooms, walls, doors, alpha_channel], axis=-1)
        else:
            return np.stack([rooms, walls, doors], axis=-1)

    def to_plan(self: ImagePlan, thin_walls: bool = False, simplify: Optional[float] = None, use_felzenszwalb: bool = False,
                image_only: boolean = False,
                target_size: Optional[tuple[int, int]] = None) -> tuple[Plan, ImagePlan]:
        def upsample(img, factor: tuple[int, int]):
            return np.repeat(np.repeat(img, factor[0], axis=0), factor[1], axis=1)

        # plt.imshow(labeled)
        # plt.show()

        # def thin(image: np.ndarray):
        #

        rooms_image = self.image.copy()
        doors_image = self.door_image.copy()
        walls_image = self.walls.copy()
        if target_size is not None:
            scale_factor = target_size[0] // self.walls.shape[0], target_size[1] // self.walls.shape[1]
            rooms_image = upsample(rooms_image, scale_factor)
            doors_image = upsample(doors_image, scale_factor)
            walls_image = upsample(walls_image, scale_factor)
        else:
            scale_factor = (1, 1)

        walls = walls_image > 0
        # walls = upsample(walls)
        # plt.imshow(walls)
        # plt.show()
        if use_felzenszwalb:
            segments = felzenszwalb(rooms_image, scale=250 * scale_factor[0] * scale_factor[1], sigma=0.4, min_size=60 * scale_factor[0] * scale_factor[1])
            # segments = felzenszwalb(self.image, scale=200, sigma=0.3, min_size=20)
            segment_walls = find_boundaries(segments, mode='outer', background=0)
            segment_walls = cv2.resize(segment_walls.astype(np.uint8), walls.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
            # segment_walls = segment_walls.astype(np.float32)
            # walls = walls.astype(np.float32)
            # window = cv2.createHanningWindow(walls.shape[::-1], cv2.CV_32F)
            #
            # # Compute shift
            # shift, response = cv2.phaseCorrelate(walls, segment_walls, window=window)
            #
            # # print(f"Estimated shift: dx={shift[0]:.2f}, dy={shift[1]:.2f}")
            #
            # # Apply the shift to mask2 to align it with mask1
            # dx, dy = shift
            # M = np.float32([[1, 0, -dx], [0, 1, -dy]])
            # aligned_mask2 = cv2.warpAffine(segment_walls, M, (segment_walls.shape[1], segment_walls.shape[0]), flags=cv2.INTER_NEAREST)
            # walls = (walls + aligned_mask2) > 0
            # walls = skeletonize((walls + segment_walls) > 0) + walls > 0
            # walls = walls + segment_walls > 0
            walls = segment_walls | walls
        if thin_walls:
            walls = skeletonize(walls)
        # plt.imshow(walls)
        # plt.show()
        # enclosing = ~walls if not use_felzenszwalb else ~segment_walls
        enclosing = ~walls
        labeled, num_rooms = label(enclosing)

        doors_diffs = np.abs(np.tile(doors_image, (3, 1, 1)) - np.tile(np.array([-1, 0, 1]).reshape(3, 1, 1),
                                                                       (1, doors_image.shape[0], doors_image.shape[1])))
        smooth_doors = np.argmin(doors_diffs, axis=0) - 1
        doors_image = smooth_doors.astype(float)

        # plt.imshow(doors_image)
        # plt.show()

        rooms = []
        for room_id in range(1, num_rooms + 1):
            mask = labeled == room_id
            if mask.sum() < 5:
                continue
            average_room_type = rooms_image[mask].mean()
            rooms_image[mask] = average_room_type

        # plt.imshow(rooms_image)
        # plt.show()

        filter_size = 5
        for x in range(rooms_image.shape[0]):
            for y in range(rooms_image.shape[1]):
                if not walls[x, y]:
                    continue
                closest_value = None
                best_diff = math.inf
                x_bounds = max(0, x - filter_size), min(rooms_image.shape[0], x + filter_size)
                y_bounds = max(0, y - filter_size), min(rooms_image.shape[1], y + filter_size)
                for i in range(x_bounds[0], x_bounds[1]):
                    for j in range(y_bounds[0], y_bounds[1]):
                        if walls[i, j] or (i == x and j == y):
                            continue
                        diff = abs(rooms_image[i, j] - rooms_image[x, y])
                        if diff < best_diff:
                            closest_value = rooms_image[i, j]
                            best_diff = diff

                if closest_value is not None:
                    rooms_image[x, y] = closest_value

        # plt.imshow(rooms_image)
        # plt.show()

        for value in np.unique(rooms_image):
            mask = rooms_image == value
            room_type = ImagePlan.value_to_room_type(value)
            room_type_value = ImagePlan.room_type_to_value(room_type)
            rooms_image[mask] = room_type_value
            if room_type is None or mask.sum() < 5 or image_only:
                continue
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polygons = [Polygon(contour.squeeze()) for contour in contours if contour.shape[0] > 2]
            polygon = max(polygons, key=lambda polygon: polygon.area)

            if simplify is not None:
                polygon = polygon.simplify(tolerance=simplify, preserve_topology=True)
            rooms.append(Plan.Room(room_type=room_type, corners=np.array(polygon.exterior.coords)))

        plan = Plan(rooms=rooms) if not image_only else None
        return plan, ImagePlan(walls=walls * 2 - 1, image=rooms_image, door_image=doors_image)

    @staticmethod
    def from_image(image: np.ndarray) -> ImagePlan:
        assert image.ndim == 3, "Image must be a 2D array."
        img = image.copy()
        rooms = img[:,:, 0]
        walls = img[:, :, 1]
        doors = img[:, :, 2]
        walls = walls / 255 * 2 - 1
        doors = doors / 255 * 2 - 1
        rooms = rooms / 255 * 2 - 1
        return ImagePlan(image=rooms, walls=walls, door_image=doors)

