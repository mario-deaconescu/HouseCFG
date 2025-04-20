from typing import Generic, TypeVar

import numpy as np

T = TypeVar('T')
class LinkedList:

    class Node:
        def __init__(self, data: T, next=None, prev=None):
            self.data = data
            self.next = next
            self.prev = prev

        def __str__(self):
            return str(self.data)

        def __repr__(self):
            return f'[{str(self.prev)} <- {str(self)} -> {str(self.next)}]'

        def __eq__(self, other):
            return self.data == other.data

    def __init__(self):
        self.head = None

    def append(self, data: T):
        if self.head is None:
            self.head = self.Node(data)
            return
        current = self.head
        while current.next is not None:
            current = current.next
        current.next = self.Node(data, None, current)

    def __iter__(self):
        current = self.head
        while current is not None:
            yield current.data
            current = current.next

    def iter_nodes(self):
        current = self.head
        while current is not None:
            yield current
            current = current.next