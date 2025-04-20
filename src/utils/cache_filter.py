from functools import partial

from src.constants import MAX_POSSIBLE_CORNERS
from src.rplan.dataset import RPlanFilter, max_corner_filter


def main():
    filter_fn = partial(max_corner_filter, 32, 100)
    RPlanFilter('data/rplan', filter_fn, 'data/filters', 'total_points_filter.npy')

if __name__ == '__main__':
    main()
