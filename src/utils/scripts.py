import argparse
from enum import StrEnum

class ModelChoice(StrEnum):
    ROOM_TYPES = 'room_types'
    BUBBLES_OLD = 'bubbles_old'
    BUBBLES = 'bubbles'
    BUBBLES_CFG = 'bubbles_cfg'

MODEL_CHOICES_STR = [choice.value for choice in ModelChoice]

def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Script to generate samples from a model.", prog='HouseCFG Sampler')
    parser.add_argument('--type', '-t', type=str, choices=MODEL_CHOICES_STR,
                        required=True, default='bubbles', )
    parser.add_argument('--input', '-i', type=str, help='Path to the input data.', default='data/rplan')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='Batch size for sampling.')
    return parser