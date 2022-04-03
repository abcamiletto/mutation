from dataclasses import dataclass

import numpy as np


@dataclass
class Variant:
    lamda: float
    gamma: float
    beta_self: np.ndarray
    alpha: float
    frequency: float
    parent: int


def create_register(l, g, B, a, f):
    variants = []
    for items in zip(l, g, np.diagonal(B), a, f):
        items = [x.item() for x in items]
        variants.append(Variant(*items, None))
    return variants
