from typing import Dict, List
import numpy as np


def get_all_zeros(shape: List[int], _: int) -> Dict[str, np.ndarray]:
    preds = np.zeros(shape=shape)
    target = np.zeros(shape=shape)
    return {
        "preds": preds,
        "target": target,
    }


def get_all_correct(shape: List[int], num_classes: int) -> Dict[str, np.ndarray]:
    preds = np.random.randint(low=0, high=num_classes, size=shape)
    target = preds.copy()
    return {
        "preds": preds,
        "target": target,
    }


def get_all_wrong(shape: List[int], num_classes: int) -> Dict[str, np.ndarray]:
    preds = np.zeros(shape=shape)
    target = np.random.randint(low=1, high=num_classes, size=shape)
    return {
        "preds": preds,
        "target": target,
    }


def get_random(shape: List[int], num_classes: int) -> Dict[str, np.ndarray]:
    preds = np.random.randint(low=0, high=num_classes, size=shape)
    target = np.random.randint(low=0, high=num_classes, size=shape)
    return {
        "preds": preds,
        "target": target,
    }


class Classification:

    getters = {
        "random": get_random,
        "all_correct": get_all_correct,
        "all_wrong": get_all_wrong,
        "all_zeros": get_all_zeros,
    }

    def __init__(self, shape: List[int], num_classes: int = 10) -> None:
        self.shape = shape
        self.num_classes = num_classes

    def get_preds_target(self, mtype: str):
        """mtype: random, all_correct, all_wrong, all_zeros."""
        return self.getters[mtype](self.shape, self.num_classes)

    def __repr__(self) -> str:
        return (
            f"Classification =>\n"
            f"         shape -> {self.shape}\n"
            f"   num_classes -> {self.num_classes}\n"
        )
