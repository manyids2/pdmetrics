from pathlib import Path
from typing import List, Union
from .metrics import pdMetrics


class pdF1(pdMetrics):
    name: str = "f1"
    tracked: List[str] = ["f1", "precision", "recall", "tp", "fp", "fn"]

    def __init__(self, db_path: Union[Path, str]) -> None:
        super().__init__(self.name, self.tracked, db_path)
