from pathlib import Path
from typing import List, Union

data_dir = Path("./data")


class pdMetrics:
    def __init__(
        self, name: str, tracked: List[str], db_path: Union[Path, str]
    ) -> None:
        self.name = name
        self.tracked = tracked
        self.db_path = Path(db_path)

    def __repr__(self) -> str:
        return (
            f"{self.name} =>\n"
            f"  db_path -> {self.db_path}\n"
            f"  tracked -> {self.tracked}\n"
        )
