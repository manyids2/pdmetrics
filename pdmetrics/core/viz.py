from typing import Dict
from rich import print
import pandas as pd

SCHEMA = {
    "image_file": "path-image",
    "mask_file": "path-image",
    "overlay_file": "path-image",
    "boxes_file": "path-npz",
    "slide_file": "path-wsi",
    "ratings": "int-ratings",
    "comments": "str-comments",
    "source": "str-source",
    "split": "str-split",
}


class pdViz:
    def __init__(self, df: pd.DataFrame, schema: Dict[str, str]):
        self.df = df
        print(schema)
