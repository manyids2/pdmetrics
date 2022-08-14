from pathlib import Path
from typing import Union
import pandas as pd
import sqlite3


def save_df(
    df: pd.DataFrame,
    db_path: Union[Path, str],
    table: str,
    if_exists: str = "replace",
    index: bool = False,
    verbose: bool = False,
):
    con = sqlite3.connect(Path(db_path).absolute())
    df.to_sql(table, con, if_exists=if_exists, index=index)
    if verbose:
        print(f"\nSaved => {table} -> {db_path}\n\n{df}\n")
    return df


def load_df(
    db_path: Union[Path, str], table: str, verbose: bool = False
) -> pd.DataFrame:
    con = sqlite3.connect(Path(db_path).absolute())
    df = pd.read_sql_query(f"SELECT * from ({table})", con)
    if verbose:
        print(f"\nLoaded => {table} -> {db_path}\n\n{df}\n")
    return df


class pdData:
    @classmethod
    def from_columns(cls, columns):
        df = pd.DataFrame(columns=columns)
        return cls(df)

    def __init__(self, df: pd.DataFrame):
        self.df = df

        # Set row type depending on column

    def __repr__(self) -> str:
        return repr(self.df)
