from pathlib import Path
import pandas as pd
import sqlite3


def save_df(df: pd.DataFrame, db_path: Path, table: str, verbose: bool = False):
    con = sqlite3.connect(db_path.absolute())
    df.to_sql(table, con)
    if verbose:
        print(f"Saved => {table} -> {db_path}\n\n{df}\n")
    return df


def load_df(db_path: Path, table: str, verbose: bool = False) -> pd.DataFrame:
    con = sqlite3.connect(db_path.absolute())
    df = pd.read_sql_table(table, con)
    if verbose:
        print(f"Loaded => {table} -> {db_path}\n\n{df}\n")
    return df


class pdData:
    @classmethod
    def from_columns(cls, columns):
        df = pd.DataFrame(columns=columns)
        return cls(df)

    def __init__(self, df: pd.DataFrame):
        self.df = df

        # Set row type depending on column
