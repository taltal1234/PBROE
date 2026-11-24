"""Small helpers that more than one pipeline script needs."""
import numpy as np
import pandas as pd
import config as C


def load(path):
    """Read a WRDS parquet file with plain float and object columns.

    WRDS returns pandas' nullable types (Float64, string). They behave differently from
    plain numpy columns in a few places, so I convert everything once, right here.
    """
    df = pd.read_parquet(path)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype("float64")
        elif pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype(object).where(df[col].notna(), None)
    return df


def months_apart(later, earlier):
    """Number of months between two columns of monthly periods (NaN where either is missing)."""
    return (later - earlier).apply(lambda d: d.n if pd.notna(d) else np.nan)


def save_table(rows, name, digits):
    """Write a list of result rows to results/tables/<name>.csv, rounded, and return it unrounded."""
    df = pd.DataFrame(rows)
    df.round(digits).to_csv(C.TABLE_DIR / f"{name}.csv", index=False)
    return df
