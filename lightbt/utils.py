import time

import numpy as np
import pandas as pd


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"{func.__name__} executed in {end_time - start_time} seconds")
        return result

    return wrapper


class Timer:
    def __init__(self):
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        print(f"code executed in {end_time - self.start_time} seconds")


def groupby_np(df: pd.DataFrame, by: str, dtype: np.dtype) -> np.ndarray:
    """简版数据分组

    Parameters
    ----------
    df: pd.DataFrame
    by: str
    dtype: np.dtype

    Returns
    -------

    """
    if isinstance(df, pd.DataFrame):
        arr = np.asarray(df[list(dtype.names)].to_records(index=False), dtype=dtype)
    else:
        arr = df

    idx = np.argwhere(arr[by]).reshape(-1)
    idx = np.append(idx, [len(arr)])
    for i, j in zip(idx[:-1], idx[1:]):
        yield arr[i:j]
