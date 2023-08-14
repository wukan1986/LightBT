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
        指定类型能提速

    Returns
    -------
    np.ndarray
        迭代器

    Notes
    -----
    `df`一定要提前按`by`排序，否则结果是错的

    """
    # 控制同样的位置。否则record转dtype失败会导致效率低
    df = df[list(dtype.names)]

    if isinstance(df, pd.DataFrame):
        # 按日期时间标记，每段的第一天标记为True
        time_0 = df[by]
        time_1 = time_0.shift(1)
        time_diff = (time_0 != time_1).to_numpy()

        # recarray转np.ndarray
        arr = np.asarray(df.to_records(index=False), dtype=dtype)
    else:
        arr = df.copy()

    idx = np.argwhere(time_diff)
    idx = np.append(idx, [len(arr)])
    for i, j in zip(idx[:-1], idx[1:]):
        yield arr[i:j]
