import time
from typing import Optional

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


def groupby(df: pd.DataFrame, by: str, dtype: Optional[np.dtype] = None) -> np.ndarray:
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
    if dtype is not None:
        # 控制同样的位置。否则record转dtype失败会导致效率低
        df = df[list(dtype.names)]

    if isinstance(df, pd.DataFrame):
        # recarray转np.ndarray
        arr = np.asarray(df.to_records(index=False), dtype=dtype)

        # 这里支持复合分组
        idx = df.groupby(by=by)['asset'].count().cumsum().to_numpy()
        idx = np.insert(idx, 0, 0)
    else:
        # 原数据是否需要复制？从代码上看没有复制之处
        arr = df  # .copy()

        dt = arr[by]
        flag = np.ones(shape=len(dt) + 1, dtype=bool)
        # 前后都为True
        flag[1:-1] = dt[:-1] != dt[1:]
        idx = np.argwhere(flag).flatten()

    for i, j in zip(idx[:-1], idx[1:]):
        # 由于标记的位置正好是每段的开始位置，所以j不需加1
        yield arr[i:j]
