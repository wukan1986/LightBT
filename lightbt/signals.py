import pandas as pd


def orders_daily(df: pd.DataFrame) -> pd.DataFrame:
    """

    Parameters
    ----------
    df

    Returns
    -------
    pd.DataFrame
        1. 已经按时间进行了排序。sort_values
        2. 添加了日期标记，用于触发内部的绩效快照

    Notes
    -----
    有多处修改了数据，所以需要`copy`。`sort_values`隐含了`copy`

    """
    # 全体数据排序，并复制
    df = df.sort_values(by=['date'])

    # 按日期标记，每段的最后一条标记为True。一定要提前排序
    date_0 = df['date'].dt.date
    df['date_diff'] = date_0 != date_0.shift(-1)

    return df
