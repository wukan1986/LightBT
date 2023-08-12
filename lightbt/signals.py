import pandas as pd


def orders_daily(df: pd.DataFrame, mapping_asset_int: dict) -> pd.DataFrame:
    """"""
    # 这一步比较慢，是否能再提速
    df['asset'] = df['asset'].map(mapping_asset_int)

    # 提前排序，之后就可以直接使用
    df.sort_values(by=['date'], inplace=True)

    # 按日期标记，每段的最后一天标记为True
    date_0 = df['date'].dt.date
    df['date_diff'] = date_0 != date_0.shift(-1)
    # 按日期时间标记，每段的第一天标记为True
    time_0 = df['date']
    df['time_diff'] = time_0 != time_0.shift(1)

    return df
