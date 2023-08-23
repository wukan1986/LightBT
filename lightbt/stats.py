from typing import List, Dict, Union

import numpy as np
import pandas as pd

from lightbt.enums import trades_stats_dt, roundtrip_stats_dt, roundtrip_dt
from lightbt.utils import groupby


def total_equity(perf: pd.DataFrame) -> pd.DataFrame:
    """总权益统计。

    Parameters
    ----------
    perf: pd.DataFrame
        输入为`bt.performances()`的输出

    Returns
    -------
    pd.DataFrame
        多加了总权益

    Examples
    --------
    >>> pd.options.plotting.backend = 'plotly'
    >>> perf = bt.performances()
    >>> equity = total_equity(perf)
    >>> equity.plot()

    """
    # 此处的amount返回的是净持仓数量
    agg = {'cash': 'last', 'value': 'sum', 'margin': 'sum', 'upnl': 'sum', 'cum_pnl': 'sum', 'cum_commission': 'sum', 'amount': 'sum'}
    p = perf.set_index(['date', 'asset']).groupby(by=['date']).agg(agg)
    # 总权益曲线。cash中已经包含了pnls和commissions
    p['equity'] = p['cash'] + p['margin'] + p['upnl']
    return p


def pnl_by_asset(perf, asset: Union[int, str], close: pd.DataFrame,
                 mapping_asset_int: Dict[str, int],
                 mapping_int_asset: Dict[int, str]) -> pd.DataFrame:
    """单资产的盈亏信息

    Parameters
    ----------
    perf: pd.DataFrame
        输入为`bt.performances()`的输出
    asset: int or str
        资产id
    close: pd.DataFrame
        行情
    mapping_asset_int: dict
        资产 字符串转数字
    mapping_int_asset: dict
        资产 数字转字符串

    Returns
    -------
    pd.DataFrame
        多加了盈亏曲线

    Examples
    --------
    >>> pd.options.plotting.backend = 'plotly'
    >>> pnls = pnls = pnl_by_asset(perf, 's_0000', df[['date', 'asset', 'CLOSE']], bt.mapping_asset_int, bt.mapping_int_asset)
    >>> pnls.plot()
    >>> pd.options.plotting.backend = 'matplotlib'
    >>> pnls[['PnL', 'CLOSE']].plot(secondary_y='CLOSE')

    """
    if pd.api.types.is_string_dtype(perf['asset']):
        if isinstance(asset, int):
            asset = mapping_int_asset.get(asset)
    elif pd.api.types.is_integer_dtype(perf['asset']):
        if isinstance(asset, str):
            asset = mapping_asset_int.get(asset)

    df1 = perf[perf['asset'] == asset]

    if close is None:
        df = df1
        agg = {'value': 'sum', 'margin': 'sum', 'upnl': 'sum', 'cum_pnl': 'sum', 'cum_commission': 'sum', 'amount': 'sum'}
    else:
        if pd.api.types.is_string_dtype(close['asset']):
            if isinstance(asset, int):
                close = close.copy()
                close['asset'] = close['asset'].map(mapping_asset_int)
        elif pd.api.types.is_integer_dtype(close['asset']):
            if isinstance(asset, str):
                close = close.copy()
                close['asset'] = close['asset'].map(mapping_int_asset)

        df2 = close[close['asset'] == asset]
        df = pd.merge(left=df1, right=df2, left_on=['date', 'asset'], right_on=['date', 'asset'])
        agg = {'value': 'sum', 'margin': 'sum', 'upnl': 'sum', 'cum_pnl': 'sum', 'cum_commission': 'sum', 'amount': 'sum', close.columns[-1]: 'last'}

    p = df.set_index(['date', 'asset']).groupby(by=['date']).agg(agg)
    # 盈亏曲线=持仓盈亏+累计盈亏+累计手续费
    p['PnL'] = p['upnl'] + p['cum_pnl'] - p['cum_commission']
    return p


def pnl_by_assets(perf: pd.DataFrame,
                  assets: Union[List[str], List[int]],
                  mapping_asset_int: Dict[str, int],
                  mapping_int_asset: Dict[int, str]) -> pd.DataFrame:
    """多个资产的盈亏曲线

    Parameters
    ----------
    perf: pd.DataFrame
        输入为`bt.performances()`的输出
    assets: list[int] or list[str]
        关注的资产列表
    mapping_asset_int: dict
        资产 字符串转数字
    mapping_int_asset: dict
        资产 数字转字符串

    Returns
    -------
    pd.DataFrame
        多资产盈亏曲线矩阵

    Examples
    --------
    >>> perf = bt.performances()
    >>> pnls = pnl_by_assets(perf, ['s_0000', 's_0100', 's_0300'], bt.mapping_asset_int, bt.mapping_int_asset)
    >>> pnls.plot()

    """
    if pd.api.types.is_string_dtype(perf['asset']):
        if isinstance(assets[0], int):
            assets = list(map(mapping_int_asset.get, assets))
    elif pd.api.types.is_integer_dtype(perf['asset']):
        if isinstance(assets[0], str):
            assets = list(map(mapping_asset_int.get, assets))

    # 单资产的盈亏曲线
    df = perf[perf['asset'].isin(assets)]
    df = df.set_index(['date', 'asset'])
    df['PnL'] = df['upnl'] + df['cum_pnl'] - df['cum_commission']
    return df['PnL'].unstack().fillna(method='ffill')


def trades_to_roundtrips(trades: np.ndarray, asset_count: int) -> np.ndarray:
    """多笔成交转为成对的交易轮

    Parameters
    ----------
    trades: np.ndarray
        全体成交记录
    asset_count: int
        总资产数。用于分配足够的空间用于返回

    Returns
    -------
    np.ndarray

    """
    trades = trades[trades['asset'].argsort(kind='stable')]
    groups = groupby(trades, by='asset', dtype=None)

    records = np.zeros(len(trades) // 2 + 1 + asset_count, dtype=roundtrip_dt)
    k = 0  # 目标位置
    for group in groups:
        # 每段开始位置
        flag = np.ones(shape=len(group) + 1, dtype=bool)
        flag[1:] = group['amount'] == 0.0
        flag[-1] = True
        idx = np.argwhere(flag).flatten()
        for i, j in zip(idx[:-1], idx[1:]):
            g = group[i:j]
            rec = records[k]

            is_open = g[g['is_open']]
            is_close = g[~g['is_open']]

            rec['asset'] = g['asset'][0]
            rec['is_long'] = g['is_buy'][0]
            rec['is_close'] = g['amount'][-1] == 0.0
            rec['qty'] = np.sum(is_open['qty'])
            rec['entry_date'] = g['date'][0]
            rec['entry_price'] = np.mean(is_open['fill_price'])
            rec['entry_commission'] = np.sum(is_open['commission'])
            rec['exit_date'] = g['date'][-1]
            if len(is_close) > 0:
                rec['exit_price'] = np.mean(is_close['fill_price'])
                rec['exit_commission'] = np.sum(is_close['commission'])
            rec['pnl'] = np.sum(g['pnl'])
            rec['pnl_com'] = rec['pnl'] - rec['entry_commission'] - rec['exit_commission']

            k += 1
    return records[:k]


def calc_roundtrips_stats(roundtrips: np.ndarray, asset_count: int) -> np.ndarray:
    """每轮交易统计

    Parameters
    ----------
    roundtrips: np.ndarray
        全体每轮交易
    asset_count: int
        总资产数。用于分配足够的空间用于返回

    Returns
    -------
    np.ndarray

    """
    roundtrips = roundtrips[roundtrips['asset'].argsort(kind='stable')]
    groups = groupby(roundtrips, by='asset', dtype=None)

    records = np.zeros(asset_count, dtype=roundtrip_stats_dt)
    for i, g in enumerate(groups):
        rec = records[i]

        is_long = g[g['is_long']]
        is_short = g[~g['is_long']]
        winning = g[g['pnl_com'] > 0.0]
        losing = g[g['pnl_com'] < 0.0]

        rec['asset'] = g['asset'][0]
        rec['total_count'] = len(g)
        rec['long_count'] = len(is_long)
        rec['short_count'] = len(is_short)
        rec['long_count'] = len(is_long)
        rec['short_count'] = len(is_short)
        rec['winning_count'] = len(winning)
        rec['losing_count'] = len(losing)
        rec['win_rate'] = rec['winning_count'] / rec['total_count']

    return records[:i + 1]


def calc_trades_stats(trades: np.ndarray, asset_count: int) -> np.ndarray:
    """成交统计

    Parameters
    ----------
    trades: np.ndarray
        全体交易记录
    asset_count: int
        总资产数。用于分配足够的空间用于返回

    Returns
    -------
    np.ndarray

    """
    trades = trades[trades['asset'].argsort(kind='stable')]  # stable一定要有，否则乱序
    groups = groupby(trades, by='asset', dtype=None)

    records = np.zeros(asset_count, dtype=trades_stats_dt)
    for i, g in enumerate(groups):
        rec = records[i]

        is_buy = g[g['is_buy']]
        is_sell = g[~g['is_buy']]

        rec['asset'] = g['asset'][0]
        rec['start'] = g['date'][0]
        rec['end'] = g['date'][-1]
        rec['period'] = rec['end'] - rec['start']
        rec['total_count'] = len(g)
        rec['buy_count'] = len(is_buy)
        rec['sell_count'] = len(is_sell)
        rec['min_qty'] = np.min(g['qty'])
        rec['max_qty'] = np.max(g['qty'])
        rec['avg_qty'] = np.mean(g['qty'])
        rec['total_commission'] = np.sum(g['commission'])
        rec['min_commission'] = np.min(g['commission'])
        rec['max_commission'] = np.max(g['commission'])
        rec['avg_commission'] = np.mean(g['commission'])

        if len(is_buy) > 0:
            rec['avg_buy_qty'] = np.mean(is_buy['qty'])
            rec['avg_buy_price'] = np.mean(is_buy['fill_price'])
            rec['avg_buy_commission'] = np.mean(is_buy['commission'])
        if len(is_sell) > 0:
            rec['avg_sell_qty'] = np.mean(is_sell['qty'])
            rec['avg_sell_price'] = np.mean(is_sell['fill_price'])
            rec['avg_sell_commission'] = np.mean(is_sell['commission'])

    return records[:i + 1]
