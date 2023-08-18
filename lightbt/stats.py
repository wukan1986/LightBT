from typing import List

import pandas as pd


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


def pnl_by_asset(perf, asset: int, close: pd.DataFrame) -> pd.DataFrame:
    """单资产的盈亏信息

    Parameters
    ----------
    perf: pd.DataFrame
        输入为`bt.performances()`的输出
    asset: int
        资产id
    close: pd.DataFrame
        行情

    Returns
    -------
    pd.DataFrame
        多加了盈亏曲线

    Examples
    --------
    >>> pd.options.plotting.backend = 'plotly'
    >>> perf = bt.performances()
    >>> tmp = df[['date', 'asset', 'CLOSE']].set_index(['date', 'asset'])
    >>> pnls = pnl_by_asset(perf, bt.asset_str2int('s_0000'), tmp)
    >>> pnls.plot()
    >>> pd.options.plotting.backend = 'matplotlib'
    >>> pnls[['PnL', 'CLOSE']].plot(secondary_y='CLOSE')

    """
    df1 = perf[perf['asset'] == asset]

    if close is None:
        df = df1
        agg = {'value': 'sum', 'margin': 'sum', 'upnl': 'sum', 'cum_pnl': 'sum', 'cum_commission': 'sum', 'amount': 'sum'}
    else:
        df0 = close.reset_index()
        df2 = df0[df0['asset'] == asset]
        df = pd.merge(left=df1, right=df2, left_on=['date', 'asset'], right_on=['date', 'asset'])
        agg = {'value': 'sum', 'margin': 'sum', 'upnl': 'sum', 'cum_pnl': 'sum', 'cum_commission': 'sum', 'amount': 'sum', close.columns[0]: 'last'}

    p = df.set_index(['date', 'asset']).groupby(by=['date']).agg(agg)
    # 盈亏曲线=持仓盈亏+累计盈亏+累计手续费
    p['PnL'] = p['upnl'] + p['cum_pnl'] - p['cum_commission']
    return p


def pnl_by_assets(perf: pd.DataFrame, assets: List) -> pd.DataFrame:
    """多个资产的盈亏曲线

    Parameters
    ----------
    perf: pd.DataFrame
        输入为`bt.performances()`的输出
    assets: list[int]
        关注的资产列表

    Returns
    -------
    pd.DataFrame
        多资产盈亏曲线矩阵

    Examples
    --------
    >>> perf = bt.performances()
    >>> pnls = pnl_by_assets(perf, bt.asset_str2int(['s_0000', 's_1000', 's_4999']))
    >>> pnls.columns = bt.asset_int2str(pnls.columns)
    >>> pnls.plot()

    """
    # 单资产的盈亏曲线
    df = perf[perf['asset'].isin(assets)]
    df = df.set_index(['date', 'asset'])
    df['PnL'] = df['upnl'] + df['cum_pnl'] - df['cum_commission']
    return df['PnL'].unstack().fillna(method='ffill')
