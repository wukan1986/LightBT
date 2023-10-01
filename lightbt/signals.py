import numpy as np
import pandas as pd
from numba import njit, float64, bool_


def state_to_action(state: pd.DataFrame):
    """将持仓状态转换成下单操作。

    1. 开头出现nan
    2. 中段出现nan

    Parameters
    ----------
    state: pd.DataFrame
        持仓状态宽表。值应当都是整数，浮点和nan都是不合法数据

    Returns
    -------
    pd.DataFrame
        下单操作

    """
    state = state.fillna(0)  # 防止nan计算有问题
    action = state.diff()
    action.iloc[0] = state.iloc[0]
    return action


def action_to_state(action: pd.DataFrame):
    """将操作转换成状态

    Parameters
    ----------
    action: pd.DataFrame
        下单操作宽表

    Examples
    --------
    s = pd.DataFrame({
    'a': [1, 1, 2, 0, -1, 0],
    'b': [np.nan, np.nan, 1, 0, 0, 0],
    })
    print(s)
    a = state_to_action(s)
    print(a)
    s = action_to_state(a)
    print(s)

    """
    action = action.fillna(0)
    return action.cumsum()


@njit(float64[:](bool_[:], bool_[:], bool_[:], bool_[:], bool_, bool_), fastmath=True, nogil=True, cache=True)
def signals_to_amount(is_long_entry: np.ndarray, is_long_exit: np.ndarray,
                      is_short_entry: np.ndarray, is_short_exit: np.ndarray,
                      accumulate: bool = False,
                      action: bool = False) -> np.ndarray:
    """将4路信号转换成持仓状态。适合按资产分组后的长表

    在`LongOnly`场景下，`is_short_entry`和`is_short_exit`输入数据值都为`False`即可

    Parameters
    ----------
    is_long_entry: np.ndarray
        是否多头入场
    is_long_exit: np.ndarray
        是否多头出场
    is_short_entry: np.ndarray
        是否空头入场
    is_short_exit: np.ndarray
        是否空头出场
    accumulate: bool
        遇到重复信号时是否累计
    action: bool
        返回持仓状态还是下单操作

    Returns
    -------
    np.ndarray
        持仓状态

    Examples
    --------
    ```python
    long_entry = np.array([True, True, False, False, False])
    long_exit = np.array([False, False, True, False, False])
    short_entry = np.array([False, False, True, False, False])
    short_exit = np.array([False, False, False, True, False])

    amount = signals_to_amount(long_entry, long_exit, short_entry, short_exit, accumulate=True, action=False)
    ```

    """
    _amount: float = 0.0  # 持仓状态
    _action: float = 0.0  # 下单方向
    out_amount = np.zeros(len(is_long_entry), dtype=float)
    out_action = np.zeros(len(is_long_entry), dtype=float)
    for i in range(len(is_long_entry)):
        if _amount == 0.0:
            # 多头信号优先级高于空头信号
            if is_long_entry[i]:
                _amount += 1.0
                _action = 1.0
            elif is_short_entry[i]:
                _amount -= 1.0
                _action = -1.0
        elif _amount > 0.0:
            if is_long_exit[i]:
                _amount -= 1.0
                _action = -1.0
            elif is_long_entry[i] and accumulate:
                _amount += 1.0
                _action = 1.0
        else:
            if is_short_exit[i]:
                _amount += 1.0
                _action = 1.0
            elif is_short_entry[i] and accumulate:
                _amount -= 1.0
                _action = -1.0
        out_amount[i] = _amount
        out_action[i] = _action

    if action:
        return out_action
    else:
        return out_amount


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
    df['date_diff'] = date_0 != date_0.shift(-1, fill_value=0)

    return df


def orders_weekly(df: pd.DataFrame) -> pd.DataFrame:
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
    date_0 = df['date'].dt.isocalendar().week
    # week的范围是1~52，所以可以fill_value写0 或>52的值
    df['date_diff'] = date_0 != date_0.shift(-1, fill_value=99)

    return df
