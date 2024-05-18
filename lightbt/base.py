import time
from typing import List, Union

import numpy as np
import pandas as pd

from lightbt.stats import calc_trades_stats, calc_roundtrips_stats, trades_to_roundtrips


class LightBT:
    def __init__(self,
                 init_cash: float = 10000,
                 positions_precision: float = 1.0,
                 max_trades: int = 10000,
                 max_performances: int = 10000,
                 ) -> None:
        """初始化

        Parameters
        ----------
        positions_precision: float
            持仓精度
        max_trades: int
            记录成交的缓存大小。空间不足时将丢弃
        max_performances: int
            记录绩效的缓存大小。空间不足时将丢弃

        """
        from lightbt.portfolio import Portfolio

        self._init_cash = init_cash
        self._positions_precision = positions_precision
        self._max_trades = max_trades
        self._max_performances = max_performances

        self.pf = Portfolio(positions_precision=self._positions_precision,
                            max_trades=self._max_trades,
                            max_performances=self._max_performances)
        # 入金
        self.deposit(self._init_cash)

        # 底层没有资产名字符串，只有纯数字
        self.mapping_asset_int = {}
        self.mapping_int_asset = {}
        self.conf: pd.DataFrame = pd.DataFrame()

    def reset(self):
        """重置。不需要再次`setup`，只需要重新跑一次`run_`即可"""
        self.pf.reset()
        # 入初始资金
        self.deposit(self._init_cash)

    def setup(self, df: pd.DataFrame) -> None:
        """映射资产，配置合约乘数和保证金率

        同名的会进行替换

        Parameters
        ----------
        df: pd.DataFrame
            - asset
            - mult
            - margin_ratio
            - commission_ratio
                手续费率
            - commission_fn
                手续费计算函数

        """
        self.conf = pd.concat([self.conf, df])
        self.conf.drop_duplicates(subset='asset', keep='first', inplace=True)

        # 资产与底层持仓位置的映射
        conf = self.conf.reset_index(drop=True)
        self.mapping_int_asset = conf['asset'].to_dict()
        self.mapping_asset_int = {v: k for k, v in self.mapping_int_asset.items()}

        # 转成底层方便的格式
        asset = np.asarray(conf.index, dtype=int)
        mult = np.asarray(conf['mult'], dtype=float)
        margin_ratio = np.asarray(conf['margin_ratio'], dtype=float)
        commission_ratio = np.asarray(conf['commission_ratio'], dtype=float)
        commission_fn = np.asarray(conf['commission_fn'])

        # 调用底层的批量处理函数
        self.pf.setup(asset, mult, margin_ratio, commission_ratio)
        # 设置手续费函数
        for aid, fn in zip(asset, commission_fn):
            self.pf.set_commission_fn(aid, fn)

    def asset_str2int(self, strings: Union[List[str], str]) -> Union[List[int], int]:
        """资产转换。字符串转数字"""
        if isinstance(strings, list) and len(strings) == 1:
            strings = strings[0]
        if isinstance(strings, str):
            return self.mapping_asset_int.get(strings)

        return list(map(self.mapping_asset_int.get, strings))

    def asset_int2str(self, integers: Union[List[int], int]) -> Union[List[str], str]:
        """资产转换。数字转字符串"""
        if isinstance(integers, list) and len(integers) == 1:
            integers = integers[0]
        if isinstance(integers, int):
            return self.mapping_int_asset.get(integers)

        return list(map(self.mapping_int_asset.get, integers))

    def deposit(self, cash: float) -> float:
        """入金

        Parameters
        ----------
        cash: float

        Returns
        -------
        float

        Notes
        -----
        默认资金为0，所以交易前需要入金

        """
        return self.pf.deposit(cash)

    def withdraw(self, cash: float) -> float:
        """出金"""
        return self.pf.withdraw(cash)

    def positions(self, readable: bool = True) -> Union[pd.DataFrame, np.ndarray]:
        """持仓记录"""
        records = self.pf.positions()
        if not readable:
            return records

        df = pd.DataFrame.from_records(records)
        df['asset'] = df['asset'].map(self.mapping_int_asset)
        return df

    def trades(self, return_all: bool, readable: bool = True) -> Union[pd.DataFrame, np.ndarray]:
        """成交记录

        Parameters
        ----------
        return_all: bool
            返回所有记录或返回最近一批记录
        readable: bool
            返回可读格式

        Returns
        -------
        pd.DataFrame or np.ndarray

        """
        records = self.pf.trades(return_all)
        if not readable:
            return records

        df = pd.DataFrame.from_records(records)
        df['date'] = pd.to_datetime(df['date'])
        df['asset'] = df['asset'].map(self.mapping_int_asset)
        return df

    def performances(self, return_all: bool, readable: bool = True) -> Union[pd.DataFrame, np.ndarray]:
        """绩效记录"""
        records = self.pf.performances(return_all)
        if not readable:
            return records

        df = pd.DataFrame.from_records(records)
        df['date'] = pd.to_datetime(df['date'])
        df['asset'] = df['asset'].map(self.mapping_int_asset)
        return df

    def trades_stats(self, readable: bool = True) -> Union[pd.DataFrame, np.ndarray]:
        """成交统计"""
        trades = self.pf.trades(True)
        stats = calc_trades_stats(trades, len(self.mapping_int_asset))
        if not readable:
            return stats

        df = pd.DataFrame.from_records(stats)
        df['start'] = pd.to_datetime(df['start'])
        df['end'] = pd.to_datetime(df['end'])
        df['period'] = pd.to_timedelta(df['period'])
        df['asset'] = df['asset'].map(self.mapping_int_asset)
        return df

    def roundtrips(self, readable: bool = True) -> Union[pd.DataFrame, np.ndarray]:
        """每轮交易记录"""
        trades = self.pf.trades(True)
        rounds = trades_to_roundtrips(trades, len(self.mapping_int_asset))
        if not readable:
            return rounds

        df = pd.DataFrame.from_records(rounds)
        df['entry_date'] = pd.to_datetime(df['entry_date'])
        df['exit_date'] = pd.to_datetime(df['exit_date'])
        df['asset'] = df['asset'].map(self.mapping_int_asset)
        return df

    def roundtrips_stats(self, readable: bool = True) -> Union[pd.DataFrame, np.ndarray]:
        """每轮交易统计"""
        rounds = self.roundtrips(False)
        stats = calc_roundtrips_stats(rounds, len(self.mapping_int_asset))
        if not readable:
            return stats
        df = pd.DataFrame.from_records(stats)
        df['asset'] = df['asset'].map(self.mapping_int_asset)
        return df

    def run_bar(self, arr) -> None:
        """同一时点，截面所有资产立即执行

        Parameters
        ----------
        arr
            - date
            - size_type
            - asset
            - size
                nan可用于只更新价格但不交易
            - fill_price
            - last_price
            - commission
            - date_diff

        """
        self.pf.run_bar2(arr)

    def run_bars(self, arrs) -> None:
        """多时点，循序分批执行

        Parameters
        ----------
        arrs
            - date
            - size_type
            - asset
            - size:
                nan可用于只更新价格但不交易
            - fill_price
            - last_price
            - commission
            - date_diff

        """
        for arr in arrs:
            self.pf.run_bar2(arr)


def warmup() -> float:
    """热身

    由于Numba JIT编译要占去不少时间，提前将大部分路径都跑一遍，之后调用就快了"""
    # import os
    # os.environ['NUMBA_DISABLE_JIT'] = '1'
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.width', 1000)

    from lightbt.enums import SizeType
    from lightbt.callbacks import commission_by_qty, commission_by_value
    from lightbt.enums import order_outside_dt
    from lightbt.signals import orders_daily
    from lightbt.utils import groupby

    symbols = [('510300', 1, 1, 0.001, commission_by_qty), ('IF2309', 300, 0.2, 0.0005, commission_by_value), ]
    config = pd.DataFrame.from_records(symbols,
                                       columns=['asset', 'mult', 'margin_ratio', 'commission_ratio', 'commission_fn'])

    df1 = pd.DataFrame({'asset': ['510300', 'IF2309'],
                        'size': [np.nan, -0.5],
                        'fill_price': [4.0, 4000.0],
                        'last_price': [4.0, 4000.0],
                        'date': '2023-08-01',
                        'size_type': SizeType.TargetValuePercent})

    df2 = pd.DataFrame({'asset': ['510300', 'IF2309'],
                        'size': [0.5, 0.5],
                        'fill_price': [4.0, 4000.0],
                        'last_price': [4.0, 4000.0],
                        'date': '2023-08-02',
                        'size_type': SizeType.TargetValuePercent})

    df = pd.concat([df1, df2])
    df['date'] = pd.to_datetime(df['date'])

    tic = time.perf_counter()

    bt = LightBT(init_cash=10000 * 50)
    bt.deposit(10000 * 20)
    bt.withdraw(10000 * 10)

    bt.setup(config)
    # 只能在setup后才能做map
    df['asset'] = df['asset'].map(bt.mapping_asset_int)

    bt.run_bars(groupby(orders_daily(df), by='date', dtype=order_outside_dt))

    bt.positions()
    bt.trades(return_all=True)
    bt.performances(return_all=True)
    bt.reset()

    toc = time.perf_counter()
    return toc - tic
