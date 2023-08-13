import time
from typing import List, Union

import numpy as np
import pandas as pd

from lightbt.enums import order_outside_dt
from lightbt.signals import orders_daily
from lightbt.utils import groupby_np


class LightBT:
    def __init__(self,
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

        self.pf = Portfolio(positions_precision=positions_precision,
                            max_trades=max_trades, max_performances=max_performances)
        # 底层没有资产名字符串，只有纯数字
        self.mapping_asset_int = {}
        self.mapping_int_asset = {}
        self.conf: pd.DataFrame = pd.DataFrame()

    def setup(self, df: pd.DataFrame) -> None:
        """映射资产，配置合约乘数和保证金率

        同名的会进行替换

        Parameters
        ----------
        df: pd.DataFrame
            - asset
            - mult
            - margin_ratio

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

        # 调用底层的批量处理函数
        self.pf.setup(asset, mult, margin_ratio)

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

    def positions(self, convert_asset: bool = True) -> pd.DataFrame:
        """持仓记录"""
        df = pd.DataFrame.from_records(self.pf.positions())
        if convert_asset:
            df['asset'] = df['asset'].map(self.mapping_int_asset)
        return df

    def trades(self, all: bool, convert_asset: bool = True) -> pd.DataFrame:
        """成交记录"""
        df = pd.DataFrame.from_records(self.pf.trades(all))
        df['date'] = pd.to_datetime(df['date'])
        if convert_asset:
            df['asset'] = df['asset'].map(self.mapping_int_asset)
        return df

    def performances(self, all: bool, convert_asset: bool = False) -> pd.DataFrame:
        """持仓记录"""
        df = pd.DataFrame.from_records(self.pf.performances(all))
        df['date'] = pd.to_datetime(df['date'])
        if convert_asset:
            df['asset'] = df['asset'].map(self.mapping_int_asset)
        return df

    def run_bar(self, arr) -> None:
        """同一时点，截面所有资产立即执行"""
        self.pf.run_bar2(arr)

    def run_bars(self, arrs) -> None:
        """多时点，循序分批执行"""
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

    symbols = [('510300', 1, 1), ('IF2309', 300, 0.2), ]
    conf = pd.DataFrame.from_records(symbols, columns=['asset', 'mult', 'margin_ratio'])

    df1 = pd.DataFrame({'asset': ['510300', 'IF2309'],
                        'size': [0.5, -0.5],
                        'fill_price': [4.0, 4000.0],
                        'last_price': [4.0, 4000.0],
                        'date': '2023-08-01',
                        'size_type': SizeType.TargetPercentValue,
                        'commission': 0.0})

    df2 = pd.DataFrame({'asset': ['510300', 'IF2309'],
                        'size': [0, 0],
                        'fill_price': [4.0, 4000.0],
                        'last_price': [4.0, 4000.0],
                        'date': '2023-08-02',
                        'size_type': SizeType.TargetPercentValue,
                        'commission': 0.0})
    df = pd.concat([df1, df2])
    df['date'] = pd.to_datetime(df['date'])

    tic = time.perf_counter()

    bt = LightBT()
    bt.setup(conf)
    bt.deposit(10000 * 50)
    bt.withdraw(10000 * 10)

    bt.run_bars(groupby_np(orders_daily(df, bt.mapping_asset_int), by='date', dtype=order_outside_dt))

    bt.positions()
    bt.trades(all=True)
    bt.performances(all=True)

    toc = time.perf_counter()
    return toc - tic
