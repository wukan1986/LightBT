# %%
"""
每月初再平衡
将因子分成10组，每组进行分别统计，然后画分组曲线

Notes
=====
只是为了看因子的区分能力，用对数收益累加的速度更快。
这里只是为了演示可以实现分层功能

"""
# import os
#
# os.environ['NUMBA_DISABLE_JIT'] = '1'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lightbt import LightBT, warmup
from lightbt.callbacks import commission_by_value
from lightbt.enums import SizeType, order_outside_dt
from lightbt.signals import orders_daily
from lightbt.stats import total_equity
from lightbt.utils import Timer, groupby

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
# pd.options.plotting.backend = 'plotly'

# %%

_K = 5000  # 多支股票

asset = [f's_{i:04d}' for i in range(_K)]
date = pd.date_range(start='2000-01-01', end='2010-12-31', freq='B')
_N = len(date)  # 10年

CLOSE = np.cumprod(1 + np.random.uniform(-0.1, 0.1, size=(_N, _K)), axis=0) * np.random.randint(10, 100, _K)
CLOSE = pd.DataFrame(CLOSE, index=date, columns=asset)

SMA10 = CLOSE.rolling(10).mean()
SMA20 = CLOSE.rolling(20).mean()

# 时间处理，每月第一个交易日调仓，每月第一天可能不是交易日
dt = pd.DataFrame(index=CLOSE.index)
dt['start'] = dt.index
dt['end'] = dt.index
dt = dt.resample('ME').agg({'start': 'first', 'end': 'last'})

# 目标市值
size_type = pd.DataFrame(SizeType.NOP, index=CLOSE.index, columns=CLOSE.columns, dtype=int)
size_type.loc[dt['start']] = SizeType.TargetValueScale

# 因子构建
factor: pd.DataFrame = SMA10 / SMA20 - 1.0  # 因子

# 收盘时产生信号，第二天开盘交易
factor = factor.shift(1)

# 分组
factor = factor.loc[dt['start']].stack()
factor.index.names = ['date', 'asset']
quantile: pd.DataFrame = factor.groupby(by=['date'], group_keys=False).apply(
    lambda x: pd.qcut(x, 10, labels=False, duplicates='drop')).unstack()
quantile, _ = quantile.align(CLOSE, fill_value=-1)

size = pd.DataFrame(0.0, index=CLOSE.index, columns=CLOSE.columns, dtype=float)

df = pd.DataFrame({
    'CLOSE': CLOSE.to_numpy().reshape(-1),
    'size_type': size_type.to_numpy().reshape(-1),
    'size': size.to_numpy().reshape(-1),
    'quantile': quantile.to_numpy().reshape(-1),
}, index=pd.MultiIndex.from_product([date, asset], names=['date', 'asset'])).reset_index()

del CLOSE
del SMA10
del SMA20
del size_type
del size
del factor
del quantile
df.columns = ['date', 'asset', 'CLOSE', 'size_type', 'size', 'quantile']

df['fill_price'] = df['CLOSE']
df['last_price'] = df['fill_price']

# %% 热身
print('warmup:', warmup())

# %% 初始化
bt = LightBT(init_cash=10000 * 100,  # 初始资金
             positions_precision=1.0,
             max_trades=_N * _K * 2 // 1,  # 反手占两条记录，所以预留2倍空间比较安全
             max_performances=_N * _K)

asset = df['asset'].unique()
config = pd.DataFrame({'asset': asset, 'mult': 1.0, 'margin_ratio': 1.0,
                       'commission_ratio': 0.0005, 'commission_fn': commission_by_value})

# %% 配置资产信息
with Timer():
    bt.setup(config)

# %% 资产转换，只做一次即可
df['asset'] = df['asset'].map(bt.mapping_asset_int)

# %% 交易
equities = pd.DataFrame()
# 如果能并行就好了
for i in range(10):
    print(i, '\t', end='')
    df['size'] = (df['quantile'] == i).astype(float)

    bt.reset()  # 必需在初始化时设置资金，之后的入金在reset后不生效
    with Timer():
        # 按周更新净值
        bt.run_bars(groupby(orders_daily(df, sort=True), by='date', dtype=order_outside_dt))

    perf = bt.performances(return_all=True)
    equities[i] = total_equity(perf)['equity']

# %%
equities.plot()
plt.show()
