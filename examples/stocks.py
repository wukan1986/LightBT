# %%
"""
每月初做多前100支，做空后100支
权重按因子值大小进行分配。分配前因子标准化

计算因子值后，第二天早上交易

由于不支持除权除息，所以价格都为后复权价
"""
import numpy as np
import pandas as pd

from lightbt import LightBT, warmup
from lightbt.callbacks import commission_by_value
from lightbt.enums import order_outside_dt, SizeType
from lightbt.signals import orders_daily
from lightbt.stats import total_equity, pnl_by_asset, pnl_by_assets
from lightbt.utils import Timer, groupby

# %%
# import os
# os.environ['NUMBA_DISABLE_JIT'] = '1'

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.options.plotting.backend = 'plotly'

# %%

_K = 5000  # 多支股票

asset = [f's_{i:04d}' for i in range(_K)]
date = pd.date_range(start='2000-01-01', end='2010-12-31', freq='B')
_N = len(date)  # 10年

CLOSE = np.cumprod(1 + np.random.uniform(-0.1, 0.1, size=(_N, _K)), axis=0) * np.random.randint(10, 100, _K)
CLOSE = pd.DataFrame(CLOSE, index=date, columns=asset)

OPEN = np.cumprod(1 + np.random.uniform(-0.1, 0.1, size=(_N, _K)), axis=0) * np.random.randint(10, 100, _K)

SMA10 = CLOSE.rolling(10).mean()
SMA20 = CLOSE.rolling(20).mean()

# 时间处理，每月第一个交易日调仓，每月第一天可能不是交易日
dt = pd.DataFrame(index=CLOSE.index)
dt['start'] = dt.index
dt['end'] = dt.index
dt = dt.resample('ME').agg({'start': 'first', 'end': 'last'})

# 目标市值
size_type = pd.DataFrame(SizeType.NOP, index=CLOSE.index, columns=CLOSE.columns, dtype=int)
# size_type.loc[dt['start']] = SizeType.TargetValuePercent
# size_type[:] = SizeType.TargetValuePercent

# 因子构建，过滤多头与空头
factor: pd.DataFrame = SMA10 / SMA20 - 1.0  # 因子

# 收盘时产生信号，第二天开盘交易
factor = factor.shift(1)

# 因为之后将按因子值进行权重分配，这里需要提前做做标准化
# 标准化后，前N一定是正数，后N一定是负数
factor = factor.subtract(factor.mean(axis=1), axis=0).div(factor.std(axis=1, ddof=0), axis=0)

top = factor.rank(axis=1, pct=False, ascending=False) <= 100  # 横截面按从大到小排序
bottom = factor.rank(axis=1, pct=False, ascending=True) <= 100  # 横截面按从小到大排序

size = pd.DataFrame(0.0, index=CLOSE.index, columns=CLOSE.columns, dtype=float)
size[top] = factor[top]  # 前N做多
size[bottom] = factor[bottom]  # 后N做空
# 因子加权。因子值大权重大。也可设成等权
size = size.div(size.abs().sum(axis=1), axis=0)

df = pd.DataFrame({
    'OPEN': OPEN.reshape(-1),
    'CLOSE': CLOSE.to_numpy().reshape(-1),
    'size_type': size_type.to_numpy().reshape(-1),
    'size': size.to_numpy().reshape(-1),
}, index=pd.MultiIndex.from_product([date, asset], names=['date', 'asset'])).reset_index()

del OPEN
del CLOSE
del SMA10
del SMA20
del size_type
del size
del top
del bottom
del factor
df.columns = ['date', 'asset', 'OPEN', 'CLOSE', 'size_type', 'size']

# 早上开盘时交易。在集合竞价交易可使用开盘价，也可以使用前5分钟VWAP价
df['fill_price'] = df['OPEN']
# 每天的收盘价或结算价
df['last_price'] = df['CLOSE']

df.to_parquet('tmp.parquet')
df = pd.read_parquet('tmp.parquet')

# %% 热身
print('warmup:', warmup())

# %% 初始化
unit = df['date'].dtype.name[-3:-1]
bt = LightBT(init_cash=0.0,
             positions_precision=1.0,
             max_trades=_N * _K * 2 // 1,  # 反手占两条记录，所以预留2倍空间比较安全
             max_performances=_N * _K,
             unit=unit)
# 入金。必需先入金，否则资金为0无法交易
bt.deposit(10000 * 100)

# %% 配置资产信息
asset = sorted(df['asset'].unique())
config = pd.DataFrame({'asset': asset, 'mult': 1.0, 'margin_ratio': 1.0,
                       'commission_ratio': 0.0005, 'commission_fn': commission_by_value})
with Timer():
    bt.setup(config)

# %% 资产转换，只做一次即可
df['asset'] = df['asset'].map(bt.mapping_asset_int)

# %% 交易
with Timer():
    # 按日更新净值
    bt.run_bars(groupby(orders_daily(df, sort=True), by='date', dtype=order_outside_dt))

# perf = bt.performances(return_all=True)
# s1 = total_equity(perf)['equity']
# print(s1.tail())


# %% 查看最终持仓
positions = bt.positions()
print(positions)
# %% 查看所有交易记录
trades = bt.trades(return_all=True)
print(trades)
trades_stats = bt.trades_stats()
print(trades_stats)
roundtrips = bt.roundtrips()
print(roundtrips)
roundtrips_stats = bt.roundtrips_stats()
print(roundtrips_stats)

# %% 查看绩效
perf = bt.performances(return_all=True)
print(perf)
# %% 总体绩效
equity = total_equity(perf)
print(equity)
equity.plot()

# %% 多个资产的收益曲线
pnls = pnl_by_assets(perf, ['s_0000', 's_0100', 's_0300'], bt.mapping_asset_int, bt.mapping_int_asset)
print(pnls)
pnls.plot()

# %% 单个资产的绩效细节
pnls = pnl_by_asset(perf, 's_0000', df[['date', 'asset', 'CLOSE']], bt.mapping_asset_int, bt.mapping_int_asset)
print(pnls)
pnls.plot()
# %%
pd.options.plotting.backend = 'matplotlib'
pnls[['PnL', 'CLOSE']].plot(secondary_y='CLOSE')
# %%
print(df)
# %%
