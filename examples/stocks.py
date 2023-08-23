# %%

# os.environ['NUMBA_DISABLE_JIT'] = '1'

import numpy as np
import pandas as pd

from lightbt import LightBT, warmup
from lightbt.callbacks import commission_by_value
from lightbt.enums import SizeType, order_outside_dt
from lightbt.signals import orders_daily
from lightbt.stats import pnl_by_assets, total_equity, pnl_by_asset
from lightbt.utils import Timer, groupby

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.options.plotting.backend = 'plotly'

# %%

_K = 5000  # 5000支股票

asset = [f's_{i:04d}' for i in range(_K)]
date = pd.date_range(start='2000-01-01', end='2010-12-31', freq='B')
_N = len(date)  # 10年

config = pd.DataFrame({'asset': asset, 'mult': 1.0, 'margin_ratio': 1.0,
                       'commission_ratio': 0.0005, 'commission_fn': commission_by_value})

CLOSE = np.cumprod(1 + np.random.uniform(-0.1, 0.1, size=(_N, _K)), axis=0) * np.random.randint(10, 100, _K)
CLOSE = pd.DataFrame(CLOSE, index=date, columns=asset)

SMA10 = CLOSE.rolling(10).mean()
SMA20 = CLOSE.rolling(20).mean()

# 时间处理，每月第一个交易日调仓，每月第一天可能不是交易日
dt = pd.DataFrame(index=CLOSE.index)
dt['start'] = dt.index
dt['end'] = dt.index
dt = dt.resample('M').agg({'start': 'first', 'end': 'last'})

# 目标市值
size_type = pd.DataFrame(SizeType.NOP, index=CLOSE.index, columns=CLOSE.columns, dtype=int)
size_type.loc[dt['start']] = SizeType.TargetPercentValue
# size_type[:] = SizeType.TargetPercentValue

# 因子构建，过滤多头与空头
factor = SMA10 / SMA20 - 1.0  # 因子
top = factor.rank(axis=1, pct=False, ascending=False) <= 100  # 横截面按从大到小排序
bottom = factor.rank(axis=1, pct=False, ascending=True) <= 100  # 横截面按从小到大排序

size = pd.DataFrame(0.0, index=CLOSE.index, columns=CLOSE.columns, dtype=float)
size[top] = factor[top]  # 前N做多
size[bottom] = factor[bottom]  # 后N做空
# 因子加权。因子值大权重大。也可设成等权
size = size.div(size.abs().sum(axis=1), axis=0)

df = pd.DataFrame({
    'CLOSE': CLOSE.to_numpy().reshape(-1),
    'SMA10': SMA10.to_numpy().reshape(-1),
    'SMA20': SMA20.to_numpy().reshape(-1),
    'size_type': size_type.to_numpy().reshape(-1),
    'size': size.to_numpy().reshape(-1),
}, index=pd.MultiIndex.from_product([date, asset], names=['date', 'asset'])).reset_index()

del CLOSE
del SMA10
del SMA20
del size_type
del size
del top
del bottom
df.columns = ['date', 'asset', 'CLOSE', 'SMA10', 'SMA20', 'size_type', 'size']

df['fill_price'] = df['CLOSE']
df['last_price'] = df['fill_price']

# %% 热身
with Timer():
    print('warmup:', warmup())

# %% 初始化
bt = LightBT(init_cash=0.0,
             positions_precision=1.0,
             max_trades=_N * _K * 2 // 1,  # 反手占两条记录，所以预留2倍空间比较安全
             max_performances=_N * _K)
# 入金。必需先入金，否则资金为0无法交易
bt.deposit(10000 * 100)

# %% 配置资产信息
with Timer():
    bt.setup(config)

# %% 资产转换，只做一次即可
df['asset'] = df['asset'].map(bt.mapping_asset_int)

# %% 交易
with Timer():
    bt.run_bars(groupby(orders_daily(df), by='date', dtype=order_outside_dt))

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
