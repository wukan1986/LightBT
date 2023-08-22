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

# 等权
size_type = pd.DataFrame(SizeType.NOP, index=CLOSE.index, columns=CLOSE.columns, dtype=int)
size_type.loc[dt['start']] = SizeType.TargetPercentValue
# size_type[:] = SizeType.TargetPercentValue

size = SMA10 / SMA20  # 因子
rank = size.rank(axis=1, pct=False, ascending=False)  # 横截面按从大到小排序。用户需
# 保留前N支股票
size[rank > 1000] = np.nan
# 因子加权。因子值大权重大。也可设成等权
size = size.div(size.sum(axis=1), axis=0)

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
del rank
df.columns = ['date', 'asset', 'CLOSE', 'SMA10', 'SMA20', 'size_type', 'size']

# df['size'] = ((df['SMA10'] > df['SMA20']) * 2 - 1) / _K  # 多空双向
# df['size'] = ((df['SMA10'] > df['SMA20']) * 1) / _K  # 只多头
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
trades = bt.trades(all=True)
print(trades)
# %% 查看绩效
perf = bt.performances(all=True)
print(perf)
# %% 总体绩效
equity = total_equity(perf)
print(equity)
equity.plot()

# %% 多个资产的收益曲线
pnls = pnl_by_assets(perf, bt.asset_str2int(['s_0000', 's_0100', 's_0300']))
pnls.columns = bt.asset_int2str(pnls.columns)
print(pnls)
pnls.plot()

# %% 单个资产的绩效细节
tmp = df[['date', 'asset', 'CLOSE']].set_index(['date', 'asset'])
pnls = pnl_by_asset(perf, bt.asset_str2int(['s_0000']), tmp)
print(pnls)
pnls.plot()
# %%
pd.options.plotting.backend = 'matplotlib'
pnls[['PnL', 'CLOSE']].plot(secondary_y='CLOSE')
# %%
print(df)
# %%
