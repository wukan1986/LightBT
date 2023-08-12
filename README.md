# LightBT
轻量级回测工具

## 注意
每种工具都有其适合的应用场景。遇到需要快速评测大批量参数或因子的场景，我们只要相对准确。所以使用对数收益累加更合适

## 现状
1. 收益率乘权重的向量计算极快，但在连续做空的场景下，收益不正确，仅仅是因为每天收益率变化不大，所以误差很难察觉。
2. 大部分回测库策略与绩效统计结合过于紧密，达不到策略与平台分离的设想
3. 大部分回测库都很慢，如：`zipline`、`rqalpha`、`backtrader`
4. 回测快的库，底层一般是用`C++`、`Rust`等语言实现。跨语言部署和二次开发对量化研究员而言比较困难
5. `vectorbt`计算准确，回测也快。但不支持保证金无法直接用在期货市场，输入宽表比较占内存
6. `bt`策略树设计很好，但也不支持保证金概念

## 目标
1. 正确处理保证金和做空
2. 架构简单易扩展
3. 回测速度快

## 技术选型
1. `C++/Cython`开发，模仿`TA-Lib`的技术方案，`Cython`版库部署麻烦，开发也复杂
2. `Rust`开发，模仿`polars`的技术方案，使用`pyo3`进行跨语言调用，但`Rust`入门困难
3. `Numba`支持`JIT`，安装部署方便，易于调试和二次开发

## 三层结构
1. 开仓成交回报产生的持仓导致持仓盈亏、平仓成交回报产生平仓盈亏。将两种盈亏累计便成盈亏曲线
    - 已经可以统计盈亏、胜率、最大回撤等指标
2. 叠加初始资金即可构成资金曲线和净值曲线，可计算收益率、最大回撤率等指标
    - 没有考虑资金不足、是否能成交等情况。已经是简化回测绩效统计工具。仅能按手数进行交易
3. 初始资金和保证金率决定了可开手数
    - 关注交易细节，考虑资金不足等情况。可以按比例进行资金分配下单

## 工作原理
1. 使用对象来维护每种资产的交易信息。其中的成员变量全是最新值
2. 根据时间的推进和交易指令，更新对应对象
3. 指定时间间隔获取对象成员变量的快照，将所有快照连起来便成总体绩效
    - 月频交易，但日频取快照
    - 周频交易，周频取快照
    - 分钟交易，日频取快照

## 安装
```commandline
pip install lightbt -U
```
## 使用
以下是代码片段。完整代码请参考[stocks.py](examples/stocks.py)
```python
# %%
import numpy as np
import pandas as pd

from lightbt import LightBT, warmup
from lightbt.enums import SizeType
from lightbt.stats import pnl_by_assets, total_equity, pnl_by_asset
from lightbt.utils import Timer

# 省略代码......

# %% 热身
with Timer():
    print('warmup:', warmup())

# %% 初始化
bt = LightBT(max_trades=_N * _K, max_performances=_N * _K)
# 入金。必需先入金，否则资金为0无法交易
bt.deposit(10000 * 100)

# %% 配置资产信息
with Timer():
    bt.setup(conf)

# %% 交易
with Timer():
    bt.run_bar(*orders_to_array(orders_daily(df, bt.mapping_asset_int)))

# %% 查看最终持仓
positions = bt.positions()
print(positions)
# %% 查看所有交易记录
trades = bt.trades()
print(trades)
# %% 查看绩效
perf = bt.performances()
print(perf)
# %% 总体绩效
equity = total_equity(perf)
print(equity)
equity.plot()

```

## 二次开发
```commandline
git --clone https://github.com/wukan1986/LightBT.git
cd LightBT
pip install -e .
```