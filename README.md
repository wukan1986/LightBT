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

from lightbt import LightBT, warmup
from lightbt.stats import total_equity
from lightbt.utils import Timer

# 省略代码......

# %% 热身
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
    bt.run_bars(groupby(orders_daily(df, sort=True), by='date', dtype=order_outside_dt))

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

```

## 注意

`run_bars`的输入是迭代类型，它处理数据是一批一批的，同一批中处理优先级如下：

1. 平仓优先。先平后开
2. 时间优先。后面订单可能资金不足而少报

所以。每天早上提供一张目标仓位清单后，内部分成两组，先下平仓单，再下开仓单。

- 本回测系统由于会立即成交，所以平仓后资金立即释放，可以立即开仓
- 但实盘中平仓要等成交释放资金后才能再开仓，如果资金非常多，一起开平也可以

`groupby`是用来分批的工具，可以使用多个参数进行多重分组，
如参数为`groupby(by=['date'])`时就是一天一个交易清单，如果需要收盘开仓，早盘平仓 ，可以`date`中的时间`精确到小时`做成每天两批

同一批中，主要参数需要完全一样，系统只取每批的最后一组。例如：`size_type`在同一批中不同概念冲突了。

### 结果稳定性

1. 部分工具的选出前10等功能，可能由于前20个值都一样，这时一定要考察更多的指标来确定顺序，比如多考察股票名。否则结果可能每次都不一样。
2. `config`函输入`asset`也需要提前排序
3. `groupby`前也要排序

## 输入格式

1. date: int64
    - 时间日期。需在外部提前转成数字。可用`astype(np.int64)`或`to_records(dtype)`等方法来实现
2. size_type: int
    - 数量类型。参考`lightbt.enums.SizeType`
3. asset: int
    - 资产ID。由`LightBT.setup`执行结果所确定。可通过`LightBT.asset_str2int`和`LightBT.asset_int2str`进行相互转换
4. size: float
    - 数量。具体含义需根据`size_type`决定。`nan`是一特殊值。用来表示当前一行不交易。在只更新最新价的需求中将频繁用到。
5. fill_price: float
    - 成交价。成交价不等于最新价也不等于收盘价。可以用成交均价等一些有意义的价格进行代替。
6. last_price: float
    - 最新价。可用收盘价、结算价等代替。它影响持仓的浮动盈亏。所以在对绩效快照前一定要更新一次
7. date_diff: bool
    - 是否换日。在换日的最后时刻需要更新最新价和记录绩效

## 配置格式

通过`LightBT.setup`进行设置

1. asset: str
    - 资产名。内部将使用对应的int进行各项处理
2. mult: float
    - 合约乘数。股票的合约乘数为1.0
3. margin_ratio: float
    - 保证金率。股票的保证金率为1.0
4. commission_ratio: float
    - 手续费率参数。具体含义参考`commission_fn`
5. commission_fn:
    - 手续费处理函数

## 调试

```python
import os

os.environ['NUMBA_DISABLE_JIT'] = '1'
```

`numba`的JIT模式下是无法直接调试的，编译也花时间，可以先添加环境变量`NUMBA_DISABLE_JIT=1`，禁止JIT模式。

数据量较小时，禁用JIT模式反而速度更快。

## 二次开发

```commandline
git --clone https://github.com/wukan1986/LightBT.git
cd LightBT
pip install -e .
```