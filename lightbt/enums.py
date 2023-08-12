from typing import NamedTuple

import numpy as np


# 此枚举定义参考于vectorbt。多加了与保证金有关类别
class SizeTypeT(NamedTuple):
    # 下单数量和方向
    Amount: int = 0
    # 下单市值和方向
    Value: int = 1
    # 下单保证金和方向
    Margin: int = 2
    # 正数使用现金比例，负数卖出持仓比例
    Percent: int = 3
    # 目标数量和方向
    TargetAmount: int = 4
    # 目标市值和方向
    TargetValue: int = 5
    # 目标保证金和方向
    TargetMargin: int = 6
    # 目标市值占比
    TargetPercentValue: int = 7
    # 目标保证金点比
    TargetPercentMargin: int = 8


SizeType = SizeTypeT()

# 绩效统计。为减少内存，使用float32
performance_dt = np.dtype([
    ('date', np.int64),
    ('asset', np.uint32),
    ('value', np.float32),
    ('cash', np.float32),
    ('margin', np.float32),
    ('upnl', np.float32),
    ('pnls', np.float32),
    ('commissions', np.float32),
], align=True)

# 成交记录。为减少内存，使用float32
trade_dt = np.dtype([
    ('date', np.int64),
    ('asset', np.uint32),
    ('is_buy', np.bool_),
    ('is_open', np.bool_),  # 是否开平。开一定是开，平有可能反手
    ('fill_price', np.float32),
    ('qty', np.float32),  # 当前交易数量
    ('amount', np.float32),  # 持仓量和方向
    ('margin', np.float32),
    ('commission', np.float32),
    ('upnl', np.float32),  # 持仓盈亏
    ('pnl', np.float32),  # 平仓盈亏(未减手续费)
    ('cash_flow', np.float32),  # 现金流=平仓盈亏-保证金-手续费
    ('cash', np.float32),  # 现金
], align=True)

# 持仓记录。中间计算用字段与计算字段类型一致，而展示用字段减少内存
position_dt = np.dtype([
    ('asset', np.uint32),
    # 用于相关下单数量计算
    ('mult', float),
    ('margin_ratio', float),
    ('amount', float),
    # 展示用字段
    ('value', np.float32),
    ('open_value', np.float32),
    ('avg_price', np.float32),
    ('last_price', np.float32),
    ('margin', np.float32),
    ('upnl', np.float32),
    ('pnls', np.float32),
    ('commissions', np.float32),
], align=True)

# 外部下单指令，用于将用户的指令转成内部指令
order_outside_dt = np.dtype([
    ('date', np.int64),
    ('size_type', int),
    ('asset', int),
    ('size', float),
    ('fill_price', float),
    ('last_price', float),
    ('commission', float),
    ('date_diff', bool),  # 标记换日，会触发绩效更新
    ('time_diff', bool),  # 标记同时间截面，用于时序分组
], align=True)

# 内部下单指令。用于将上层的目标持仓等信息转换成实际下单指令
order_inside_dt = np.dtype([
    ('asset', int),
    ('is_buy', bool),
    ('is_open', bool),
    ('fill_price', float),
    ('qty', float),
    ('amount', float),
    ('size', float),
    ('commission', float),
], align=True)
