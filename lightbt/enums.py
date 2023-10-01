from typing import NamedTuple

import numpy as np


# 此枚举定义参考于vectorbt。多加了与保证金有关类别
class SizeTypeT(NamedTuple):
    # 空操作指令。通过此值比size全nan能减少代码执行
    NOP: int = 0
    # 下单数量和方向
    Amount: int = 1
    # 下单市值和方向
    Value: int = 2
    # 下单保证金和方向
    Margin: int = 3
    # 正数使用现金比例，负数卖出持仓比例
    Percent: int = 4
    # 目标数量和方向
    TargetAmount: int = 5
    # 目标市值和方向
    TargetValue: int = 6
    # 目标保证金和方向
    TargetMargin: int = 7
    # 目标市值百分比。size绝对值之和范围[0,1]
    TargetPercentValue: int = 8
    # 目标市值比例。size值可能为1.5:1:-1等
    TargetScaleValue: int = 9
    # 目标保证金百分比。size绝对值之和范围[0,1]
    TargetPercentMargin: int = 10
    # 目标保证金比例。size值可能为1.5:1:-1等
    TargetScaleMargin: int = 11


SizeType = SizeTypeT()

# 绩效统计。为减少内存，使用float32
performance_dt = np.dtype([
    ('date', np.int64),  # 日期时间
    ('asset', np.uint32),  # 资产ID
    ('amount', np.float32),  # 净持仓量。负数表示空头
    ('value', np.float32),  # 净持仓市值。负数表示空头
    ('cash', np.float32),  # 当前批量单执行后所剩余现金
    ('margin', np.float32),  # 占用保证金
    ('upnl', np.float32),  # 持仓盈亏
    ('cum_pnl', np.float32),  # 累计平仓盈亏（未扣除手续费）
    ('cum_commission', np.float32),  # 累计手续费
], align=True)

# 成交记录。为减少内存，使用float32
trade_dt = np.dtype([
    ('date', np.int64),  # 日期时间
    ('asset', np.uint32),  # 资产ID
    ('is_buy', np.bool_),  # 是否买入。
    ('is_open', np.bool_),  # 是否开平。开一定是开，平有可能含反手。反手也可以拆成两单
    ('fill_price', np.float32),  # 成交价
    ('qty', np.float32),  # 当前交易数量
    ('amount', np.float32),  # 持仓量和方向
    ('margin', np.float32),  # 保证金
    ('commission', np.float32),  # 手续费
    ('upnl', np.float32),  # 持仓盈亏
    ('pnl', np.float32),  # 平仓盈亏(未扣除手续费)
    ('cash_flow', np.float32),  # 现金流=平仓盈亏-保证金-手续费
    ('cash', np.float32),  # 现金
], align=True)

# 持仓记录。中间计算用字段与计算字段类型一致，而展示用字段减少内存
position_dt = np.dtype([
    ('asset', np.uint32),  # 资产ID
    # 用于相关下单数量计算
    ('mult', float),  # 合约乘数
    ('margin_ratio', float),  # 保证金率
    ('amount', float),  # 净持仓数量
    # 展示用字段
    ('value', np.float32),  # 市值
    ('open_value', np.float32),  # 开仓市值
    ('avg_price', np.float32),  # 平均价
    ('last_price', np.float32),  # 最新价
    ('margin', np.float32),  # 保证金
    ('upnl', np.float32),  # 持仓盈亏
    ('cum_pnl', np.float32),  # 累计平仓盈亏（未扣除手续费）
    ('cum_commission', np.float32),  # 累计手续费
], align=True)

# 外部下单指令，用于将用户的指令转成内部指令
order_outside_dt = np.dtype([
    ('date', np.int64),  # 日期时间
    ('size_type', int),  # size字段类型
    ('asset', int),  # 资产ID
    ('size', float),  # nan时表示此行不参与交易。可用于有持仓但不交易的资产更新最新价
    ('fill_price', float),  # 成交价
    ('last_price', float),  # 最新价
    ('date_diff', bool),  # 标记换日，会触发绩效更新
], align=True)

# 内部下单指令。用于将上层的目标持仓等信息转换成实际下单指令
order_inside_dt = np.dtype([
    ('asset', int),  # 资产ID
    ('is_buy', bool),  # 是否买入
    ('is_open', bool),  # 是否开仓
    ('fill_price', float),  # 成交价
    ('qty', float),  # 成交数量
], align=True)

# 成交统计。条目数一般等于资产数量
trades_stats_dt = np.dtype([
    ('asset', np.uint32),  # 资产ID
    ('start', np.int64),  # 第一条记录时间
    ('end', np.int64),  # 最后一条记录时间
    ('period', np.int64),  # 期
    ('total_count', np.uint32),  # 总条数
    ('buy_count', np.uint32),  # 买入条数
    ('sell_count', np.uint32),  # 卖出条数
    ('min_qty', np.float32),  # 最小交易量
    ('max_qty', np.float32),  # 最大交易量
    ('avg_qty', np.float32),  # 平均交易量
    ('avg_buy_qty', np.float32),  # 平均买入量
    ('avg_sell_qty', np.float32),  # 平均卖出量
    ('avg_buy_price', np.float32),  # 平均买入价
    ('avg_sell_price', np.float32),  # 平均卖出价
    ('total_commission', np.float32),  # 总手续费
    ('min_commission', np.float32),  # 最小手续费
    ('max_commission', np.float32),  # 最大手续费
    ('avg_commission', np.float32),  # 平均手续费
    ('avg_buy_commission', np.float32),  # 平均买入手续费
    ('avg_sell_commission', np.float32),  # 平均卖出手续费
], align=True)

# 交易每轮。由入场和出场组成
roundtrip_dt = np.dtype([
    ('asset', np.uint32),  # 资产ID
    ('is_long', np.bool_),  # 是否多头
    ('is_close', np.bool_),  # 是否已平
    ('qty', np.float32),  # 数量
    ('entry_date', np.int64),  # 入场时间
    ('entry_price', np.float32),  # 入场价
    ('entry_commission', np.float32),  # 入场手续费
    ('exit_date', np.int64),  # 出场时间
    ('exit_price', np.float32),  # 出场价
    ('exit_commission', np.float32),  # 出场手续费
    ('pnl', np.float32),  # 本轮平仓盈亏
    ('pnl_com', np.float32),  # 本轮平仓盈亏（已减手续费）
], align=True)

# 每轮统计
roundtrip_stats_dt = np.dtype([
    ('asset', np.uint32),  # 资产ID
    ('total_count', np.uint32),  # 总条数
    ('long_count', np.uint32),  # 多头条数
    ('short_count', np.uint32),  # 空头条数
    ('winning_count', np.uint32),  # 盈利条数
    ('losing_count', np.uint32),  # 亏损条数
    ('win_rate', np.float32),  # 胜率
], align=True)
