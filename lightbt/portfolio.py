import os

import numpy as np
from numba import typeof, objmode, types, prange
from numba.experimental import jitclass
from numba.typed.typedlist import List

from lightbt.enums import SizeType, trade_dt, position_dt, performance_dt, order_inside_dt
from lightbt.position import Position


class Portfolio:
    _positions_precision: float
    _cash: float

    _idx_curr_trade: int
    _idx_curr_performance: int
    _idx_last_trade: int
    _idx_last_performance: int
    _max_trades: int
    _max_performances: int

    def __init__(self,
                 positions_precision: float = 1.0,
                 max_trades: int = 1024,
                 max_performances: int = 1024) -> None:
        """初始化

        Parameters
        ----------
        positions_precision: float
            持仓精度
                - 1.0 表示整数量
                - 0.01 表示持仓可以精确到0.01，用于数字货币等场景
                - 0.000001 精度高，相当于对持仓精度不做限制
        max_trades: int
            记录成交的缓存大小。空间不足时将丢弃
        max_performances: int
            记录绩效的缓存大小。空间不足时将丢弃
        """
        # https://github.com/numba/numba/issues/8733
        list_tmp = List()
        list_tmp.append(Position(0))
        list_tmp.clear()
        self._positions = list_tmp

        self._trade_records = np.empty(max_trades, dtype=trade_dt)
        self._position_records = np.empty(1, dtype=position_dt)
        self._performance_records = np.empty(max_performances, dtype=performance_dt)

        self._positions_precision = positions_precision

        self._max_trades = max_trades
        self._max_performances = max_performances

        self.reset()

    def reset(self):
        self._cash = 0.0

        self._idx_curr_trade = 0
        self._idx_curr_performance = 0
        self._idx_last_trade = 0
        self._idx_last_performance = 0

        for p in self._positions:
            p.reset()

    @property
    def Cash(self) -> float:
        """现金"""
        return self._cash

    @property
    def Value(self) -> float:
        """持仓市值。空头为负数"""
        return np.sum(np.array([pos.Value for pos in self._positions]))

    @property
    def Margin(self) -> float:
        """保证金占用"""
        return np.sum(np.array([pos.Margin for pos in self._positions]))

    @property
    def UPnL(self) -> float:
        """未平仓盈亏"""
        return np.sum(np.array([pos.UPnL for pos in self._positions]))

    @property
    def Equity(self) -> float:
        """权益=子权益+现金"""
        return np.sum(np.array([pos.Equity for pos in self._positions])) + self._cash

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
        self._cash += cash
        return self._cash

    def withdraw(self, cash: float) -> float:
        """出金"""
        self._cash -= cash
        return self._cash

    def set_commission_fn(self, asset: int, func) -> None:
        """设置手续费函数"""
        pos: Position = self._positions[asset]
        pos.set_commission_fn(func)

    def setup(self, asset: np.ndarray,
              mult: np.ndarray, margin_ratio: np.ndarray,
              commission_ratio: np.ndarray) -> None:
        """批量配置各品种的参数。

        1. 有部分品种以前是一种配置，后来又换了配置. 如黄金
        2. 新添品种

        Parameters
        ----------
        asset: np.ndarray
            资产ID
        mult: np.ndarray
            合约乘数
        margin_ratio: np.ndarray
            保证金率
        commission_ratio: np.ndarray
            手续费率

        """
        # 指定长度进行初始化
        count = len(mult)
        while len(self._positions) < count:
            self._positions.append(Position(len(self._positions)))

        # 创建记录体，用于最终显示持仓
        self._position_records = np.empty(len(self._positions), dtype=position_dt)

        for i in prange(count):
            #
            self._positions[asset[i]].setup(mult[i], margin_ratio[i], commission_ratio[i])

    def _fill_trade_record(self,
                           date: np.int64, asset: int,
                           is_buy: bool, is_open: bool, fill_price: float, qty: float) -> None:
        """遇到有效成交时自动更新，所以内容直接取即可"""
        if self._idx_curr_trade >= self._max_trades:
            return
        rec = self._trade_records[self._idx_curr_trade]

        self._positions[asset].to_record_trade(rec, date, is_buy, is_open, fill_price, qty, self._cash)

        self._idx_curr_trade += 1

    def _fill_position_records(self, detail: bool) -> None:
        """更新持仓记录"""
        for i, pos in enumerate(self._positions):
            rec = self._position_records[i]
            pos.to_record_position(rec, detail)

    def update_last_price(self, asset: np.ndarray, last_price: np.ndarray) -> None:
        """更新结算价"""
        for i in prange(len(asset)):
            pos: Position = self._positions[asset[i]]
            if pos.Amount == 0:
                # 只对有持仓的更新最新价即可
                continue
            pos.update_last_price(last_price[i])

    def update_performances(self, date: np.int64) -> None:
        """更新绩效"""
        cash: float = self._cash
        # 上次的位置
        self._idx_last_performance = self._idx_curr_performance
        for i, pos in enumerate(self._positions):
            if self._idx_curr_performance >= self._max_performances:
                return

            rec = self._performance_records[self._idx_curr_performance]
            pos.to_record_performance(rec, date, cash)

            self._idx_curr_performance += 1

    def update(self, date: np.int64, asset: np.ndarray, last_price: np.ndarray, do_settlement: bool) -> None:
        """更新价格。记录绩效

        Parameters
        ----------
        date: np.int64
            日期。可以转成pandas时间
        asset: np.ndarray
            需更新行情的资产
        last_price: np.ndarray
            最新价。日频可以是结算价
        do_settlement: bool
            是否结算

        """
        self.update_last_price(asset, last_price)
        if do_settlement:
            self.settlement()
        self.update_performances(date)

    def settlement(self) -> None:
        """结算"""
        for i, pos in enumerate(self._positions):
            self._cash += pos.settlement()

    def performances(self, return_all: bool) -> np.ndarray:
        """绩效记录"""
        if return_all:
            return self._performance_records[:self._idx_curr_performance]
        else:
            return self._performance_records[self._idx_last_performance:self._idx_curr_performance]

    def trades(self, return_all: bool) -> np.ndarray:
        """很多变量只记录了瞬时值，当需要时序值时，通过此函数记录下来备用"""
        if return_all:
            return self._trade_records[:self._idx_curr_trade]
        else:
            return self._trade_records[self._idx_last_trade:self._idx_curr_trade]

    def positions(self) -> np.ndarray:
        """最新持仓记录"""
        self._fill_position_records(True)
        return self._position_records

    def order(self, date: np.int64, asset: int, is_buy: bool, is_open: bool, fill_price: float, qty: float) -> bool:
        """下单

        Parameters
        ----------
        date: int
        asset: int
        is_buy: bool
        is_open: bool
            是否开仓。反手暂时归属于平仓。
        fill_price: float
        qty: float

        Returns
        -------
        bool

        """
        # convert_size时已经过滤了不合法的数量，所以这里注释了

        # if qty <= 0.0:
        #     # 数量不合法，返回。可用于刷新行情但不产生交易记录
        #     return False

        pos: Position = self._positions[asset]
        # 成交价所对应的市值和手续费
        value = pos.calc_value(fill_price, qty)
        commission = pos.calc_commission(is_buy, is_open, value, qty)
        if is_open:
            # 可开手数检查
            if not pos.openable(self._cash, value, commission):
                return False
        else:
            # TODO: 可能有反手情况。这个以后再处理
            pass

        pos.fill(is_buy, is_open, value, fill_price, qty, commission)
        self._cash += pos.CashFlow

        self._fill_trade_record(date, asset, is_buy, is_open, fill_price, qty)

        return True

    def convert_size(self, size_type: int, asset: np.ndarray, size: np.ndarray, fill_price: np.ndarray) -> np.ndarray:
        """交易数量转换

        Parameters
        ----------
        size_type
        asset
        size: float
            nan时表示不交易
        fill_price

        """
        self._fill_position_records(False)
        # asset不能出现重复
        _rs: np.ndarray = self._position_records[asset]
        margin_ratio: np.ndarray = _rs['margin_ratio']
        amount: np.ndarray = _rs['amount']
        mult: np.ndarray = _rs['mult']

        # 所有的TargetXxx类型，如果出现size=0, 直接处理更精确
        is_target: bool = size_type >= SizeType.TargetAmount
        is_zero: np.ndarray = size == 0

        # 归一时做分母。但必需是没有上游改动
        _equity: float = 0.0
        equity: float = self.Equity
        cash: float = self._cash
        size_abs_sum: float = np.nansum(np.abs(size))
        if size_abs_sum == 0:
            # 全0表示清仓
            if size_type > SizeType.TargetAmount:
                size_type = SizeType.TargetAmount

        # 目标保证金比率相关计算。最后转成目标市值
        if size_type >= SizeType.TargetMargin:
            if size_type == SizeType.TargetMarginScale:
                size /= size_abs_sum  # 归一。最终size和是1
                _equity = equity
                size *= _equity
            if size_type == SizeType.TargetMarginPercent:
                _equity = equity * size_abs_sum
                size *= _equity
            if size_type == SizeType.TargetMargin:
                pass

            # 统一保证金
            size /= margin_ratio
            size_type = SizeType.TargetValue

        # 目标市值比率相关计算。最后转成目标市值
        if size_type > SizeType.TargetValue:
            if size_type == SizeType.TargetValueScale:
                size /= size_abs_sum
                _equity = equity
            if size_type == SizeType.TargetValuePercent:
                _equity = equity * size_abs_sum

            # 特殊处理，通过保证金率还原市值占比
            _ratio: float = np.nansum((np.abs(size) * margin_ratio))
            size *= _equity
            if _ratio != 0:
                size /= _ratio
            size_type = SizeType.TargetValue

        # 使用次数最多的类型
        if size_type == SizeType.TargetValue:
            # 前后市值之差
            size -= (fill_price * mult * amount)
            size_type = SizeType.Value
        if size_type == SizeType.TargetAmount:
            # 前后Amout差值
            size -= amount
            size_type = SizeType.Amount
        if size_type == SizeType.Percent:
            # 买入开仓，用现金转市值
            # 卖出平仓，持仓市值的百分比
            # TODO: 由于无法表示卖出开仓。所以只能用在股票市场
            size *= np.where(size >= 0, cash / (fill_price * mult * margin_ratio), amount)
            size_type = SizeType.Amount
        if size_type == SizeType.Margin:
            # 将保证金转换成市值
            size /= margin_ratio
            size_type = SizeType.Value
        if size_type == SizeType.Value:
            # 将市值转成手数
            size /= (fill_price * mult)
            size_type = SizeType.Amount
        if size_type == SizeType.Amount:
            pass

        if is_target:
            # 直接取反，回避了前期各种计算导致的误差
            size[is_zero] = -amount[is_zero]

        is_open: np.ndarray = np.sign(amount) * np.sign(size)
        is_open = np.where(is_open == 0, amount == 0, is_open > 0)

        amount_abs = np.abs(amount)
        size_abs = np.abs(size)

        # 创建一个原始订单表，其中存在反手单
        orders = np.empty(len(asset), dtype=order_inside_dt)
        orders['asset'][:] = asset
        orders['fill_price'][:] = fill_price
        orders['qty'][:] = size_abs
        orders['is_buy'][:] = size >= 0
        orders['is_open'][:] = is_open

        # 是否有反手单
        is_reverse = (~is_open) & (size_abs > amount_abs)

        # 将反手单分离成两单。注意：trades表占用翻倍
        if np.any(is_reverse):
            orders1 = orders.copy()
            orders2 = orders.copy()
            orders2['is_open'][:] = True

            orders1['qty'][is_reverse] = amount_abs[is_reverse]
            orders2['qty'][is_reverse] -= amount_abs[is_reverse]
            # print(orders2[is_reverse])

            orders = np.concatenate((orders1, orders2[is_reverse]))

        qty = orders['qty']
        is_open = orders['is_open']

        if self._positions_precision == 1.0:
            # 提前条件判断，速度能快一些
            qty = np.where(is_open, np.floor(qty + 1e-9), np.ceil(qty - 1e-9))
        else:
            # 开仓用小数量，平仓用大数量。接近于0时自动调整为0
            qty /= self._positions_precision
            # 10.2/0.2=50.99999999999999
            qty = np.where(is_open, np.floor(qty + 1e-9), np.ceil(qty - 1e-9)) * self._positions_precision
            # 原数字处理后会有小尾巴，简单处理一下
            qty = np.round(qty, 9)

        orders['qty'][:] = qty

        # 过滤无效操作。nan正好也被过滤了不会下单
        return orders[orders['qty'] > 0]

    def run_bar1(self,
                 date: np.int64, size_type: int,
                 asset: np.ndarray, size: np.ndarray, fill_price: np.ndarray) -> None:
        """一层截面信号处理。只处理同时间截面上所有资产的交易信号

        Parameters
        ----------
        date
        size_type
        asset
        size
        fill_price

        """
        # 空指令直接返回
        if size_type == SizeType.NOP:
            return
        # 全空，返回
        if np.all(np.isnan(size)):
            return

        # size转换
        orders: np.ndarray = self.convert_size(size_type, asset, size, fill_price)

        # 过滤后为空
        if len(orders) == 0:
            return

        # 记录上次位置
        self._idx_last_trade = self._idx_curr_trade

        # 先平仓
        orders_close = orders[~orders['is_open']]
        for i in prange(len(orders_close)):
            _o = orders_close[i]
            self.order(date, _o['asset'], _o['is_buy'], _o['is_open'], _o['fill_price'], _o['qty'])

        # 后开仓
        orders_open = orders[orders['is_open']]
        for i in prange(len(orders_open)):
            _o = orders_open[i]
            self.order(date, _o['asset'], _o['is_buy'], _o['is_open'], _o['fill_price'], _o['qty'])

    def run_bar2(self, arr: np.ndarray) -> None:
        """二层截面信号处理。在一层截面信号的基础上多了最新价更新，以及绩效记录

        Parameters
        ----------
        arr
            - date
            - size_type
            - asset
            - size
            - fill_price
            - last_price
            - date_diff

        """
        _date: np.int64 = arr['date'][-1]
        _size_type: int = arr['size_type'][-1]
        _date_diff: bool = arr['date_diff'][-1]
        _asset = arr['asset']

        # 先执行交易
        self.run_bar1(_date, _size_type, _asset, arr['size'], arr['fill_price'])
        # 更新最新价。浮动盈亏得到了调整
        self.update_last_price(_asset, arr['last_price'])
        # 每日收盘记录绩效
        if _date_diff:
            self.update_performances(_date)

    def __str__(self):
        # 这个要少调用，很慢
        with objmode(string=types.unicode_type):
            string = f'Portfolio(Value={self.Value}, Cash={self.Cash}, Equity={self.Equity})'
        return string


# 这种写法是为了方便开关断点调试
if os.environ.get('NUMBA_DISABLE_JIT', '0') != '1':
    # TODO: List支持有问题，不得不这么写，等以后numba修复了再改回来
    list_tmp = List()
    list_tmp.append(Position(0))
    position_list_type = typeof(list_tmp)

    trade_type = typeof(np.empty(1, dtype=trade_dt))
    position_type = typeof(np.empty(1, dtype=position_dt))
    performance_type = typeof(np.empty(1, dtype=performance_dt))

    Portfolio = jitclass(Portfolio,
                         [('_positions', position_list_type),
                          ('_trade_records', trade_type),
                          ('_position_records', position_type),
                          ('_performance_records', performance_type), ])
