import os

import numpy as np
from numba import typeof, objmode, types, prange
from numba.experimental import jitclass
from numba.typed.typedlist import List

from lightbt.enums import SizeType, order_inside_dt, trade_dt, position_dt, performance_dt
from lightbt.position import Position


class Portfolio:
    _cash: float
    _idx_trade: int
    _idx_performance: int
    _max_trades: int
    _max_performances: int

    def __init__(self, max_trades: int = 1024, max_performances: int = 1024) -> None:
        """初始化

        Parameters
        ----------
        max_trades: int
            记录成交的缓存大小。空间不足时将丢弃
        max_performances: int
            记录绩效的缓存大小。空间不足时将丢弃
        """
        # https://github.com/numba/numba/issues/8733
        data_tmp = List()
        data_tmp.append(Position(0))
        data_tmp.clear()

        self._positions = data_tmp
        self._trade_records = np.empty(max_trades, dtype=trade_dt)
        self._position_records = np.empty(1, dtype=position_dt)
        self._performance_records = np.empty(max_performances, dtype=performance_dt)

        self._cash = 0.0
        self._idx_trade = 0
        self._idx_performance = 0
        self._max_trades = max_trades
        self._max_performances = max_performances

    def reset(self):
        self._idx_trade = 0
        self._idx_performance = 0

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

    def setup(self, asset: np.ndarray, mult: np.ndarray, margin_ratio: np.ndarray) -> None:
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

        """
        # 指定长度进行初始化
        count = len(mult)
        while len(self._positions) < count:
            self._positions.append(Position(len(self._positions)))

        # 创建记录体，用于最终显示持仓
        self._position_records = np.empty(len(self._positions), dtype=position_dt)

        for i in prange(count):
            self._positions[asset[i]].setup(mult[i], margin_ratio[i])

    def _fill_trade_record(self,
                           date: np.int64, asset: int,
                           is_buy: bool, is_open: bool, fill_price: float, qty: float) -> None:
        """遇到有效成交时自动更新，所以内容直接取即可"""
        if self._idx_trade >= self._max_trades:
            return
        rec = self._trade_records[self._idx_trade]

        self._positions[asset].to_record_trade(rec, date, is_buy, is_open, fill_price, qty, self._cash)

        self._idx_trade += 1

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
        for i, pos in enumerate(self._positions):
            if self._idx_performance >= self._max_performances:
                return

            rec = self._performance_records[self._idx_performance]
            pos.to_record_performance(rec, date, cash)

            self._idx_performance += 1

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
            pos.settlement()

    def performances(self) -> np.ndarray:
        """绩效记录"""
        return self._performance_records[:self._idx_performance]

    def trades(self) -> np.ndarray:
        """很多变量只记录了瞬时值，当需要时序值时，通过此函数记录下来备用"""
        return self._trade_records[:self._idx_trade]

    def positions(self) -> np.ndarray:
        """最新持仓记录"""
        self._fill_position_records(True)
        return self._position_records

    def order(self, date: np.int64, asset: int, is_buy: bool, is_open: bool, fill_price: float, qty: float, commission: float = 0.0) -> bool:
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
        commission: float

        Returns
        -------
        bool

        """
        if qty <= 0:
            # 数量不合法，返回。可用于刷新行情但不产生交易记录
            return False

        pos: Position = self._positions[asset]
        if is_open:
            # 可开手数检查
            if not pos.openable(self._cash, fill_price, qty, commission):
                return False
        else:
            # TODO: 可能有反手情况。这个以后再处理
            pass

        pos.fill(is_buy, is_open, fill_price, qty, commission)
        self._cash += pos.CashFlow

        self._fill_trade_record(date, asset, is_buy, is_open, fill_price, qty)

        return True

    def convert_size(self, size_type: int, asset: np.ndarray, size: np.ndarray, fill_price: np.ndarray, commission: np.ndarray) -> np.ndarray:

        self._fill_position_records(False)
        # asset不能出现重复
        _rs: np.ndarray = self._position_records[asset]
        margin_ratio: np.ndarray = _rs['margin_ratio']
        amount: np.ndarray = _rs['amount']
        mult: np.ndarray = _rs['mult']

        if size_type == SizeType.TargetPercentMargin:
            # 总权益转分别使用保证金再转市值
            _equity: float = self.Equity * np.sum(np.abs(size))
            size = _equity * size / margin_ratio
            size_type = SizeType.TargetValue
        if size_type == SizeType.TargetPercentValue:
            # 总权益除保证金率占比，得到总市值，然后得到分别市值
            _equity: float = self.Equity * np.sum(np.abs(size))
            _ratio: float = np.sum((np.abs(size) * margin_ratio))
            if _ratio == 0:
                size = _equity * size
            else:
                size = _equity / _ratio * size
            size_type = SizeType.TargetValue
        if size_type == SizeType.TargetMargin:
            # 保证金转成市值
            size = size / margin_ratio
            size_type = SizeType.TargetValue
        if size_type == SizeType.TargetValue:
            # 前后市值之差
            size = size - (fill_price * amount * mult)
            size_type = SizeType.Value
        if size_type == SizeType.TargetAmount:
            # 前后Amout差值
            size = size - amount
            size_type = SizeType.Amount
        if size_type == SizeType.Percent:
            # 买入开仓，用现金转市值
            # 卖出平仓，持仓市值的百分比
            # TOOD: 由于无法表示卖出开仓。所以只能用在股票市场
            _cash_per_lot = fill_price * mult * margin_ratio
            size = np.where(size >= 0, self._cash / _cash_per_lot, amount) * size
            size_type = SizeType.Amount
        if size_type == SizeType.Margin:
            # 将保证金转换成市值
            size = size / margin_ratio
            size_type = SizeType.Value
        if size_type == SizeType.Value:
            # 将市值转成手数
            size = size / (fill_price * mult)
            size_type = SizeType.Amount
        if size_type == SizeType.Amount:
            # 最后都汇总到此，手数是否调整？
            pass

        # TODO: 是否要将反手拆分成两条？
        orders = np.empty(len(asset), dtype=order_inside_dt)
        orders['asset'][:] = asset
        orders['commission'][:] = commission
        orders['fill_price'][:] = fill_price
        orders['size'][:] = size
        orders['qty'][:] = np.abs(size)
        orders['is_buy'][:] = size >= 0

        is_open: np.ndarray = np.sign(amount) * np.sign(size)
        orders['is_open'][:] = np.where(is_open == 0, amount == 0, is_open > 0)
        orders['amount'][:] = amount

        # 过滤无效操作
        return orders[orders['qty'] > 0]

    def run_bar(self,
                date: np.int64, size_type: int,
                asset: np.ndarray, size: np.ndarray, fill_price: np.ndarray, commission: np.ndarray) -> None:
        """同一截面，时间相同，先平后开"""
        orders: np.ndarray = self.convert_size(size_type, asset, size, fill_price, commission)

        # 先平仓
        orders_close = orders[~orders['is_open']]
        for i in prange(len(orders_close)):
            _o = orders_close[i]
            self.order(date, _o['asset'], _o['is_buy'], _o['is_open'], _o['fill_price'], _o['qty'], _o['commission'])

        # 后开仓
        orders_open = orders[orders['is_open']]
        for i in prange(len(orders_open)):
            _o = orders_open[i]
            self.order(date, _o['asset'], _o['is_buy'], _o['is_open'], _o['fill_price'], _o['qty'], _o['commission'])

    def run_bar2(self,
                 date: np.int64, size_type: int,
                 asset: np.ndarray, size: np.ndarray, fill_price: np.ndarray, commission: np.ndarray,
                 last_price: np.ndarray,
                 date_diff: bool) -> None:
        # 先执行交易
        self.run_bar(date, size_type, asset, size, fill_price, commission)
        # 更新最新价。浮动盈亏得到了调整
        self.update_last_price(asset, last_price)
        # 每日收盘记录绩效
        if date_diff:
            self.update_performances(date)

    def run_bar3(self, idx: np.ndarray, arr: np.ndarray) -> None:
        for i, j in zip(idx[:-1], idx[1:]):
            a = arr[i:j]
            self.run_bar2(a['date'][-1], a['size_type'][-1],
                          a['asset'], a['size'], a['fill_price'], a['commission'], a['last_price'],
                          a['date_diff'][-1])

    def __str__(self):
        # 这个要少调用，很慢
        with objmode(string=types.unicode_type):
            string = f'Portfolio(Value={self.Value}, Cash={self.Cash}, Equity={self.Equity})'
        return string


# 这种写法是为了方便开关断点调试
if os.environ.get('NUMBA_DISABLE_JIT', '0') != '1':
    # TODO: List支持有问题，不得不这么写，等以后numba修复了再改回来
    l = List()
    l.append(Position(0))
    position_list_type = typeof(l)
    trade_type = typeof(np.empty(1, dtype=trade_dt))
    position_type = typeof(np.empty(1, dtype=position_dt))
    performance_type = typeof(np.empty(1, dtype=performance_dt))

    Portfolio = jitclass(Portfolio,
                         [('_positions', position_list_type), ('_trade_records', trade_type),
                          ('_position_records', position_type), ('_performance_records', performance_type), ])
