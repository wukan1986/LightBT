import os

import numpy as np
from numba import objmode, types, njit, float64, typeof, bool_
from numba.experimental import jitclass

from lightbt.callbacks import commission_0

__TOL__: float = 1e-6


@njit(float64(float64, float64, float64), fastmath=True, nogil=True, cache=True)
def _value_with_mult(price: float, qty: float, mult: float) -> float:
    """计算市值"""
    if mult == 1.0:
        # 少算一次乘法，速度快一些
        return price * qty
    return price * qty * mult


@njit(float64(float64, float64, float64), fastmath=True, nogil=True, cache=True)
def _avg_with_mult(value: float, qty: float, mult: float) -> float:
    """计算均价"""
    if mult == 1.0:
        # 少算一次除法法，速度快一些
        return value / qty
    return value / qty / mult


@njit(float64(float64, float64), fastmath=True, nogil=True, cache=True)
def _net_cash_flow_with_margin(value: float, margin_ratio: float) -> float:
    """计算净现金流"""
    if margin_ratio == 1.0:
        # 少算一次乘法，速度快一些
        return value
    return value * margin_ratio


@njit(bool_(float64), fastmath=True, nogil=True)
def _is_zero(x: float) -> bool:
    """是否为0

    float，double分别遵循R32-24,R64-53的标准。
    所以float的精度误差在1e-6；double精度误差在1e-15
    """
    return (x <= __TOL__) and (x >= -__TOL__)


# 部分参考了SmartQuant部分代码，但又做了大量调整
class Position:
    Asset: int
    # TODO: 由于List中使用嵌套jitclass有问题，不得得将SubPosition简化成float
    # 多头数量
    LongQty: float
    # 空头数量
    ShortQty: float

    # !!! 本来应当将合约乘数，保证金率等信息存在Instrument对象中，但为了简化，安排在Position对象中
    # 合约乘数
    _mult: float
    # 保证金率
    _margin_ratio: float
    # 手续费率
    _commission_ratio: float

    # 最新价。用于计算Value/UPnL
    LastPrice: float
    # 持仓量。空头为负数
    Amount: float
    # 累计买入量
    QtyBought: float
    # 累计卖出量
    QtySold: float
    # 开仓均价
    AvgPx: float

    # 手续费
    _commission: float
    # 累计手续费
    _cum_commission: float
    # 盈亏
    _pnl: float
    # 累计盈亏
    _cum_pnl: float
    # 保证金
    _margin: float
    # 开仓市值
    _open_value: float
    # 市值流变化
    _value_flow: float
    # 净现金流
    _net_cash_flow: float
    # 现金流
    _cash_flow: float

    def __init__(self, asset: int) -> None:
        """初始化

        Parameters
        ----------
        asset: int
            预设的资产顺序ID
        """
        self.Asset = asset

        self._mult = 1.0
        self._margin_ratio = 1.0
        self._commission_ratio = 0.0
        self._commission_fn = commission_0

        self.reset()

    def reset(self):
        self.LongQty = 0.0
        self.ShortQty = 0.0
        self.LastPrice = 0.0
        self.Amount = 0.0
        self.QtyBought = 0.0
        self.QtySold = 0.0
        self.AvgPx = 0.0
        self._commission = 0.0
        self._cum_commission = 0.0
        self._margin = 0.0
        self._open_value = 0.0
        self._pnl = 0.0
        self._cum_pnl = 0.0
        self._value_flow = 0.0
        self._net_cash_flow = 0.0
        self._cash_flow = 0.0

    @property
    def is_long(self) -> bool:
        """是否多头"""
        return self.Amount >= 0

    @property
    def Qty(self) -> float:
        """持仓数量"""
        if self.Amount >= 0:
            return self.Amount
        else:
            return -self.Amount

    @property
    def Value(self) -> float:
        """持仓市值。受last_prce影响"""
        if self.Amount == 0:
            return 0.0
        return self.LastPrice * self.Amount * self._mult

    @property
    def OpenValue(self) -> float:
        """开仓市值"""
        if self.Amount < 0:
            return -self._open_value
        else:
            return self._open_value

    @property
    def Margin(self) -> float:
        """保证金占用"""
        return self._margin

    @property
    def UPnL(self) -> float:
        """持仓盈亏=持仓市值-开仓市值。受`last_prce`影响"""
        if self.Amount == 0:
            return 0.0
        return self.Value - self.OpenValue

    @property
    def PnL(self) -> float:
        """平仓盈亏(未减手续费)"""
        return self._pnl

    @property
    def CumPnL(self) -> float:
        """累计平仓盈亏(未减手续费)"""
        return self._cum_pnl

    @property
    def CumCommission(self) -> float:
        """累计手续费"""
        return self._cum_commission

    @property
    def CashFlow(self) -> float:
        """现金流=平仓盈亏-保证金-手续费"""
        return self._cash_flow

    @property
    def Equity(self) -> float:
        """持仓权益=保证金+浮动盈亏。受`last_price`影响"""
        return self.Margin + self.UPnL

    def calc_value(self, price: float, qty: float)->float:
        """计算市值"""
        return _value_with_mult(price, qty, self._mult)

    def calc_commission(self, is_buy: bool, is_open: bool, value: float, qty: float) -> float:
        """计算手续费"""
        return self._commission_fn(is_buy, is_open, value, qty, self._commission_ratio)

    def openable(self, cash: float, value: float, commission: float) -> bool:
        """可开手数

        需考虑负价格情况
        1. 原油、天然气、电价都出现过负价格
        2. 套利合约的价差也可能是负数

        Parameters
        ----------
        cash: float
            分配的可用现金。注意：不是总现金。
        value: float
            报单市值
        commission: float
            手续费

        Returns
        -------
        bool
            是否成功

        """
        # TODO: 负手续费的情况下是如何处理？
        if cash < 0:
            return False

        if self._margin_ratio == 1.0:
            return (cash + commission) >= value
        else:
            return (cash + commission) >= value * self._margin_ratio

    def closable(self, is_long: bool) -> float:
        """可平手数

        Parameters
        ----------
        is_long: bool
            是否多头

        Returns
        -------
        float

        """
        if is_long:
            return self.LongQty
        else:
            return self.ShortQty

    def set_commission_fn(self, func=commission_0) -> None:
        self._commission_fn = func

    def setup(self, mult: float = 1.0, margin_ratio: float = 1.0, commission_ratio: float = 0.0) -> None:
        """配置资产信息

        Parameters
        ----------
        mult: float
            合约乘数
        margin_ratio: float
            保证金率
        commission_ratio: float
            手续费率

        """
        self._mult = mult
        self._margin_ratio = margin_ratio
        self._commission_ratio = commission_ratio

    def settlement(self) -> float:
        """结算。结算后可能产生现金变动

        1. 逆回购返利息
        2. 分红
        3. 手续费减免
        """
        return 0.0

    def update_last_price(self, last_price: float) -> None:
        """更新最新价。用于计算资金曲线

        Parameters
        ----------
        last_price: float

        """
        self.LastPrice = last_price

    def fill(self, is_buy: bool, is_open: bool, value: float, fill_price: float, qty: float,
             commission: float = 0.0) -> None:
        """通过成交回报，更新各字段

        Parameters
        ----------
        is_buy: bool
            是否买入
        is_open: bool
            是否开仓。反手需要标记成平仓
        value: float
            开仓市值
        fill_price: float
            成交价
        qty: float
            成交量
        commission: float
            手续费

        """
        self._net_cash_flow = 0.0
        self._cash_flow = 0.0

        # 计算开仓市值，平均价。返回改变的持仓市值
        self._calculate(is_open, value, fill_price, qty, commission, self._mult)

        if is_buy:
            self.QtyBought += qty
        else:
            self.QtySold += qty

        # 此处需要更新正确的子持仓对像。如股票买入时只能更新昨仓对象，而买入时只能更新今仓对象
        # !!!注意: is_open与is_long是有区别的
        if is_open:
            if is_buy:
                self.LongQty += qty
            else:
                self.ShortQty += qty
        else:
            if is_buy:
                self.ShortQty -= qty
            else:
                self.LongQty -= qty

        # 新持仓量需做部分计算后再更新
        self.Amount = self.QtyBought - self.QtySold
        # 净现金流
        self._net_cash_flow = -self._value_flow * self._margin_ratio
        # 现金流。考虑了盈亏和手续费
        self._cash_flow = self._pnl + self._net_cash_flow - commission
        # 保证金占用
        self._margin -= self._net_cash_flow
        # 更新最新价，用于计算盈亏
        self.LastPrice = fill_price
        # 累计盈亏
        self._cum_pnl += self._pnl
        # 累计手续费
        self._cum_commission += self._commission

        # 这几个值出现接近0时调整成0
        # 有了这个调整后，回测速度反而加快
        if _is_zero(self.Amount):
            self.Amount = 0.0
            self._margin = 0.0

    def _calculate_pnl(self, is_long: bool, avg_price: float, fill_price: float, qty: float, mult: float) -> float:
        """根据每笔成交计算盈亏。只有平仓才会调用此函数"""
        value: float = fill_price - avg_price if is_long else avg_price - fill_price
        return qty * value * mult

    def _calculate(self, is_open: bool, value: float, fill_price: float, qty: float, commission: float,
                   mult: float) -> None:
        """更新开仓市值和平均价。需用到合约乘数。
        内部函数，不检查合法性。检查提前，有利于加快速度"""
        self._pnl = 0.0
        self._value_flow = 0.0
        self._commission = 0.0

        # 当前空仓
        if self.Amount == 0.0:
            self._value_flow = value  # _value_with_mult(fill_price, qty, mult)
            self._open_value = self._value_flow
            self.AvgPx = fill_price
            self._commission = commission
            return

        # 开仓。已经到这只能是加仓
        if is_open:
            # 开仓改变的市值流与外部计算结果一样
            self._value_flow = value  # _value_with_mult(fill_price, qty, mult)
            self._open_value += self._value_flow
            self.AvgPx = _avg_with_mult(self._open_value, self.Qty + qty, mult)
            self._commission = commission
            return

        # 平仓
        self._pnl = self._calculate_pnl(self.is_long, self.AvgPx, fill_price, qty, mult)

        if _is_zero(self.Qty - qty):
            # 清仓，市值流正好是之前持仓市值
            self._value_flow = -self._open_value
            self._open_value = 0.0
            self.AvgPx = 0.0
            self._commission = commission
            return
        elif self.Qty > qty:
            # 减仓
            self._value_flow = -_value_with_mult(self.AvgPx, qty, mult)
            self._open_value += self._value_flow
            self._commission = commission
            return

        # !!! 为简化外部操作。对于反手情况也支持，但is_open=False

        # 反手。平仓后开仓
        num: float = qty - self.Qty
        old_frozen_value: float = self._open_value
        new_frozen_value: float = _value_with_mult(fill_price, num, mult)
        self._open_value = new_frozen_value
        self.AvgPx = fill_price
        self._value_flow = new_frozen_value - old_frozen_value
        self._commission = commission
        return

    def to_record_position(self, rec: np.ndarray, detail: bool) -> np.ndarray:
        """持仓对象转持仓记录

        Parameters
        ----------
        rec: np.ndarray
        detail: bool

        Returns
        -------
        np.ndarray

        """
        rec['mult'] = self._mult
        rec['margin_ratio'] = self._margin_ratio
        rec['amount'] = self.Amount

        if detail:
            rec['upnl'] = self.UPnL
            rec['value'] = self.Value
            rec['open_value'] = self.OpenValue
            rec['avg_price'] = self.AvgPx
            rec['last_price'] = self.LastPrice
            rec['margin'] = self._margin
            rec['asset'] = self.Asset
            rec['cum_pnl'] = self._cum_pnl
            rec['cum_commission'] = self._cum_commission

        return rec

    def to_record_trade(self, rec: np.ndarray,
                        date: np.uint64, is_buy: bool, is_open: bool, fill_price: float, qty: float,
                        cash: float) -> np.ndarray:
        """订单对象转订单记录"""
        rec['asset'] = self.Asset
        rec['amount'] = self.Amount
        rec['margin'] = self._margin
        rec['commission'] = self._commission
        rec['upnl'] = self.UPnL  # 最新价会导致此项发生变化
        rec['pnl'] = self._pnl
        rec['cash_flow'] = self._cash_flow

        rec['date'] = date
        rec['is_buy'] = is_buy
        rec['is_open'] = is_open
        rec['fill_price'] = fill_price
        rec['qty'] = qty
        rec['cash'] = cash

        return rec

    def to_record_performance(self, rec: np.ndarray, date: np.uint64, cash: float) -> np.ndarray:
        """转绩效"""
        rec['date'] = date
        rec['cash'] = cash
        rec['asset'] = self.Asset
        rec['amount'] = self.Amount
        rec['value'] = self.Value
        rec['margin'] = self._margin
        rec['upnl'] = self.UPnL
        # pnl与commission只记录了产生交易舜时的值，还需要累计值
        rec['cum_pnl'] = self._cum_pnl
        rec['cum_commission'] = self._cum_commission

        return rec

    def __str__(self) -> str:
        # f-string的实现方法比较特殊
        # https://github.com/numba/numba/issues/8969
        with objmode(string=types.unicode_type):  # declare that the "escaping" string variable is of unicode type.
            string = f'Position(Asset={self.Asset}, Value={self.Value}, OpenValue={self.OpenValue}, Margin={self.Margin}, UPnL={self.UPnL}, PnL={self.PnL}, Amount={self.Amount})'
        return string


# 这种写法是为了方便开关断点调试
if os.environ.get('NUMBA_DISABLE_JIT', '0') != '1':
    commission_fn_type = typeof(commission_0)

    Position = jitclass(Position,
                        [('_commission_fn', commission_fn_type)])
