from numba import cfunc, float64, bool_


# 以下是手续费处理函数，用户也可以自己定义，通过setup进行设置

@cfunc(float64(bool_, bool_, float64, float64, float64))
def commission_0(is_buy: bool, is_open: bool, value: float, qty: float, commission_ratio: float) -> float:
    """0手续费"""
    return 0.0


@cfunc(float64(bool_, bool_, float64, float64, float64))
def commission_by_qty(is_buy: bool, is_open: bool, value: float, qty: float, commission_ratio: float) -> float:
    """按数量计算手续费"""
    return qty * commission_ratio


@cfunc(float64(bool_, bool_, float64, float64, float64))
def commission_by_value(is_buy: bool, is_open: bool, value: float, qty: float, commission_ratio: float) -> float:
    """按市值计算手续费"""
    return value * commission_ratio


@cfunc(float64(bool_, bool_, float64, float64, float64))
def commission_AStock(is_buy: bool, is_open: bool, value: float, qty: float, commission_ratio: float) -> float:
    """按市值计算手续费"""
    if is_open:
        commission = value * commission_ratio
    else:
        # 卖出平仓，多收千1的税
        commission = value * (commission_ratio + 0.001)

    if commission < 5.0:
        return 5.0
    return commission
