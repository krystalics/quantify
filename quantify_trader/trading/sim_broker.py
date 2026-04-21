from __future__ import annotations

from dataclasses import dataclass

from quantify_trader.trading.broker import PortfolioSnapshot
from quantify_trader.trading.orders import Order, Side


@dataclass
class SimBroker:
    """
    一个最小模拟撮合：
    - 市价成交（用给定 price）
    - 不考虑滑点/手续费（可以很容易扩展）
    """

    initial_cash: float = 1_000_000.0
    cash: float = 0.0
    position_qty: float = 0.0

    def __post_init__(self) -> None:
        self.cash = float(self.initial_cash)

    def snapshot(self, symbol: str, price: float) -> PortfolioSnapshot:
        pos_val = self.position_qty * price
        return PortfolioSnapshot(
            cash=self.cash,
            position_qty=self.position_qty,
            position_value=pos_val,
            equity=self.cash + pos_val,
        )

    def place_order(self, order: Order) -> None:
        if order.qty <= 0:
            return
        cost = order.qty * order.price
        if order.side == Side.BUY:
            if cost > self.cash:
                # 资金不足：降到可买数量
                order_qty = self.cash / order.price
                cost = order_qty * order.price
                self.position_qty += order_qty
                self.cash -= cost
                return
            self.position_qty += order.qty
            self.cash -= cost
            return

        # SELL
        qty = min(order.qty, self.position_qty)
        self.position_qty -= qty
        self.cash += qty * order.price

