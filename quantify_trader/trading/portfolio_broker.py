from __future__ import annotations

from dataclasses import dataclass

from quantify_trader.trading.orders import Order, Side


@dataclass(frozen=True)
class PortfolioSnapshot:
    cash: float
    equity: float
    positions_qty: dict[str, float]
    positions_value: dict[str, float]
    weights: dict[str, float]


@dataclass
class PortfolioSimBroker:
    """
    多标的组合模拟券商（最小实现）：
    - 市价成交（用传入的 price）
    - 不考虑手续费/滑点（可扩展）
    - 允许多个 symbol 持仓
    """

    initial_cash: float = 1_000_000.0
    cash: float = 0.0
    positions_qty: dict[str, float] | None = None

    def __post_init__(self) -> None:
        self.cash = float(self.initial_cash)
        if self.positions_qty is None:
            self.positions_qty = {}

    def snapshot(self, prices: dict[str, float]) -> PortfolioSnapshot:
        positions_value: dict[str, float] = {}
        total_pos = 0.0
        for sym, qty in self.positions_qty.items():
            px = float(prices.get(sym, 0.0))
            val = float(qty) * px
            positions_value[sym] = val
            total_pos += val
        equity = self.cash + total_pos

        weights: dict[str, float] = {}
        if equity > 0:
            for sym, val in positions_value.items():
                weights[sym] = val / equity
        return PortfolioSnapshot(
            cash=self.cash,
            equity=equity,
            positions_qty=dict(self.positions_qty),
            positions_value=positions_value,
            weights=weights,
        )

    def place_order(self, order: Order) -> None:
        if order.qty <= 0:
            return
        sym = order.symbol
        px = float(order.price)
        qty = float(order.qty)

        if order.side == Side.BUY:
            cost = qty * px
            if cost > self.cash and px > 0:
                qty = self.cash / px
                cost = qty * px
            if qty <= 0:
                return
            self.cash -= cost
            self.positions_qty[sym] = self.positions_qty.get(sym, 0.0) + qty
            return

        # SELL
        held = self.positions_qty.get(sym, 0.0)
        sell_qty = min(qty, held)
        if sell_qty <= 0:
            return
        self.positions_qty[sym] = held - sell_qty
        if self.positions_qty[sym] <= 1e-12:
            self.positions_qty.pop(sym, None)
        self.cash += sell_qty * px

