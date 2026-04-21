from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from quantify_trader.trading.orders import Order


@dataclass(frozen=True)
class PortfolioSnapshot:
    cash: float
    position_qty: float
    position_value: float
    equity: float


class Broker(Protocol):
    def snapshot(self, symbol: str, price: float) -> PortfolioSnapshot: ...

    def place_order(self, order: Order) -> None: ...

