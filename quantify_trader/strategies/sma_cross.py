from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quantify_trader.strategies.base import SignalFrame


@dataclass(frozen=True)
class SmaCrossParams:
    fast: int = 10
    slow: int = 30


class SmaCrossStrategy:
    def __init__(self, params: SmaCrossParams) -> None:
        if params.fast >= params.slow:
            raise ValueError("fast must be < slow")
        self.params = params

    def generate_signals(self, ohlcv: pd.DataFrame) -> SignalFrame:
        close = ohlcv["close"]
        fast = close.rolling(self.params.fast).mean()
        slow = close.rolling(self.params.slow).mean()

        # 简单规则：快线上穿慢线 -> 1；下穿 -> 0
        long = (fast > slow).astype(float).fillna(0.0)
        target = long.rename("target_position")
        return SignalFrame(target_position=target)

