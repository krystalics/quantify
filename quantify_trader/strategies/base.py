from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


@dataclass(frozen=True)
class SignalFrame:
    """
    信号约定：
    - target_position: 目标仓位（-1..1），1 表示满仓做多，-1 表示满仓做空（股票通常只用 0..1）
    """

    target_position: pd.Series


class Strategy(Protocol):
    def generate_signals(self, ohlcv: pd.DataFrame) -> SignalFrame: ...

