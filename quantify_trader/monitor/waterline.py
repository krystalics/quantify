from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WaterlineSnapshot:
    """
    资产水位快照：当前净值、峰值、回撤等。
    """

    equity: float
    peak: float
    drawdown: float
    max_drawdown: float

    @property
    def max_drawdown_pct(self) -> float:
        return 0.0 if self.peak <= 0 else float(self.max_drawdown / self.peak)

    @staticmethod
    def from_equity_series(equity: pd.Series) -> "WaterlineSnapshot":
        eq = equity.astype(float)
        peak = eq.cummax()
        dd = eq - peak
        mdd = dd.min()
        return WaterlineSnapshot(
            equity=float(eq.iloc[-1]),
            peak=float(peak.iloc[-1]),
            drawdown=float(dd.iloc[-1]),
            max_drawdown=float(mdd),
        )

