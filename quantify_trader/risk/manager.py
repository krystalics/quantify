from __future__ import annotations

from dataclasses import dataclass

from quantify_trader.risk.controls import RiskLimits


@dataclass
class RiskManager:
    limits: RiskLimits
    yesterday_equity: float | None = None
    stopped: bool = False

    def update_day_boundary(self, equity: float) -> None:
        self.yesterday_equity = equity
        self.stopped = False

    def allow_target_position(self, target_position: float) -> float:
        if self.stopped:
            return 0.0
        return float(max(0.0, min(target_position, self.limits.max_position_pct)))

    def check_daily_loss(self, equity: float) -> None:
        if self.yesterday_equity is None:
            return
        dd = (equity - self.yesterday_equity) / self.yesterday_equity
        if dd <= -abs(self.limits.max_daily_loss_pct):
            self.stopped = True

