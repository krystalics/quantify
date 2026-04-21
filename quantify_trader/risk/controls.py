from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskLimits:
    # 目标仓位上限（权益占比）
    max_position_pct: float = 0.95
    # 最大杠杆（这里用最简：position_value / equity）
    max_leverage: float = 1.0
    # 单日最大亏损（相对昨日权益），触发则强制清仓
    max_daily_loss_pct: float = 0.05

