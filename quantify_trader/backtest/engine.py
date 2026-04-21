from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantify_trader.data.demo_data import OhlcvFrame
from quantify_trader.risk.controls import RiskLimits
from quantify_trader.risk.manager import RiskManager
from quantify_trader.strategies.base import Strategy
from quantify_trader.trading.orders import Order, Side
from quantify_trader.trading.sim_broker import SimBroker


@dataclass(frozen=True)
class BacktestRequest:
    symbol: str
    ohlcv: OhlcvFrame
    strategy: Strategy
    broker: SimBroker
    risk_limits: RiskLimits


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: pd.DataFrame
    stats: dict


class BacktestEngine:
    """
    最小回测引擎（单标的、按日收盘价、目标仓位法）：
    - 策略产生 target_position（0..1）
    - 风控裁剪 target_position，必要时清仓
    - 交易用“调整到目标仓位”的方式生成买卖
    """

    def run(self, req: BacktestRequest) -> BacktestResult:
        df = req.ohlcv.df.copy()
        signals = req.strategy.generate_signals(df)

        rm = RiskManager(req.risk_limits)
        broker = req.broker

        equity_list: list[float] = []
        dd_list: list[float] = []
        pos_list: list[float] = []
        trades = 0

        peak = -np.inf

        # “单日”边界：这里用 index 的日期变化判定（演示数据按交易日）
        last_day = None
        for ts, row in df.iterrows():
            day = ts.date()
            if last_day is None or day != last_day:
                snap = broker.snapshot(req.symbol, float(row["close"]))
                rm.update_day_boundary(snap.equity)
                last_day = day

            price = float(row["close"])
            snap = broker.snapshot(req.symbol, price)

            rm.check_daily_loss(snap.equity)

            target_pos = float(signals.target_position.loc[ts])
            target_pos = rm.allow_target_position(target_pos)

            # 目标头寸（以权益计）
            target_value = target_pos * snap.equity
            current_value = snap.position_value
            diff_value = target_value - current_value

            # 生成“调仓”订单
            if abs(diff_value) / max(snap.equity, 1e-9) > 1e-4:
                if diff_value > 0:
                    qty = diff_value / price
                    broker.place_order(Order(symbol=req.symbol, side=Side.BUY, qty=qty, price=price))
                    trades += 1
                else:
                    qty = (-diff_value) / price
                    broker.place_order(
                        Order(symbol=req.symbol, side=Side.SELL, qty=qty, price=price)
                    )
                    trades += 1

            snap2 = broker.snapshot(req.symbol, price)
            equity = snap2.equity
            peak = max(peak, equity)
            drawdown = equity - peak

            equity_list.append(equity)
            dd_list.append(drawdown)
            pos_list.append(snap2.position_value / max(snap2.equity, 1e-9))

        equity_curve = pd.DataFrame(
            {
                "equity": np.asarray(equity_list, dtype=float),
                "drawdown": np.asarray(dd_list, dtype=float),
                "position_pct": np.asarray(pos_list, dtype=float),
            },
            index=df.index,
        )
        stats = _stats_from_equity(equity_curve["equity"])
        stats["trades"] = trades
        return BacktestResult(equity_curve=equity_curve, stats=stats)


def _stats_from_equity(equity: pd.Series) -> dict:
    eq = equity.astype(float)
    rets = eq.pct_change().fillna(0.0)
    vol = float(rets.std(ddof=0))
    mean = float(rets.mean())

    # 近似年化：按 252 交易日
    cagr = float((eq.iloc[-1] / max(eq.iloc[0], 1e-12)) ** (252.0 / max(len(eq), 1)) - 1.0)
    sharpe = float((mean / vol) * np.sqrt(252.0)) if vol > 1e-12 else float("nan")
    return {"cagr": cagr, "sharpe": sharpe}

