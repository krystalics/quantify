from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantify_trader.trading.orders import Order, Side
from quantify_trader.trading.portfolio_broker import PortfolioSimBroker


@dataclass(frozen=True)
class PortfolioBacktestRequest:
    symbols: list[str]
    close_prices: pd.DataFrame  # index: datetime, columns: symbols, values: close
    strategy: object
    broker: PortfolioSimBroker


@dataclass(frozen=True)
class PortfolioBacktestResult:
    equity_curve: pd.DataFrame
    stats: dict


class PortfolioBacktestEngine:
    """
    多标的组合回测引擎（按日收盘价、再平衡策略）。
    """

    def run(self, req: PortfolioBacktestRequest) -> PortfolioBacktestResult:
        df = req.close_prices.copy()
        df = df[req.symbols].dropna(how="any")
        broker = req.broker
        strategy = req.strategy

        equity_list: list[float] = []
        dd_list: list[float] = []
        rebalance_flags: list[int] = []
        trades = 0
        rebalances = 0
        peak = -np.inf

        for ts, row in df.iterrows():
            prices = {sym: float(row[sym]) for sym in req.symbols}
            snap = broker.snapshot(prices)

            if strategy.should_rebalance(snap.weights):
                target_w = strategy.target_weights()
                snap = _rebalance_to_targets(
                    symbols=req.symbols,
                    prices=prices,
                    broker=broker,
                    equity=snap.equity,
                    current_values=snap.positions_value,
                    target_weights=target_w,
                )
                trades += snap["trades"]
                rebalances += 1
                rebalance_flags.append(1)
            else:
                rebalance_flags.append(0)

            snap2 = broker.snapshot(prices)
            equity = snap2.equity
            peak = max(peak, equity)
            drawdown = equity - peak

            equity_list.append(equity)
            dd_list.append(drawdown)

        equity_curve = pd.DataFrame(
            {
                "equity": np.asarray(equity_list, dtype=float),
                "drawdown": np.asarray(dd_list, dtype=float),
                "rebalance": np.asarray(rebalance_flags, dtype=int),
            },
            index=df.index,
        )
        stats = _stats_from_equity(equity_curve["equity"])
        stats["trades"] = trades
        stats["rebalances"] = rebalances
        stats["years"] = float(len(equity_curve)) / 252.0
        return PortfolioBacktestResult(equity_curve=equity_curve, stats=stats)


def _rebalance_to_targets(
    *,
    symbols: list[str],
    prices: dict[str, float],
    broker: PortfolioSimBroker,
    equity: float,
    current_values: dict[str, float],
    target_weights: dict[str, float],
) -> dict:
    # 目标每个资产的市值
    target_values = {s: float(target_weights.get(s, 0.0)) * equity for s in symbols}
    diffs = {s: target_values[s] - float(current_values.get(s, 0.0)) for s in symbols}

    trades = 0

    # 先卖超配的，释放现金（避免买入被现金约束）
    for s in symbols:
        diff = diffs[s]
        if diff < 0:
            px = float(prices[s])
            qty = (-diff) / max(px, 1e-12)
            broker.place_order(Order(symbol=s, side=Side.SELL, qty=qty, price=px))
            trades += 1

    # 再买入低配的
    for s in symbols:
        diff = diffs[s]
        if diff > 0:
            px = float(prices[s])
            qty = diff / max(px, 1e-12)
            broker.place_order(Order(symbol=s, side=Side.BUY, qty=qty, price=px))
            trades += 1

    return {"trades": trades}


def _stats_from_equity(equity: pd.Series) -> dict:
    eq = equity.astype(float)
    rets = eq.pct_change().fillna(0.0)
    vol = float(rets.std(ddof=0))
    mean = float(rets.mean())
    cagr = float((eq.iloc[-1] / max(eq.iloc[0], 1e-12)) ** (252.0 / max(len(eq), 1)) - 1.0)
    sharpe = float((mean / vol) * np.sqrt(252.0)) if vol > 1e-12 else float("nan")
    return {"cagr": cagr, "sharpe": sharpe}

