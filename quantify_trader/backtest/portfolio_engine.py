from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from quantify_trader.trading.orders import Order, Side
from quantify_trader.trading.portfolio_broker import PortfolioSimBroker

logger = logging.getLogger("quantify.backtest")


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

        daily_rows: list[dict] = []

        logger.info("=" * 80)
        logger.info("回测开始 | 初始资金: %s", f"{broker.initial_cash:,.2f}")
        logger.info("策略: 等权再平衡(20%/30%触发 -> 25%)")
        logger.info("标的: %s", req.symbols)
        logger.info("回测区间: %s ~ %s", df.index[0].strftime("%Y-%m-%d"), df.index[-1].strftime("%Y-%m-%d"))
        logger.info("交易日数: %d", len(df))
        logger.info("=" * 80)

        prev_prices: dict[str, float] | None = None
        prev_positions_qty: dict[str, float] | None = None

        for i, (ts, row) in enumerate(df.iterrows()):
            prices = {sym: float(row[sym]) for sym in req.symbols}
            snap = broker.snapshot(prices)

            did_rebalance = False
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
                did_rebalance = True
            else:
                rebalance_flags.append(0)

            snap2 = broker.snapshot(prices)
            equity = snap2.equity
            peak = max(peak, equity)
            drawdown = equity - peak

            equity_list.append(equity)
            dd_list.append(drawdown)

            date_str = ts.strftime("%Y-%m-%d")

            next_prices: dict[str, float] | None = None
            if i + 1 < len(df):
                next_row = df.iloc[i + 1]
                next_prices = {sym: float(next_row[sym]) for sym in req.symbols}

            daily_pnl = None
            daily_ret = None
            if prev_prices is not None and prev_positions_qty is not None:
                prev_value = sum(
                    float(prev_positions_qty.get(s, 0.0)) * float(prev_prices.get(s, 0.0))
                    for s in req.symbols
                )
                cur_value = sum(
                    float(snap2.positions_qty.get(s, 0.0)) * float(prices.get(s, 0.0))
                    for s in req.symbols
                )
                daily_pnl = cur_value - prev_value
                prev_total = prev_value + snap2.cash - daily_pnl
                if abs(prev_total) > 1e-12:
                    daily_ret = daily_pnl / prev_total

            row_data: dict[str, float | str] = {
                "日期": date_str,
                "权益": round(equity, 2),
                "现金": round(snap2.cash, 2),
                "再平衡": 1 if did_rebalance else 0,
            }
            if daily_pnl is not None:
                row_data["当日盈亏"] = round(daily_pnl, 2)
                row_data["当日收益率"] = round(daily_ret, 6) if daily_ret is not None else 0.0
            else:
                row_data["当日盈亏"] = 0.0
                row_data["当日收益率"] = 0.0

            for s in req.symbols:
                qty = snap2.positions_qty.get(s, 0.0)
                cur_px = prices[s]
                market_val = qty * cur_px
                weight = market_val / equity if equity > 0 else 0.0
                nxt_px = next_prices[s] if next_prices is not None else None
                row_data[f"{s}_数量"] = round(qty, 2)
                row_data[f"{s}_现价"] = cur_px
                row_data[f"{s}_市值"] = round(market_val, 2)
                row_data[f"{s}_占比"] = round(weight, 6)
                row_data[f"{s}_次日价"] = nxt_px if nxt_px is not None else 0.0

            daily_rows.append(row_data)

            if did_rebalance:
                logger.info(
                    "[%s] 再平衡 | 权益: %s | 现金: %s",
                    date_str,
                    f"{equity:>12,.2f}",
                    f"{snap2.cash:>10,.2f}",
                )

            prev_prices = prices
            prev_positions_qty = dict(snap2.positions_qty)

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
        stats["annual_returns"] = _annual_returns(equity_curve["equity"])
        stats["total_return"] = float(equity_curve["equity"].iloc[-1] / equity_curve["equity"].iloc[0] - 1.0)

        daily_df = pd.DataFrame(daily_rows)
        daily_df = daily_df.set_index("日期")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = Path("logs") / f"backtest_daily_{timestamp}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        daily_df.to_csv(str(csv_path), encoding="utf-8-sig")
        logger.info("每日明细已写入: %s", csv_path)

        logger.info("=" * 80)
        logger.info("回测结束")
        logger.info("最终权益: %s", f"{equity_curve['equity'].iloc[-1]:,.2f}")
        logger.info("总收益率: %s", f"{stats['total_return']:.2%}")
        logger.info("年化收益: %s", f"{stats['cagr']:.2%}")
        logger.info("夏普比率: %s", f"{stats['sharpe']:.2f}")
        logger.info("再平衡次数: %d", stats["rebalances"])
        logger.info("交易次数: %d", stats["trades"])
        logger.info("=" * 80)
        logger.info("各年收益:")
        for year, ret in stats["annual_returns"].items():
            logger.info("  %s: %+.2f%%", year, ret * 100)
        logger.info("=" * 80)

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


def _annual_returns(equity: pd.Series) -> dict[str, float]:
    eq = equity.astype(float)
    yearly = eq.resample("YE").last()
    returns: dict[str, float] = {}
    prev = float(eq.iloc[0])
    for ts, val in yearly.items():
        ret = float(val) / prev - 1.0
        returns[str(ts.year)] = ret
        prev = float(val)
    return returns

