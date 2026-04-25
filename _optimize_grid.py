from __future__ import annotations

import itertools
import logging
import time

logging.basicConfig(level=logging.WARNING, format="%(message)s")

import numpy as np

from quantify_trader.data.demo_data import load_real_close_data
from quantify_trader.trading.portfolio_broker import PortfolioSimBroker
from quantify_trader.trading.orders import Order, Side

logger = logging.getLogger("quantify.backtest")
logger.setLevel(logging.WARNING)

close = load_real_close_data("data")

groups = {
    "股票": ["510300"],
    "国债": ["511010"],
    "短债或货币基金": ["511880"],
    "黄金": ["518880"],
}
symbols = ["510300", "511010", "511880", "518880"]
group_names = list(groups.keys())

df = close[symbols].dropna(how="any")


COMMISSION_RATE = 0.00015  # 万分之一点五


def run_backtest(group_targets: dict[str, float], lower: float, upper: float) -> dict:
    broker = PortfolioSimBroker(initial_cash=1_000_000.0, commission_rate=COMMISSION_RATE)
    rebalances = 0
    equity_list: list[float] = []

    for ts, row in df.iterrows():
        prices = {sym: float(row[sym]) for sym in symbols}
        snap = broker.snapshot(prices)

        group_weights: dict[str, float] = {}
        for g, members in groups.items():
            gw = sum(float(snap.weights.get(s, 0.0)) for s in members)
            group_weights[g] = gw

        do_rebalance = any(gw > upper or gw < lower for gw in group_weights.values())

        if do_rebalance:
            target_values = {s: group_targets[g] / len(members) * snap.equity
                             for g, members in groups.items()
                             for s in members}
            diffs = {s: target_values[s] - float(snap.positions_value.get(s, 0.0)) for s in symbols}

            for s in symbols:
                diff = diffs[s]
                if abs(diff) / max(snap.equity, 1e-12) < 1e-6:
                    continue
                px = float(prices[s])
                qty = abs(diff) / max(px, 1e-12)
                side = Side.SELL if diff < 0 else Side.BUY
                broker.place_order(Order(symbol=s, side=side, qty=qty, price=px))
            rebalances += 1

        snap2 = broker.snapshot(prices)
        equity_list.append(snap2.equity)

    eq = np.asarray(equity_list, dtype=float)
    rets = (eq[1:] - eq[:-1]) / eq[:-1]
    vol = float(rets.std(ddof=0)) if len(rets) > 1 else 0.0
    mean = float(rets.mean())
    cagr = float((eq[-1] / max(eq[0], 1e-12)) ** (252.0 / max(len(eq), 1)) - 1.0) * 100
    total_ret = (eq[-1] / eq[0] - 1.0) * 100
    sharpe = float((mean / vol) * np.sqrt(252.0)) if vol > 1e-12 else float("nan")

    return {"cagr": cagr, "total_return": total_ret, "sharpe": sharpe, "rebalances": rebalances}


# 资产占比 10%~40%, 步长 5%
candidates = [round(v * 0.01, 2) for v in range(10, 41, 5)]

results: list[tuple] = []
total_combos = 0

start = time.time()

for a, b, c, d in itertools.product(candidates, repeat=4):
    total = round(a + b + c + d, 2)
    if abs(total - 1.0) > 1e-9:
        continue
    total_combos += 1

    group_targets = {group_names[0]: a, group_names[1]: b, group_names[2]: c, group_names[3]: d}

    # 波动 5%~15%
    for pct in range(5, 16):
        lower = round(0.25 - pct / 100, 2)
        upper = round(0.25 + pct / 100, 2)

        stats = run_backtest(group_targets, lower, upper)
        results.append((
            a, b, c, d,
            pct, lower, upper,
            stats["cagr"],
            stats["total_return"],
            stats["sharpe"],
            stats["rebalances"],
        ))

elapsed = time.time() - start

print(f"资产10~40%(步长5%) × 波动5~15%: {total_combos} 种资产组合 × 11 种波动 = {len(results)} 种组合, 耗时 {elapsed:.1f}s")
print()

results.sort(key=lambda r: r[7], reverse=True)

print(f"{'排名':>4}  {'股票':>5}  {'国债':>5}  {'短债':>5}  {'黄金':>5}  {'波动%':>5}  {'下限':>5}  {'上限':>5}  {'年化%':>8}  {'总收益%':>9}  {'夏普':>7}  {'再平衡':>5}")
print("=" * 85)

for rank, r in enumerate(results[:30], 1):
    a, b, c, d, pct, lower, upper, cagr, total_ret, sharpe, rebs = r
    print(f"{rank:>4}  {a*100:>4.0f}%  {b*100:>4.0f}%  {c*100:>4.0f}%  {d*100:>4.0f}%  {pct:>4}%  {lower:>5.2f}  {upper:>5.2f}  {cagr:>7.2f}%  {total_ret:>8.2f}%  {sharpe:>7.2f}  {rebs:>5}")
