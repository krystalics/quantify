from __future__ import annotations
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")

from quantify_trader.data.demo_data import load_real_close_data
from quantify_trader.backtest.portfolio_engine import PortfolioBacktestEngine, PortfolioBacktestRequest
from quantify_trader.strategies.rebalance_equal_weight import GroupedEqualWeightRebalanceStrategy, RebalanceParams
from quantify_trader.trading.portfolio_broker import PortfolioSimBroker

close = load_real_close_data("data")
groups = {"沪深300": ["510300"], "国债ETF": ["511010"], "银华日利": ["511880"], "黄金ETF": ["518880"]}
symbols = ["510300", "511010", "511880", "518880"]
strategy = GroupedEqualWeightRebalanceStrategy(groups, RebalanceParams(0.20, 0.30))
broker = PortfolioSimBroker(initial_cash=1_000_000.0)
req = PortfolioBacktestRequest(symbols=symbols, close_prices=close, strategy=strategy, broker=broker)
result = PortfolioBacktestEngine().run(req)

import pandas as pd, glob
csv_file = sorted(glob.glob("logs/backtest_daily_*.csv"))[-1]
df = pd.read_csv(csv_file)
print("Columns:", list(df.columns))
print("Shape:", df.shape)
print()
print(df.head(10).to_string(index=False))
