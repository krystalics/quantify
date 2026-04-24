from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OhlcvFrame:
    df: pd.DataFrame

    @property
    def close(self) -> pd.Series:
        return self.df["close"]


def make_demo_ohlcv(n: int = 500, seed: int = 1) -> OhlcvFrame:
    """
    生成一份可回测的演示 OHLCV 数据（合成随机游走 + 波动聚类）。
    真实项目中你可以用同样的 OhlcvFrame 适配 CSV/数据库/券商历史行情。
    """
    rng = np.random.default_rng(seed)

    # GARCH-ish 的简化：用状态波动控制噪声强度
    vol = np.zeros(n, dtype=float)
    eps = rng.normal(0, 1, size=n)
    vol[0] = 0.01
    for i in range(1, n):
        vol[i] = 0.85 * vol[i - 1] + 0.15 * (0.005 + 0.03 * abs(eps[i - 1]))
    rets = vol * eps

    price0 = 100.0
    close = price0 * np.exp(np.cumsum(rets))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0, 0.01, size=n))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0, 0.01, size=n))
    volume = rng.integers(100_000, 2_000_000, size=n).astype(float)

    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )
    return OhlcvFrame(df=df)


def make_demo_multi_close(symbols: list[str], years: int = 3, seed: int = 7) -> pd.DataFrame:
    """
    生成多标的收盘价 DataFrame（用于组合回测演示）：
    - index: 交易日
    - columns: symbols
    - values: close
    """
    if len(symbols) < 2:
        raise ValueError("symbols must have at least 2")
    n = int(years) * 252
    idx = pd.date_range("2020-01-01", periods=n, freq="B")

    # 相关的收益：共享因子 + 个体噪声
    rng = np.random.default_rng(seed)
    mkt = rng.normal(0, 0.008, size=n)

    closes: dict[str, np.ndarray] = {}
    for i, sym in enumerate(symbols):
        idio = rng.normal(0, 0.01, size=n)
        drift = 0.00015 + (i - len(symbols) / 2) * 0.00002
        rets = drift + 0.55 * mkt + 0.45 * idio
        close = (80.0 + 10.0 * i) * np.exp(np.cumsum(rets))
        closes[sym] = close

    return pd.DataFrame(closes, index=idx)


def load_real_close_data(data_dir: str | Path) -> pd.DataFrame:
    """
    从 quantify/data 目录加载真实 CSV 数据，返回收盘价 DataFrame。
    CSV 文件格式支持中文列名（日期/收盘）和英文列名（date/close）。
    """
    data_path = Path(data_dir)
    csv_files = sorted(data_path.glob("*_hfq_daily_10y.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    close_dict: dict[str, pd.Series] = {}
    for fpath in csv_files:
        symbol = fpath.stem.split("_")[0]
        df = pd.read_csv(fpath, encoding="utf-8")

        date_col = "日期" if "日期" in df.columns else "date"
        close_col = "收盘" if "收盘" in df.columns else "close"

        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        close_dict[symbol] = df[close_col].astype(float)

    result = pd.DataFrame(close_dict)
    result = result.sort_index()
    result = result[result.index.duplicated(keep="first") == False]
    return result

