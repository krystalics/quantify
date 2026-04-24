from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
from time import sleep


def _yyyymmdd(d: date) -> str:
    return d.strftime("%Y%m%d")


def _fetch_via_akshare(symbol: str, start: date, end: date):
    import akshare as ak

    return ak.fund_etf_hist_em(
        symbol=str(symbol),
        period="daily",
        start_date=_yyyymmdd(start),
        end_date=_yyyymmdd(end),
        adjust="hfq",
    )


def _fetch_via_akshare_stock_hist(symbol: str, start: date, end: date):
    """
    备用：部分 ETF 用 stock_zh_a_hist 也可取到（同样支持 adjust="hfq"）。
    """
    import akshare as ak

    # 交易所前缀：51xxxx 多为上交所；其它按常见规则兜底
    prefix = "sh" if str(symbol).startswith(("5", "6")) else "sz"
    sym = f"{prefix}{symbol}"
    return ak.stock_zh_a_hist(
        symbol=sym,
        period="daily",
        start_date=_yyyymmdd(start),
        end_date=_yyyymmdd(end),
        adjust="hfq",
    )


def _fetch_via_curl(symbol: str, start: date, end: date):
    """
    与 AkShare fund_etf_hist_em 同源的东财 kline 接口。
    用 curl_cffi 绕开 requests 连接被断开的问题。
    """
    import pandas as pd
    from curl_cffi import requests as creq

    # ETF: 1=上交所, 0=深交所。这里做个简单判定：51xxxx 通常是上交所货基/债券ETF。
    market = "1" if str(symbol).startswith(("5", "6")) else "0"
    secid = f"{market}.{symbol}"

    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f116",
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "klt": "101",  # daily
        "fqt": "2",  # hfq
        "beg": _yyyymmdd(start),
        "end": _yyyymmdd(end),
        "secid": secid,
    }
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"}

    r = creq.get(url, params=params, headers=headers, timeout=30, impersonate="chrome")
    r.raise_for_status()
    j = r.json()
    data = (j or {}).get("data") or {}
    klines = data.get("klines") or []
    if not klines:
        return pd.DataFrame()

    rows = [str(x).split(",") for x in klines]
    cols = ["日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率"]
    df = pd.DataFrame(rows, columns=cols)
    for c in cols[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch ETF daily HFQ history via AkShare and save CSV.")
    parser.add_argument("--symbol", default="511880", help="ETF code, e.g. 511880")
    parser.add_argument("--years", type=int, default=10, help="Lookback years")
    parser.add_argument("--out", default=None, help="Output CSV path (default: data/<symbol>_hfq_daily_<years>y.csv)")
    parser.add_argument("--retries", type=int, default=3, help="AkShare request retries")
    args = parser.parse_args()

    end = date.today()
    start = end - timedelta(days=int(args.years) * 365)

    df = None
    last_err: Exception | None = None
    for i in range(max(1, int(args.retries))):
        try:
            df = _fetch_via_akshare(str(args.symbol), start, end)
            break
        except Exception as e:
            last_err = e
            sleep(0.8 * (i + 1))

    if df is None or getattr(df, "empty", True):
        try:
            df = _fetch_via_akshare_stock_hist(str(args.symbol), start, end)
        except Exception as e:
            last_err = e
            df = None

    if df is None or getattr(df, "empty", True):
        df = _fetch_via_curl(str(args.symbol), start, end)
        if df.empty and last_err is not None:
            raise last_err

    if df is None or df.empty:
        raise RuntimeError("AkShare returned empty dataframe")

    out_path = Path(args.out) if args.out else Path("data") / f"{args.symbol}_hfq_daily_{args.years}y.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"saved: {out_path} rows={len(df)}")


if __name__ == "__main__":
    main()

