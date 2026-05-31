# ma_pe_analysis.py — Moving Average Strategy Effectiveness Across P/E Regimes
# Reid Fintak | See README.md and the accompanying paper for full methodology.

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy import stats


# -- Config --------------------------------------------------------------------

START_DATE   = "2020-01-01"
END_DATE     = "2025-01-01"
FEE_RATE     = 0.001          # 0.1% per trade
START_CASH   = 1000.0
MA_WINDOWS   = [5, 10, 20, 50]


# -- Quarterly GAAP diluted EPS ------------------------------------------------
# Dates the earnings were REPORTED (made public), not the quarter-end date.
# Values were verified using SEC 8-K filings and macrotrends.net.

EPS = {

    "AAPL": {
        "2020-01-28":1.25, "2020-04-30":0.64, "2020-07-30":0.65, "2020-10-29":0.74,
        "2021-01-27":1.68, "2021-04-28":1.40, "2021-07-27":1.30, "2021-10-28":1.23,
        "2022-01-27":2.10, "2022-04-28":1.52, "2022-07-28":1.20, "2022-10-27":1.29,
        "2023-02-02":1.88, "2023-05-04":1.52, "2023-08-03":1.26, "2023-11-02":1.47,
        "2024-02-01":2.18, "2024-05-02":1.53, "2024-08-01":1.40, "2024-10-31":0.97,
        "2025-01-30":2.40,
    },

    "MSFT": {
        "2020-01-29":1.51, "2020-04-29":1.40, "2020-07-22":1.47, "2020-10-28":1.82,
        "2021-01-26":2.03, "2021-04-27":2.03, "2021-07-27":2.17, "2021-10-27":2.71,
        "2022-01-25":2.48, "2022-04-26":2.22, "2022-07-26":2.24, "2022-10-25":2.35,
        "2023-01-24":2.20, "2023-04-25":2.45, "2023-07-25":2.68, "2023-10-24":2.99,
        "2024-01-30":2.93, "2024-04-25":2.94, "2024-07-30":2.94, "2024-10-30":3.30,
        "2025-01-29":3.23,
    },

    # XOM had negative EPS throughout 2020 (large write-down in Q4: -$4.70).
    # Days where TTM EPS <= 0 are excluded from P/E analysis (P/E ratio is undefined).

    "XOM": {
        "2020-05-01":-0.14, "2020-07-31":-0.26, "2020-10-30":-0.15, "2021-02-02":-4.70,
        "2021-04-30": 0.64, "2021-07-30": 1.10, "2021-10-29": 1.57, "2022-02-01": 2.08,
        "2022-04-29": 1.28, "2022-07-29": 4.21, "2022-10-28": 4.68, "2023-01-31": 3.09,
        "2023-04-28": 2.79, "2023-07-28": 1.94, "2023-10-27": 2.25, "2024-02-02": 1.91,
        "2024-04-26": 2.06, "2024-07-26": 2.14, "2024-11-01": 1.92, "2025-01-31": 1.72,
    },

    "JPM": {
        "2020-04-14":0.78, "2020-07-14":1.38, "2020-10-13":2.92, "2021-01-15":3.80,
        "2021-04-14":4.50, "2021-07-13":3.78, "2021-10-13":3.74, "2022-01-14":3.34,
        "2022-04-13":2.63, "2022-07-14":2.76, "2022-10-14":3.12, "2023-01-13":3.58,
        "2023-04-14":4.10, "2023-07-14":4.75, "2023-10-13":4.33, "2024-01-12":3.05,
        "2024-04-12":4.44, "2024-07-12":6.12, "2024-10-11":4.37, "2025-01-15":4.82,
    },

    "KO": {
        "2020-02-20":0.47, "2020-04-21":0.64, "2020-07-21":0.41, "2020-10-22":0.40,
        "2021-02-10":0.34, "2021-04-19":0.52, "2021-07-21":0.61, "2021-10-27":0.57,
        "2022-02-10":0.55, "2022-04-25":0.64, "2022-07-26":0.44, "2022-10-26":0.65,
        "2023-02-14":0.46, "2023-04-24":0.72, "2023-07-26":0.59, "2023-10-23":0.71,
        "2024-02-13":0.45, "2024-04-30":0.74, "2024-07-23":0.56, "2024-10-23":0.66,
        "2025-02-05":0.50,
    },

    # JNJ Q3 2023 ($10.21) reflects a one-time talc litigation settlement gain.
    # This inflates TTM EPS Oct 2023–Oct 2024; noted as a limitation.
    # Q1 2023 (-$0.03) is also a one-time charge.

    "JNJ": {
        "2020-01-22":1.50, "2020-04-14":2.17, "2020-07-16":1.36, "2020-10-13":1.33,
        "2021-01-26":0.65, "2021-04-20":2.32, "2021-07-21":2.35, "2021-10-19":1.37,
        "2022-01-25":1.77, "2022-04-19":1.93, "2022-07-19":1.80, "2022-10-18":1.68,
        "2023-01-24":1.32, "2023-04-18":-0.03, "2023-07-20":1.96, "2023-10-17":10.21,
        "2024-01-23":1.58, "2024-04-16":1.34, "2024-07-17":1.93, "2024-10-15":1.11,
        "2025-01-22":1.41,
    },
}

SECTORS = {
    "AAPL": "Tech", "MSFT": "Tech", "XOM": "Energy",
    "JPM":  "Finance", "KO": "Staples", "JNJ": "Healthcare",
}

STRATEGY_NAMES = ["Price_vs_MA20", "MA20_vs_MA50", "MA10_vs_MA20", "MA5_vs_MA20"]


# -- Core functions-------------------------------------------------------------

# Import daily stock data from Yahoo Finance.

def load_price_data(ticker):
    df = yf.download(ticker, start=START_DATE, end=END_DATE,
                     auto_adjust=True, progress=False)
    df.columns = df.columns.get_level_values(0)
    return df


def add_signals(df):
    for w in MA_WINDOWS:
        df[f"MA_{w}"] = df["Close"].rolling(w).mean()

    # Signal: +1 = long, -1 = short, 0 = flat.

    df["sig_price_ma20"] = np.where(df["Close"]   > df["MA_20"], 1,
                           np.where(df["Close"]   < df["MA_20"], -1, 0))
    df["sig_ma20_ma50"]  = np.where(df["MA_20"]   > df["MA_50"], 1,
                           np.where(df["MA_20"]   < df["MA_50"], -1, 0))
    df["sig_ma10_ma20"]  = np.where(df["MA_10"]   > df["MA_20"], 1,
                           np.where(df["MA_10"]   < df["MA_20"], -1, 0))
    df["sig_ma5_ma20"]   = np.where(df["MA_5"]    > df["MA_20"], 1,
                           np.where(df["MA_5"]    < df["MA_20"], -1, 0))
    return df


def backtest(df, signal_col):

    cash, shares, values = START_CASH, 0.0, []
    for i in range(len(df)):
        price = float(df["Close"].iloc[i])
        sig   = int(df[signal_col].iloc[i])
        if sig == 1 and shares == 0 and cash > 0:
            shares = (cash * (1 - FEE_RATE)) / price
            cash   = 0.0
        elif sig == -1 and shares > 0:
            cash   = shares * price * (1 - FEE_RATE)
            shares = 0.0
        values.append(cash + shares * price)
    return values


def build_ttm_pe(df, eps_dict):

    eps_s = pd.Series(eps_dict)
    eps_s.index = pd.to_datetime(eps_s.index)
    eps_ttm = eps_s.sort_index().rolling(4).sum().dropna()

    # Reindex quarterly EPS to daily, filling each daily value until the next report.

    combined    = df.index.union(eps_ttm.index)
    eps_daily   = eps_ttm.reindex(combined).ffill()
    df["ttm_eps"] = eps_daily.reindex(df.index)
    df["pe"]      = np.where(df["ttm_eps"] > 0,
                             df["Close"] / df["ttm_eps"], np.nan)
    return df


def regime_analysis(df, split="median"):
    valid_pe = df["pe"].dropna()
    if split == "median":
        lo_thr = hi_thr = valid_pe.median()
        df["regime"] = np.where(df["pe"] >= lo_thr, "high",
                       np.where(df["pe"].isna(), "undef", "low"))
    else:  # Low/high quartiles.
        lo_thr = valid_pe.quantile(0.25)
        hi_thr = valid_pe.quantile(0.75)
        df["regime"] = np.where(df["pe"] >= hi_thr, "high",
                       np.where(df["pe"] <= lo_thr, "low", "mid"))

    bh_low  = df[df["regime"] == "low" ]["ret_bh"].dropna().mean()
    bh_high = df[df["regime"] == "high"]["ret_bh"].dropna().mean()

    rows = []
    sig_cols = {"Price_vs_MA20":"ret_price_ma20", "MA20_vs_MA50":"ret_ma20_ma50",
                "MA10_vs_MA20":"ret_ma10_ma20",   "MA5_vs_MA20":"ret_ma5_ma20"}

    for name, col in sig_cols.items():
        low  = df[df["regime"] == "low" ][col].dropna()
        high = df[df["regime"] == "high"][col].dropna()
        if len(low) < 30 or len(high) < 30:
            continue

        _, p_val    = stats.ttest_ind(low, high, equal_var=False)
        pooled_std  = np.sqrt((low.std()**2 + high.std()**2) / 2)
        cohens_d    = abs(low.mean() - high.mean()) / pooled_std if pooled_std else 0

        low_edge  = (low.mean()  - bh_low)  * 100
        high_edge = (high.mean() - bh_high) * 100

        rows.append({
            "strategy":    name,
            "low_edge_%":  round(low_edge,  4),
            "high_edge_%": round(high_edge, 4),
            "p_value":     round(p_val, 4),
            "sig_p05":     p_val < 0.05,
            "cohens_d":    round(cohens_d, 3),
            "hyp_dir":     low_edge > high_edge,   # True = supports hypothesis.
        })

    return pd.DataFrame(rows), lo_thr, hi_thr


# -- Full system, per-ticker ---------------------------------------------------

def run_ticker(ticker, eps_dict, plot=False):
    df = load_price_data(ticker)
    df = add_signals(df)

    sig_map = {"price_ma20":"sig_price_ma20", "ma20_ma50":"sig_ma20_ma50",
               "ma10_ma20":"sig_ma10_ma20",   "ma5_ma20":"sig_ma5_ma20"}
    for label, sig_col in sig_map.items():
        df[f"val_{label}"] = backtest(df, sig_col)
    df["val_bh"] = (START_CASH / float(df["Close"].iloc[0])) * df["Close"]

    for label in list(sig_map.keys()) + ["bh"]:
        df[f"ret_{label}"] = df[f"val_{label}"].pct_change()

    df = build_ttm_pe(df, eps_dict)

    res_median,   lo_med,  hi_med  = regime_analysis(df, split="median")
    res_quartile, lo_q25,  hi_q75  = regime_analysis(df, split="quartile")

    valid_pe = df["pe"].dropna()
    print(f"\n{'='*68}")
    print(f"  {ticker}  [{SECTORS[ticker]}]"
          f"  |  P/E {valid_pe.min():.1f}–{valid_pe.max():.1f}"
          f"  median {valid_pe.median():.1f}"
          f"  |  {len(valid_pe)} valid days")
    print(f"{'='*68}")

    for label, res, lo, hi in [
        (f"Median split  (p50={lo_med:.1f})",     res_median,   lo_med, hi_med),
        (f"Quartile split (p25={lo_q25:.1f}, p75={hi_q75:.1f})", res_quartile, lo_q25, hi_q75),
    ]:
        print(f"\n  {label}")
        print(f"  {'Strategy':<18} {'Low edge':>10} {'High edge':>11}"
              f" {'p-value':>9} {'Sig?':>5} {'d':>6} {'Dir?':>5}")
        print(f"  {'-'*66}")
        for _, row in res.iterrows():
            sig   = "YES" if row["sig_p05"]  else "no"
            hyp   = "[+]" if row["hyp_dir"]  else "[x]"
            print(f"  {row['strategy']:<18}"
                  f" {row['low_edge_%']:>9.4f}%"
                  f" {row['high_edge_%']:>10.4f}%"
                  f" {row['p_value']:>9.4f}"
                  f" {sig:>5}"
                  f" {row['cohens_d']:>6.3f}"
                  f" {hyp:>5}")

    if plot:
        _plot_ticker(df, ticker)

    return df, res_median, res_quartile

# Plots price shaded by P/E regime, and P/E over time.
def _plot_ticker(df, ticker):
    valid_pe = df["pe"].dropna()
    median_pe = valid_pe.median()
    df["regime_plot"] = np.where(df["pe"] >= median_pe, "high",
                        np.where(df["pe"].isna(), "undef", "low"))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    lo  = df["regime_plot"] == "low"
    hi  = df["regime_plot"] == "high"
    y_lo, y_hi = df["Close"].min(), df["Close"].max()

    ax1.fill_between(df.index, y_lo, y_hi, where=lo,  alpha=0.15, color="green")
    ax1.fill_between(df.index, y_lo, y_hi, where=hi,  alpha=0.15, color="red")
    ax1.plot(df.index, df["Close"], color="black", linewidth=0.8)
    ax1.set_title(f"{ticker} — green = low P/E, red = high P/E")

    ax2.plot(df.index, df["pe"], color="purple", linewidth=0.8)
    ax2.axhline(median_pe, color="black", linestyle="--",
                linewidth=1, label=f"Median ({median_pe:.1f})")
    ax2.set_title(f"{ticker} TTM P/E Ratio")
    ax2.set_ylabel("P/E")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"{ticker}_pe_regime.png", dpi=150, bbox_inches="tight")
    plt.show()


# -- Ticker summary ------------------------------------------------------------

# Prints a concise comparison table.
# [+] = significant (p < .05) & correct direction (low P/E edge > high P/E edge).
# [~] = correct direction only
# [x] = wrong direction
def print_summary(all_results):
    print(f"\n\n{'='*70}")
    print("  Ticker SUMMARY  (median split)")
    print("  [+] = p<0.05 + correct dir   [~] = correct dir only   [x] = wrong dir")
    print(f"{'='*70}")
    print(f"  {'Ticker':<8} {'Sector':<10}"
          f" {'Pr>MA20':>9} {'MA20>50':>9} {'MA10>20':>9} {'MA5>20':>9}")
    print(f"  {'-'*58}")

    for ticker, (_, res_med, _) in all_results.items():
        row_str = f"  {ticker:<8} {SECTORS[ticker]:<10}"
        for sname in STRATEGY_NAMES:
            match = res_med[res_med["strategy"] == sname]
            if match.empty:
                row_str += f"{'?':>9}"
                continue
            r = match.iloc[0]
            if r["sig_p05"] and r["hyp_dir"]:
                mark = "[+]"
            elif r["hyp_dir"]:
                mark = "[~]"
            else:
                mark = "[x]"
            row_str += f"{mark:>9}"
        print(row_str)


# -- Begin testing -------------------------------------------------------------

if __name__ == "__main__":
    all_results = {}
    for ticker, eps_dict in EPS.items():
        all_results[ticker] = run_ticker(ticker, eps_dict, plot=True)

    print_summary(all_results)
