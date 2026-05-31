# Moving Average Strategy Effectiveness Across P/E Regimes
### Evidence from Six U.S. Equities (2020–2025)

**Author:** Reid Fintak  
**Preprint:** *LINK_PLACEHOLDER*

---

## Overview

This repository contains the analysis code and paper for a study testing whether a stock's trailing P/E ratio regime (high vs. low) predicts whether moving average crossover strategies outperform a passive buy-and-hold benchmark. Six U.S. equities across five sectors are examined over a five-year window (January 2020 – January 2025).

## Research Question

Does a stock's P/E ratio regime predict whether moving average strategies outperform buy-and-hold?

## Methodology

- **Stocks:** AAPL, MSFT (Technology), XOM (Energy), JPM (Financial Services), KO (Consumer Staples), JNJ (Healthcare)
- **Strategies:** Four MA crossover rules — Price vs. MA20, MA20 vs. MA50, MA10 vs. MA20, MA5 vs. MA20
- **Regimes:** Days classified as high/low P/E via median split (primary) and 25th/75th percentile split (robustness check)
- **Statistics:** Welch's t-test for significance; Cohen's d for effect size
- **EPS data:** GAAP diluted, sourced from SEC 8-K filings, recorded at public report dates (not quarter-end dates) to prevent lookahead bias

## Repository Contents

```
├── ma_pe_analysis.py          # Full analysis script
├── ma_pe_research_paper.pdf   # Paper (also on SSRN — see link above)
├── requirements.txt
└── README.md
```

## Requirements

Python 3.8 or later.

```bash
pip install -r requirements.txt
```

## Running the Code

```bash
python ma_pe_analysis.py
```

Price data downloads automatically via `yfinance` — no separate data files needed.

**Output includes:**
- Per-stock results tables for both median and quartile splits
- Cross-stock summary table
- Price/P/E regime charts saved as `<TICKER>_pe_regime.png`

## Data Notes

**Price data** is retrieved at runtime from Yahoo Finance via `yfinance`.

**EPS data** is hardcoded in the `EPS` dictionary in `ma_pe_analysis.py`. All values are quarterly GAAP diluted EPS, recorded at their SEC 8-K report dates rather than fiscal quarter-end dates. Using report dates is essential to prevent lookahead bias — no earnings information is assigned to a trading day before it was publicly available.

Two known data anomalies are retained as reported and discussed in the paper's Limitations section:
- **XOM Q4 2020** (`-$4.70`): Large non-cash write-down causes negative TTM EPS for several quarters. Days with non-positive TTM EPS are excluded from P/E analysis (P/E is undefined).
- **JNJ Q3 2023** (`$10.21`): One-time talc litigation settlement gain inflates TTM EPS for approximately one year following that report date.

EPS values were cross-verified against [macrotrends.net](https://www.macrotrends.net) historical EPS records.

## License

[MIT License](LICENSE)
