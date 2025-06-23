# Multi-Factor Momentum Trading Strategy

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/tanish35/Momentum-Investing)

## Overview

This repository contains a sophisticated trading strategy implemented in Python using the `backtrader` framework. The strategy is designed to achieve robust portfolio performance by integrating multiple modern quantitative concepts. It combines a top-down market regime filter with a bottom-up, multi-factor stock selection process, and uses a risk-parity approach for capital allocation.

The strategy is long-only and dynamically rebalances its holdings based on changes in the top-ranked stocks, ensuring it adapts to evolving market conditions.

## Key Ideas and Concepts

The strategy is built upon a foundation of several key quantitative finance ideas:

1. **Market Regime Filter**: The strategy first assesses the overall market health using a simple but effective filter.

   - **Mechanism**: It checks if the market index (e.g., SPY) is trading above its 200-day Simple Moving Average (SMA).
   - **Action**: Long positions are only initiated or held if the market is in a bullish regime (price > 200-day SMA). If the market enters a bearish regime, all positions are exited to mitigate systemic risk.

2. **Time-Series Momentum (TSMOM)**: This is a trend-following filter applied to each individual stock.

   - **Mechanism**: A stock is only considered a candidate for investment if it is trading above its own 200-day SMA.

3. **Cross-Sectional Momentum**: After filtering for trend, stocks are ranked against each other based on their momentum.

   - **Mechanism**: Momentum is calculated over multiple lookback periods (60, 120, and 252 days) and then averaged to produce a robust momentum score.

4. **Frog-in-the-Pan (FIP) Score**: This factor measures the _quality and consistency_ of a stock's trend.

   - **Mechanism**: It calculates the ratio of positive daily returns over a 252-day period. A higher score indicates a smoother, more consistent uptrend, rather than one driven by a few large, volatile spikes.

5. **Skewness Factor**: To mitigate the risk of holding stocks prone to sudden crashes, a skewness factor is included.

   - **Mechanism**: It calculates the rolling skewness of stock returns over a 90-day window. A high negative skew suggests a stock has a "tail" of large negative returns and is thus penalized in its final score.

6. **Inverse Volatility Weighting**: This is a risk parity approach to portfolio construction.

   - **Mechanism**: Once the top `N` stocks are selected, portfolio capital is allocated inversely proportional to each stock's historical volatility (standard deviation over 126 days). Less volatile stocks receive a larger capital allocation, and more volatile stocks receive a smaller one, with the goal of equalizing the risk contribution of each position.

## Important Formulas

- **Daily Return:**  
  `rt = (Pt / P_{t-1}) - 1`

- **Momentum Score (Average over Lookbacks):**  
  `Momentum = (1/N) * Σ Momentum_lookback_i`

- **Frog-in-the-Pan (FIP) Score:**  
  `FIP Score = (Number of Positive Daily Returns) / T`

- **Rolling Skewness:**  
  `Skewness_t = skewness(r_{t−τ+1}, ..., r_t)`

- **Combined Ranking Score:**  
  `Score = (w_m * Momentum) + (w_f * FIP) - (w_s * Skewness)`

- **Inverse Volatility Weighting:**  
  `w_i = (1 / σ_i) / Σ (1 / σ_j)`

## Performance Summary

Over a 2-year backtest period, the strategy demonstrated strong growth in both held positions and realized trades:

- **Unrealized Profits (Total Portfolio Growth):** Up by **89.71%**
- **Realized Profits (from Closed Trades):** Up by **73.93%**

## Usage

The strategy is implemented as a Python script using the `backtrader` library.

### Requirements

- Python 3.x
- backtrader
- pandas
- numpy
- scipy
- yfinance

### How to Run

1. Ensure all required Python libraries are installed:

   ```bash
   pip install backtrader pandas numpy scipy yfinance
   ```

2. Execute the main Python script from your terminal:

   ```bash
   python main.py
   ```

3. Upon completion, the script will generate a detailed Excel report named `trading_results_final.xlsx` containing multiple sheets for in-depth analysis.

## Notes

- **Dynamic Rebalancing**: The strategy does not use a fixed rebalancing schedule. Instead, it rebalances the portfolio dynamically whenever the composition of the top `N` ranked stocks changes.

- **Indicator Warm-up**: The strategy relies on several long-term indicators (e.g., 200-day SMA, 252-day FIP). As a result, it requires a significant "warm-up" period (at least one year of data) before it can generate reliable signals and begin trading.
