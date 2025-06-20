import backtrader as bt
import numpy as np
from scipy.stats import skew
import pandas as pd
import yfinance as yf


class RollingSkewness(bt.Indicator):
    lines = ("skewness",)
    params = (("period", 90),)

    def __init__(self):
        self.addminperiod(self.p.period)

    def next(self):
        data_slice = np.array(self.data.get(size=self.p.period))
        returns = np.diff(np.log(data_slice))
        self.lines.skewness[0] = skew(returns)


class FrogInThePan(bt.Indicator):
    lines = ("fip_score",)
    params = (("period", 252),)

    def __init__(self):
        daily_ret = self.data.close / self.data.close(-1) - 1
        is_positive = daily_ret > 0
        # Return was positive how many days
        positive_days_sum = bt.indicators.SumN(is_positive, period=self.p.period)
        # Ratio of positive return days/total days
        self.lines.fip_score = positive_days_sum / self.p.period


class NewMom(bt.Strategy):
    params = dict(
        lookbacks=[60, 120, 252],
        top_n=5,
        vol_lookback=126,
        skewness_lookback=90,
        fip_lookback=252,
        ts_mom_lookback=200,
        regime_ma_period=200,
        momentum_weight=0.5,
        fip_weight=0.5,
        skewness_penalty=0.5,
    )

    def __init__(self):
        self.market = self.datas[0]
        self.stocks = self.datas[1:]

        self.stock_daily_data = {}
        self.portfolio_daily_values = []
        self.order_history = []

        # Checks SMA for 200 days(Market)
        self.regime_ma = bt.indicators.SimpleMovingAverage(
            self.market.close, period=self.p.regime_ma_period
        )

        self.indicators = {}
        for d in self.stocks:
            self.indicators[d._name] = {
                # Check momentum over multiple periods
                "momentum": [
                    bt.indicators.Momentum(d.close, period=lb)
                    for lb in self.p.lookbacks
                ],
                # Check volaitlity for 6 months
                "volatility": bt.indicators.StdDev(d.close, period=self.p.vol_lookback),
                # Check for 4.5 months
                "skewness": RollingSkewness(d, period=self.p.skewness_lookback),
                # Check for 1 year
                "fip": FrogInThePan(d, period=self.p.fip_lookback),
                # Check SMA for 200 days(Stocks)
                "ts_mom": bt.indicators.SimpleMovingAverage(
                    d.close, period=self.p.ts_mom_lookback
                ),
            }
            self.stock_daily_data[d._name] = {
                "dates": [],
                "prices": [],
                "values": [],
            }
        self.last_rebalanced_stocks = []
        max_overall_lookback = max(
            self.p.regime_ma_period,
            max(self.p.lookbacks),
            self.p.vol_lookback,
            self.p.skewness_lookback,
            self.p.fip_lookback,
            self.p.ts_mom_lookback,
        )
        self.addminperiod(max_overall_lookback)

    def notify_order(self, order):
        if order.status in [order.Completed]:
            order_record = {
                "datetime": self.datetime.date(),
                "stock": order.data._name,
                "action": "buy" if order.isbuy() else "sell",
                "price": order.executed.price,
                "size": order.executed.size,
                "value": order.executed.value,
                "commission": order.executed.comm,
                "profit": order.executed.pnl if hasattr(order.executed, "pnl") else 0,
            }
            self.order_history.append(order_record)

    def notify_trade(self, trade):
        # Round trip trade data not required
        pass

    def next(self):
        current_date = self.datetime.date()

        # Today's portfolio value
        total_portfolio_value = self.broker.getvalue()
        self.portfolio_daily_values.append(
            {"date": current_date, "total_value": total_portfolio_value}
        )

        # Stock values saved(used in trade sheets)
        for d in self.stocks:
            position = self.getposition(d)
            self.stock_daily_data[d._name]["dates"].append(current_date)
            self.stock_daily_data[d._name]["prices"].append(d.close[0])
            self.stock_daily_data[d._name]["values"].append(
                position.size * d.close[0] if position.size > 0 else 0
            )

        # if not (self.regime_ma[0] == self.regime_ma[0]):
        #     return

        # Bearish market(Sell all stocks)
        if self.market.close[0] < self.regime_ma[0]:
            for d in self.stocks:
                if self.getposition(d).size:
                    self.close(data=d)
                if self.last_rebalanced_stocks:
                    self.last_rebalanced_stocks = []
            return

        scores = []
        for d in self.stocks:
            min_len = max(
                self.p.regime_ma_period, self.p.fip_lookback, self.p.ts_mom_lookback
            )
            if len(d) < min_len:
                continue

            # Checks if bullish stock(Greater than 200MA)
            is_in_uptrend = d.close[0] > self.indicators[d._name]["ts_mom"][0]
            if not is_in_uptrend:
                continue

            # Mean of all 3 momentum periods
            mom_score = np.mean([m[0] for m in self.indicators[d._name]["momentum"]])
            if mom_score <= 0:
                continue

            # Frog in pan strat
            fip_score = self.indicators[d._name]["fip"].fip_score[0]
            # Skewness strat(Positive skewness is better)
            skew_score = self.indicators[d._name]["skewness"].skewness[0]

            # 0.5*Mom+0.5*FIP+0.5*Skew
            combined_score = (
                (self.p.momentum_weight * mom_score)
                + (self.p.fip_weight * fip_score)
                + (self.p.skewness_penalty * skew_score)
            )
            scores.append({"data": d, "score": combined_score})

        # Sorting stocks on scores
        scores.sort(key=lambda x: x["score"], reverse=True)
        new_top_stocks = [x["data"] for x in scores[: self.p.top_n]]
        new_top_stock_names = sorted([d._name for d in new_top_stocks])
        last_rebalanced_stock_names = sorted(
            [d._name for d in self.last_rebalanced_stocks]
        )
        # Top stocks not changed then exit the next
        if new_top_stock_names == last_rebalanced_stock_names:
            return
        self.last_rebalanced_stocks = new_top_stocks
        longs = [x["data"] for x in scores[: self.p.top_n]]

        # Enter into the stocks which are now in top 5
        current_positions = [
            d
            for d, pos in self.broker.positions.items()
            if pos.size != 0 and d in self.stocks
        ]
        # Exit stocks which are not in top 5
        exits = [d for d in current_positions if d not in longs]
        for d in exits:
            self.close(data=d)

        # If no longs exit
        if not longs:
            # self.last_rebalanced_stocks = []
            return

        total_value = self.broker.getvalue()
        # Volatality of each top stock
        vols = [self.indicators[d._name]["volatility"][0] for d in longs]
        # Inverse volatility
        inv_vols = [1 / v if v > 0 else 0 for v in vols]
        total_inv_vol = sum(inv_vols)

        for d, inv_v in zip(longs, inv_vols):
            if total_inv_vol > 0 and inv_v > 0:
                # Higher weight to lesser volatile stocks and vice-versa
                target_value = total_value * (inv_v / total_inv_vol)
                self.order_target_value(data=d, target=target_value)


def get_backtest_summary_metrics(strategy):
    total_return_analyzer = strategy.analyzers.returns.get_analysis()
    max_drawdown_analyzer = strategy.analyzers.drawdown.get_analysis()
    sharpe_ratio_analyzer = strategy.analyzers.sharpe.get_analysis()

    total_return = total_return_analyzer["rnorm100"] if total_return_analyzer else 0
    max_drawdown_percent = (
        max_drawdown_analyzer["max"]["drawdown"]
        if max_drawdown_analyzer and "max" in max_drawdown_analyzer
        else 0
    )
    sharpe_ratio = (
        sharpe_ratio_analyzer["sharperatio"] if sharpe_ratio_analyzer else "N/A"
    )

    metrics = {
        "Metric": ["Total Return (%)", "Max Drawdown (%)", "Sharpe Ratio"],
        "Value": [
            f"{total_return:.2f}",
            f"{max_drawdown_percent:.2f}",
            f"{sharpe_ratio:.2f}"
            if isinstance(sharpe_ratio, (int, float))
            else sharpe_ratio,
        ],
    }
    return pd.DataFrame(metrics)


def generate_reports(strategy):
    # Summary of all entire backtest
    summary_metrics_df = get_backtest_summary_metrics(strategy)

    # All buy/sell orders
    transaction_log_df = pd.DataFrame(strategy.order_history)
    if not transaction_log_df.empty:
        transaction_log_df["datetime"] = pd.to_datetime(transaction_log_df["datetime"])
        transaction_log_df.sort_values(by=["datetime", "stock"], inplace=True)
        transaction_log_df["datetime"] = transaction_log_df["datetime"].dt.strftime(
            "%Y-%m-%d"
        )
        transaction_log_df["profit"] = transaction_log_df["profit"].apply(
            lambda x: f"{x:.2f}"
        )

    if not strategy.portfolio_daily_values:
        print("no portfolio records")
        return transaction_log_df, pd.DataFrame(), pd.DataFrame(), summary_metrics_df

    # Get date and portfolio value
    portfolio_values_df = pd.DataFrame(strategy.portfolio_daily_values)
    portfolio_values_df["date"] = pd.to_datetime(portfolio_values_df["date"])
    portfolio_values_df.set_index("date", inplace=True)

    all_stocks_daily_dfs = []
    # Get date,price,invested value of the each stock
    for stock_name, daily_data in strategy.stock_daily_data.items():
        if daily_data["dates"]:
            df = pd.DataFrame(daily_data)
            df["stock"] = stock_name
            all_stocks_daily_dfs.append(df)

    if not all_stocks_daily_dfs:
        print("No stock records")
        return transaction_log_df, pd.DataFrame(), pd.DataFrame(), summary_metrics_df

    # Multiple df's combined into 1 df
    combined_daily_df = pd.concat(all_stocks_daily_dfs)
    combined_daily_df["dates"] = pd.to_datetime(combined_daily_df["dates"])
    combined_daily_df.set_index("dates", inplace=True)

    # Merging this with portfolio values
    merged_df = combined_daily_df.join(portfolio_values_df)
    merged_df["weightage"] = (merged_df["values"] / merged_df["total_value"]).fillna(0)

    # Monthly weight per stock from daily values
    monthly_weights_grouped = (
        merged_df.groupby("stock").resample("M")["weightage"].mean().reset_index()
    )
    # Year and months added
    monthly_weights_grouped["year"] = monthly_weights_grouped["dates"].dt.year
    monthly_weights_grouped["month"] = monthly_weights_grouped["dates"].dt.month

    # Moving stocks from vertical to horizontal
    stock_weights_pivot = (
        monthly_weights_grouped.pivot_table(
            index=["year", "month"], columns="stock", values="weightage"
        )
        .reset_index()
        .fillna(0)
    )

    # Get daily portfolio returns
    time_returns = strategy.analyzers.timereturn.get_analysis()
    returns_series = pd.Series(time_returns, name="Portfolio_Returns")
    returns_series.index = pd.to_datetime(returns_series.index)
    # Daily to monthly
    portfolio_monthly_returns = (
        returns_series.resample("M").apply(lambda x: (1 + x).prod() - 1).reset_index()
    )
    portfolio_monthly_returns.rename(columns={"index": "date"}, inplace=True)
    portfolio_monthly_returns["Year"] = portfolio_monthly_returns["date"].dt.year
    portfolio_monthly_returns["Month"] = portfolio_monthly_returns["date"].dt.month

    # Merge portfolio returns with stock weights
    portfolio_weights_df = (
        pd.merge(
            portfolio_monthly_returns[["Year", "Month", "Portfolio_Returns"]],
            stock_weights_pivot,
            left_on=["Year", "Month"],
            right_on=["year", "month"],
            how="left",
        )
        .drop(columns=["year", "month"], errors="ignore")
        .fillna(0)
    )

    # Monthly stock performance
    monthly_performance_data = []
    # Daily stock return percentages
    merged_df["returns"] = merged_df.groupby("stock")["prices"].pct_change().fillna(0)

    for stock_name, stock_daily_df in merged_df.groupby("stock"):
        held_stock_df = stock_daily_df[stock_daily_df["values"] > 0]
        if held_stock_df.empty:
            continue

        # Calculates cumulative return from start of month(Cumulative daily returns)
        stock_monthly_returns_series = (
            held_stock_df["returns"].resample("M").apply(lambda x: (1 + x).prod() - 1)
        )
        # Calculates since the first time I entered the position(Compounded)
        cumulative_daily_returns_stock = (1 + held_stock_df["returns"]).cumprod() - 1
        # Cumulative average monthly returns
        mean_monthly_returns_stock_expanding = (
            stock_monthly_returns_series.expanding().mean()
        )

        # Monthly return of each stock
        for date_idx, monthly_ret_val in stock_monthly_returns_series.items():
            # Days when stock was held filtered
            month_df_filtered = held_stock_df.loc[
                held_stock_df.index.to_period("M") == date_idx.to_period("M")
            ]

            if month_df_filtered.empty:
                continue

            # Closing price at start
            first_price = month_df_filtered["prices"].iloc[0]
            # Closing price at end
            last_price = month_df_filtered["prices"].iloc[-1]
            # Max Price in month
            max_price = month_df_filtered["prices"].max()

            monthly_return_val = monthly_ret_val
            # Cumulative returns from when the position has been held
            cum_ret_val = (
                cumulative_daily_returns_stock.loc[date_idx]
                if date_idx in cumulative_daily_returns_stock.index
                else np.nan
            )
            # Average monthly returns
            mean_monthly_return_val = mean_monthly_returns_stock_expanding.loc[date_idx]
            # SD of current month
            monthly_volatility_val = month_df_filtered["returns"].std() * np.sqrt(21)

            # Profit from start to end of the month
            profit_percent = (last_price / first_price - 1) if first_price != 0 else 0
            # Max peak in current month-End of month price
            drawdown_percent = (
                (last_price - max_price) / max_price if max_price != 0 else 0
            )
            # Gain in price from start of month to peak
            upside_percent = (
                (max_price - first_price) / first_price if first_price != 0 else 0
            )
            # Simplified monthly sharpe ratio
            risk_adj_return = (
                mean_monthly_return_val / monthly_volatility_val
                if monthly_volatility_val > 0
                else 0
            )

            record = {
                "year": date_idx.year,
                "month": date_idx.month,
                "stock": stock_name,
                "position": "buy",
                "buy": first_price,
                "sell": last_price,
                "profit": profit_percent,
                "drawdown": drawdown_percent,
                "upside": upside_percent,
                "Monthly_Return": monthly_return_val,
                "Cumulative_Return": cum_ret_val,
                "Mean_Monthly_Return": mean_monthly_return_val,
                "Monthly_Volatility": monthly_volatility_val,
                "Risk_Adjusted_Mean_Return": risk_adj_return,
            }
            monthly_performance_data.append(record)

    monthly_stock_performance_df = pd.DataFrame(monthly_performance_data)

    if not monthly_stock_performance_df.empty:
        monthly_stock_performance_df.sort_values(
            by=["year", "month", "stock"], inplace=True
        )
        for col in ["profit", "drawdown", "upside"]:
            monthly_stock_performance_df[col] = monthly_stock_performance_df[col].apply(
                lambda x: f"{x:.2%}" if pd.notna(x) else "N/A"
            )
        for col in [
            "Monthly_Return",
            "Cumulative_Return",
            "Mean_Monthly_Return",
            "Monthly_Volatility",
            "Risk_Adjusted_Mean_Return",
        ]:
            monthly_stock_performance_df[col] = monthly_stock_performance_df[col].apply(
                lambda x: f"{x:.9f}" if pd.notna(x) else "N/A"
            )

    return (
        transaction_log_df,
        monthly_stock_performance_df,
        portfolio_weights_df,
        summary_metrics_df,
    )


def save_reports_to_excel(
    transaction_log_df,
    monthly_stock_performance_df,
    portfolio_weights_df,
    summary_metrics_df,
):
    with pd.ExcelWriter("trading_results_final.xlsx", engine="openpyxl") as writer:
        if not summary_metrics_df.empty:
            summary_metrics_df.to_excel(
                writer, sheet_name="Backtest Summary", index=False
            )
        if not portfolio_weights_df.empty:
            portfolio_weights_df.to_excel(
                writer, sheet_name="Overall Portfolio Sheet", index=False
            )

        if not transaction_log_df.empty:
            transaction_log_df.to_excel(
                writer, sheet_name="Transaction Log", index=False
            )

        if not monthly_stock_performance_df.empty:
            monthly_stock_performance_df.to_excel(
                writer, sheet_name="Trades Sheet", index=False
            )

            for stock_name, stock_df in monthly_stock_performance_df.groupby("stock"):
                sheet_name = stock_name[:31]
                stock_df.to_excel(writer, sheet_name=sheet_name, index=False)
    print("All reports saved to 'trading_results_final.xlsx'")


tickers = [
    "MSFT",
    "AAPL",
    "NVDA",
    "GOOGL",
    "AMZN",
    "META",
    "AVGO",
    "TSLA",
    "LLY",
    "UNH",
    "JNJ",
    "MRK",
    "ABBV",
    "PFE",
    "TMO",
    "DHR",
    "MDT",
    "BRK-B",
    "JPM",
    "V",
    "MA",
    "BAC",
    "WFC",
    "GS",
    "BLK",
    "AXP",
    "COST",
    "WMT",
    "HD",
    "PG",
    "KO",
    "PEP",
    "MCD",
    "NKE",
    "SBUX",
    "LOW",
    "XOM",
    "CVX",
    "CAT",
    "UNP",
    "GE",
    "BA",
    "LMT",
    "DE",
    "SMCI",
    "TSM",
    "ORCL",
    "ADBE",
    "CRM",
    "AMD",
    "INTC",
    "QCOM",
    "IBM",
    "LIN",
    "NFLX",
    "DIS",
    "VZ",
    "CMCSA",
    "ABT",
    "ACN",
    "CSCO",
    "TMUS",
    "TXN",
    "HON",
    "GILD",
    "BKNG",
    "C",
    "SPG",
    "PLD",
    "AMT",
    "EQIX",
    "NOW",
    "PLTR",
    "UBER",
    "PYPL",
    "SPY",
    "QQQ",
    "VTI",
    "DIA",
    "IWM",
    "XLK",
    "XLV",
    "XLF",
    "XLY",
    "XLC",
    "XLE",
    "XLI",
    "XLP",
    "XLB",
    "XLU",
    "XLRE",
    "VEA",
    "VWO",
    "EFA",
    "EWJ",
    "EWG",
    "INDA",
    "MCHI",
    "EEM",
    "ACWI",
    "MTUM",
    "QUAL",
    "USMV",
    "VLUE",
    "VIG",
    "SOXX",
    "HACK",
    "ICLN",
    "BOTZ",
    "ARKK",
    "TAN",
    "IBB",
    "FDN",
    "PAVE",
    "URA",
    "AGG",
    "TLT",
    "LQD",
    "HYG",
    "SHY",
    "GLD",
    "SLV",
    "DBC",
    "VNQ",
    "USO",
    "BSE.NS",
    "NIFTYBEES.NS",
    "JUNIORBEES.NS",
    "BANKBEES.NS",
    "LIQUIDBEES.NS",
    "GOLDBEES.NS",
    "FEZ",
    "FXI",
]


# Trading starts from 2023 but backtrader warms up the data from 2022
start_date = "2022-01-01"
end_date = "2024-12-31"

print("Downloading all ticker data...")
all_data_raw = yf.download(tickers, start=start_date, end=end_date, group_by="ticker")

cerebro = bt.Cerebro()

print("Preparing and adding data feeds...")
spy_df = all_data_raw.get("SPY", pd.DataFrame()).dropna()
if not spy_df.empty:
    cerebro.adddata(bt.feeds.PandasData(dataname=spy_df, name="SPY"))
else:
    print("SPY data is missing or empty. The regime filter will not work. Exiting.")
    exit()

for ticker in tickers:
    if ticker == "SPY":
        continue
    df = all_data_raw.get(ticker, pd.DataFrame()).dropna()
    if not df.empty:
        cerebro.adddata(bt.feeds.PandasData(dataname=df, name=ticker))
    else:
        print(f"  Skipping {ticker} due to no data.")

cerebro.addstrategy(NewMom)
cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn")
cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")

cerebro.broker.setcash(100000.0)

print("Running backtest...")
results = cerebro.run()
strategy = results[0]

print("Generating comprehensive reports...")
(
    transaction_log_df,
    monthly_stock_performance_df,
    portfolio_weights_df,
    summary_metrics_df,
) = generate_reports(strategy)

if all(
    df.empty
    for df in [
        transaction_log_df,
        monthly_stock_performance_df,
        portfolio_weights_df,
        summary_metrics_df,
    ]
):
    print("No data was generated to create an Excel file.")
else:
    save_reports_to_excel(
        transaction_log_df,
        monthly_stock_performance_df,
        portfolio_weights_df,
        summary_metrics_df,
    )

final_value = cerebro.broker.getvalue()
print(f"\nStarting Portfolio Value: ₹1,00,000.00")
print(f"Final Portfolio Value: ₹{final_value:,.2f}")

# Currently unrealized profits: Rs. 89,717 in 2 years, a return of 89.71% in 2 years
# Realized profit is Rs. 73,931
