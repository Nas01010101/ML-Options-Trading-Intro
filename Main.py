import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
import ta
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
import os
import warnings

warnings.filterwarnings("ignore")

print("SPY Reversal Trading Algorithm - Timing Opportunities Finder")
print(f"Current working directory: {os.getcwd()}")


def fetch_spy_data(start_date, end_date, interval='1d'):
    """Fetch SPY data from Yahoo Finance."""
    print(f"Fetching SPY data from {start_date} to {end_date} with {interval} interval...")
    spy = yf.download("SPY", start=start_date, end=end_date, interval=interval)
    print(f"Downloaded {len(spy)} data points")
    return spy


def add_technical_indicators(df):
    """Add technical indicators useful for reversal detection."""
    print("Calculating technical indicators...")

    # Make a copy to avoid modifying the original
    data = df.copy()

    # RSI - Relative Strength Index (Oversold/Overbought)
    rsi = RSIIndicator(close=data['Close'], window=14)
    data['RSI'] = rsi.rsi()

    # Stochastic Oscillator
    stoch = StochasticOscillator(high=data['High'], low=data['Low'], close=data['Close'], window=14, smooth_window=3)
    data['Stoch_K'] = stoch.stoch()
    data['Stoch_D'] = stoch.stoch_signal()

    # MACD - Moving Average Convergence Divergence
    macd = MACD(close=data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['MACD_Hist'] = macd.macd_diff()

    # Bollinger Bands
    bollinger = BollingerBands(close=data['Close'], window=20)
    data['BB_Upper'] = bollinger.bollinger_hband()
    data['BB_Middle'] = bollinger.bollinger_mavg()
    data['BB_Lower'] = bollinger.bollinger_lband()
    data['BB_Width'] = bollinger.bollinger_wband()
    data['BB_PCT'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])

    # Moving Averages
    for window in [20, 50, 200]:
        data[f'SMA{window}'] = SMAIndicator(close=data['Close'], window=window).sma_indicator()
        data[f'EMA{window}'] = EMAIndicator(close=data['Close'], window=window).ema_indicator()

    # ATR - Average True Range (Volatility)
    atr = AverageTrueRange(high=data['High'], low=data['Low'], close=data['Close'], window=14)
    data['ATR'] = atr.average_true_range()

    # OBV - On Balance Volume (Volume trend confirmation)
    if 'Volume' in data.columns:
        obv = OnBalanceVolumeIndicator(close=data['Close'], volume=data['Volume'])
        data['OBV'] = obv.on_balance_volume()

        # Add OBV moving average for divergence detection
        data['OBV_EMA20'] = EMAIndicator(close=data['OBV'], window=20).ema_indicator()

    # Rate of Change
    data['ROC_5'] = data['Close'].pct_change(5) * 100
    data['ROC_10'] = data['Close'].pct_change(10) * 100
    data['ROC_20'] = data['Close'].pct_change(20) * 100

    # Calculate Daily Returns
    data['Daily_Return'] = data['Close'].pct_change()

    # Fill any missing values caused by indicators calculation
    data.fillna(method='bfill', inplace=True)

    return data


def identify_reversal_signals(data):
    """Identify potential reversal signals based on technical indicators."""
    print("Identifying potential reversal signals...")

    # Initialize signals column
    data['Reversal_Signal'] = 0  # 0: No signal, 1: Bullish reversal, -1: Bearish reversal

    # --- Oversold conditions (Bullish Reversal) ---

    # RSI oversold and turning up
    data.loc[(data['RSI'] < 30) & (data['RSI'] > data['RSI'].shift(1)) &
             (data['RSI'].shift(1) <= data['RSI'].shift(2)), 'Reversal_Signal'] = 1

    # Stochastic oversold and crossing up
    data.loc[(data['Stoch_K'] < 20) & (data['Stoch_K'] > data['Stoch_D']) &
             (data['Stoch_K'].shift(1) <= data['Stoch_D'].shift(1)), 'Reversal_Signal'] = 1

    # Price below lower Bollinger Band and turning up
    data.loc[(data['Close'] < data['BB_Lower']) & (data['Close'] > data['Close'].shift(1)) &
             (data['BB_PCT'] < 0.05), 'Reversal_Signal'] = 1

    # MACD histogram turning up from below zero
    data.loc[(data['MACD_Hist'] < 0) & (data['MACD_Hist'] > data['MACD_Hist'].shift(1)) &
             (data['MACD_Hist'].shift(1) <= data['MACD_Hist'].shift(2)), 'Reversal_Signal'] = 1

    # Bullish divergence: Price making lower lows but RSI making higher lows
    data.loc[(data['Close'] < data['Close'].shift(1)) & (data['Close'].shift(1) < data['Close'].shift(2)) &
             (data['RSI'] > data['RSI'].shift(1)) & (
                         data['RSI'].shift(1) > data['RSI'].shift(2)), 'Reversal_Signal'] = 1

    # --- Overbought conditions (Bearish Reversal) ---

    # RSI overbought and turning down
    data.loc[(data['RSI'] > 70) & (data['RSI'] < data['RSI'].shift(1)) &
             (data['RSI'].shift(1) >= data['RSI'].shift(2)), 'Reversal_Signal'] = -1

    # Stochastic overbought and crossing down
    data.loc[(data['Stoch_K'] > 80) & (data['Stoch_K'] < data['Stoch_D']) &
             (data['Stoch_K'].shift(1) >= data['Stoch_D'].shift(1)), 'Reversal_Signal'] = -1

    # Price above upper Bollinger Band and turning down
    data.loc[(data['Close'] > data['BB_Upper']) & (data['Close'] < data['Close'].shift(1)) &
             (data['BB_PCT'] > 0.95), 'Reversal_Signal'] = -1

    # MACD histogram turning down from above zero
    data.loc[(data['MACD_Hist'] > 0) & (data['MACD_Hist'] < data['MACD_Hist'].shift(1)) &
             (data['MACD_Hist'].shift(1) >= data['MACD_Hist'].shift(2)), 'Reversal_Signal'] = -1

    # Bearish divergence: Price making higher highs but RSI making lower highs
    data.loc[(data['Close'] > data['Close'].shift(1)) & (data['Close'].shift(1) > data['Close'].shift(2)) &
             (data['RSI'] < data['RSI'].shift(1)) & (
                         data['RSI'].shift(1) < data['RSI'].shift(2)), 'Reversal_Signal'] = -1

    # --- Additional signal: Moving average crossovers ---

    # EMA20 crossing above EMA50 (medium-term bullish)
    data.loc[
        (data['EMA20'] > data['EMA50']) & (data['EMA20'].shift(1) <= data['EMA50'].shift(1)), 'Reversal_Signal'] = 1

    # EMA20 crossing below EMA50 (medium-term bearish)
    data.loc[
        (data['EMA20'] < data['EMA50']) & (data['EMA20'].shift(1) >= data['EMA50'].shift(1)), 'Reversal_Signal'] = -1

    # --- Volume confirmation ---

    # Strong volume on reversal day
    if 'Volume' in data.columns:
        # Volume much higher than average on bullish signal
        data.loc[(data['Reversal_Signal'] == 1) &
                 (data['Volume'] < data['Volume'].rolling(20).mean() * 1.5), 'Reversal_Signal'] = 0

        # Volume much higher than average on bearish signal
        data.loc[(data['Reversal_Signal'] == -1) &
                 (data['Volume'] < data['Volume'].rolling(20).mean() * 1.5), 'Reversal_Signal'] = 0

    # Count signals
    bullish_signals = (data['Reversal_Signal'] == 1).sum()
    bearish_signals = (data['Reversal_Signal'] == -1).sum()

    print(f"Identified {bullish_signals} bullish reversal signals and {bearish_signals} bearish reversal signals")

    return data


def identify_intraday_reversal_windows(data):
    """Identify the best time windows for intraday reversals."""
    print("Analyzing intraday reversal patterns...")

    # Convert to intraday if using minute data
    if isinstance(data.index, pd.DatetimeIndex) and data.index.time.max() != data.index.time.min():
        # Group by hour of day
        data['Hour'] = data.index.hour
        hourly_stats = data.groupby('Hour').agg({
            'Reversal_Signal': lambda x: (x != 0).mean(),
            'Daily_Return': ['mean', 'std']
        })

        print("Intraday reversal timing analysis:")
        print(hourly_stats)

        # Identify best windows
        best_times = hourly_stats.sort_values(('Reversal_Signal', '<lambda>'), ascending=False).head(3)
        print("\nBest time windows for reversals:")
        for hour in best_times.index:
            print(
                f"{hour}:00 - {hour + 1}:00: {best_times.loc[hour, ('Reversal_Signal', '<lambda>')] * 100:.1f}% reversal probability")

        return hourly_stats
    else:
        print("Intraday timing analysis requires minute data")
        return None


def create_and_test_strategy(data):
    """Create and backtest a trading strategy based on reversal signals."""
    print("Creating and testing reversal-based trading strategy...")

    # Make a copy to avoid modifying the original
    strategy_data = data.copy()

    # Create position column (1: Long, -1: Short, 0: Cash)
    strategy_data['Position'] = 0

    # Enter a long position on bullish reversal
    strategy_data.loc[strategy_data['Reversal_Signal'] == 1, 'Position'] = 1

    # Enter a short position on bearish reversal
    strategy_data.loc[strategy_data['Reversal_Signal'] == -1, 'Position'] = -1

    # Hold position for 5 days max (if no opposing signal appears)
    for i in range(len(strategy_data)):
        if i <= 5:
            continue

        # If we have a position but no new signal, maintain for up to 5 days
        if strategy_data['Position'].iloc[i] == 0 and strategy_data['Position'].iloc[i - 1] != 0:
            # Check if there was a counter signal in the last 5 days
            if strategy_data['Reversal_Signal'].iloc[i - 5:i].abs().sum() == 0:
                strategy_data.loc[strategy_data.index[i], 'Position'] = strategy_data['Position'].iloc[i - 1]

    # Strategy returns (shifted because we trade on the signal of the previous day)
    strategy_data['Strategy_Return'] = strategy_data['Position'].shift(1) * strategy_data['Daily_Return']

    # Cumulative returns
    strategy_data['Cumulative_Market_Return'] = (1 + strategy_data['Daily_Return']).cumprod()
    strategy_data['Cumulative_Strategy_Return'] = (1 + strategy_data['Strategy_Return']).cumprod()

    # Calculate performance metrics
    total_days = len(strategy_data)
    in_market_days = (strategy_data['Position'] != 0).sum()
    market_exposure = in_market_days / total_days

    # Calculate win rate
    winning_trades = (strategy_data['Strategy_Return'] > 0).sum()
    total_trades = (strategy_data['Strategy_Return'] != 0).sum()
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    # Calculate returns
    total_return = strategy_data['Cumulative_Strategy_Return'].iloc[-1] - 1
    market_return = strategy_data['Cumulative_Market_Return'].iloc[-1] - 1

    # Calculate drawdowns
    strategy_data['Strategy_Peak'] = strategy_data['Cumulative_Strategy_Return'].cummax()
    strategy_data['Strategy_Drawdown'] = (strategy_data['Cumulative_Strategy_Return'] - strategy_data[
        'Strategy_Peak']) / strategy_data['Strategy_Peak']
    max_drawdown = strategy_data['Strategy_Drawdown'].min()

    # Calculate risk-adjusted returns
    annualized_return = (1 + total_return) ** (252 / total_days) - 1
    annualized_volatility = strategy_data['Strategy_Return'].std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0

    # Print performance metrics
    print("\nPerformance Metrics:")
    print(f"Total Return: {total_return:.2%}")
    print(f"Market Return: {market_return:.2%}")
    print(f"Outperformance: {total_return - market_return:.2%}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Market Exposure: {market_exposure:.2%}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

    # Visualize strategy performance
    plt.figure(figsize=(12, 8))
    plt.plot(strategy_data.index, strategy_data['Cumulative_Market_Return'], label='Buy & Hold')
    plt.plot(strategy_data.index, strategy_data['Cumulative_Strategy_Return'], label='Reversal Strategy')

    # Mark reversal signals on the chart
    bullish_signals = strategy_data[strategy_data['Reversal_Signal'] == 1].index
    bearish_signals = strategy_data[strategy_data['Reversal_Signal'] == -1].index

    for signal_date in bullish_signals:
        plt.scatter(signal_date, strategy_data.loc[signal_date, 'Cumulative_Strategy_Return'],
                    color='green', marker='^', s=100)

    for signal_date in bearish_signals:
        plt.scatter(signal_date, strategy_data.loc[signal_date, 'Cumulative_Strategy_Return'],
                    color='red', marker='v', s=100)

    plt.title('SPY Reversal Strategy Performance')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)

    # Save performance chart
    performance_path = 'spy_reversal_strategy_performance.png'
    plt.savefig(performance_path)
    plt.close()
    print(f"Performance chart saved to {os.path.abspath(performance_path)}")

    # Create equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(strategy_data.index, strategy_data['Cumulative_Strategy_Return'])
    plt.title('Strategy Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.grid(True)

    # Save equity curve
    equity_path = 'spy_reversal_equity_curve.png'
    plt.savefig(equity_path)
    plt.close()
    print(f"Equity curve saved to {os.path.abspath(equity_path)}")

    # Create drawdown chart
    plt.figure(figsize=(12, 6))
    plt.plot(strategy_data.index, strategy_data['Strategy_Drawdown'])
    plt.title('Strategy Drawdown')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.grid(True)

    # Save drawdown chart
    drawdown_path = 'spy_reversal_drawdown.png'
    plt.savefig(drawdown_path)
    plt.close()
    print(f"Drawdown chart saved to {os.path.abspath(drawdown_path)}")

    # Analyze when reversals are most effective
    # Group by month
    strategy_data['Month'] = strategy_data.index.month
    monthly_performance = strategy_data.groupby('Month')['Strategy_Return'].mean()

    plt.figure(figsize=(12, 6))
    monthly_performance.plot(kind='bar')
    plt.title('Average Monthly Strategy Returns')
    plt.xlabel('Month')
    plt.ylabel('Average Return')
    plt.grid(True)

    # Save monthly performance chart
    monthly_path = 'spy_reversal_monthly_performance.png'
    plt.savefig(monthly_path)
    plt.close()
    print(f"Monthly performance chart saved to {os.path.abspath(monthly_path)}")

    # Analyze performance by VIX level (if available)
    if 'VIX' in strategy_data.columns:
        strategy_data['VIX_Quintile'] = pd.qcut(strategy_data['VIX'], 5, labels=False)
        vix_performance = strategy_data.groupby('VIX_Quintile')['Strategy_Return'].mean()

        plt.figure(figsize=(12, 6))
        vix_performance.plot(kind='bar')
        plt.title('Strategy Performance by VIX Level')
        plt.xlabel('VIX Quintile (0=Low, 4=High)')
        plt.ylabel('Average Return')
        plt.grid(True)

        # Save VIX performance chart
        vix_path = 'spy_reversal_vix_performance.png'
        plt.savefig(vix_path)
        plt.close()
        print(f"VIX performance chart saved to {os.path.abspath(vix_path)}")

    return strategy_data, {
        'total_return': total_return,
        'market_return': market_return,
        'win_rate': win_rate,
        'market_exposure': market_exposure,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'total_trades': total_trades,
        'bullish_signals': len(bullish_signals),
        'bearish_signals': len(bearish_signals)
    }


def create_report(strategy_data, metrics, hourly_stats=None):
    """Create a comprehensive report of the reversal strategy."""
    print("Generating strategy report...")

    report = f"""
    SPY Reversal Trading Strategy Report
    ===================================

    Strategy Overview:
    -----------------
    This strategy identifies potential reversal points in SPY based on technical indicators
    including RSI, Stochastic Oscillator, Bollinger Bands, and MACD. It looks for oversold
    and overbought conditions combined with trend exhaustion signals to time entries.

    The strategy employs bullish reversal signals for long positions and bearish reversal
    signals for short positions, with a maximum holding period of 5 days.

    Performance Summary:
    ------------------
    Period: {strategy_data.index[0].strftime('%Y-%m-%d')} to {strategy_data.index[-1].strftime('%Y-%m-%d')}

    Total Return: {metrics['total_return']:.2%}
    Market Return: {metrics['market_return']:.2%}
    Outperformance: {metrics['total_return'] - metrics['market_return']:.2%}

    Win Rate: {metrics['win_rate']:.2%}
    Total Trades: {metrics['total_trades']}
    Market Exposure: {metrics['market_exposure']:.2%}

    Max Drawdown: {metrics['max_drawdown']:.2%}
    Sharpe Ratio: {metrics['sharpe_ratio']:.2f}

    Signal Distribution:
    ------------------
    Bullish Signals: {metrics['bullish_signals']}
    Bearish Signals: {metrics['bearish_signals']}

    Implementation for Options Trading:
    ---------------------------------
    For a 0DTE SPY options strategy based on these reversal signals:

    1. Entry: 
       - On bullish reversal signal: Buy slightly OTM calls
       - On bearish reversal signal: Buy slightly OTM puts

    2. Exit:
       - Take profit at 50% gain
       - Stop loss at 50% loss
       - Time-based exit 30 minutes before market close

    3. Position Sizing:
       - 1-2% of portfolio per trade
       - Adjust based on current market volatility

    4. Signal Considerations:
       - Strong signals occur when multiple indicators align
       - Volume confirmation is essential for signal validity
       - Higher probability setups occur near key support/resistance levels
    """

    # Add intraday timing analysis if available
    if hourly_stats is not None:
        best_hours = hourly_stats.sort_values(('Reversal_Signal', '<lambda>'), ascending=False).head(3).index
        report += f"""
    Optimal Intraday Timing:
    ----------------------
    Based on historical analysis, reversals are most frequent during these hours:
    """
        for hour in best_hours:
            signal_prob = hourly_stats.loc[hour, ('Reversal_Signal', '<lambda>')] * 100
            avg_return = hourly_stats.loc[hour, ('Daily_Return', 'mean')] * 100
            report += f"    - {hour}:00 - {hour + 1}:00: {signal_prob:.1f}% reversal probability, {avg_return:.2f}% avg return\n"

    report += f"""
    Risk Management:
    --------------
    1. Consider market context (trending vs. range-bound)
    2. Be cautious of reversals against strong trends
    3. Favor trades with positive risk:reward ratios (>1:1)
    4. Pay attention to important economic releases
    5. Be aware of overall market sentiment and volatility

    Notes:
    -----
    1. Reversals tend to be more reliable when confirmed by multiple indicators
    2. False signals are common in strongly trending markets
    3. Best performance occurs at market extremes (very overbought/oversold)
    4. Strategy performance varies by market regime

    Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """

    # Save the report
    report_path = 'spy_reversal_strategy_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"Report saved to {os.path.abspath(report_path)}")
    return report


def optimize_reversal_parameters(data):
    """Optimize parameters for reversal detection."""
    print("Optimizing reversal detection parameters...")

    # Define parameter ranges
    rsi_thresholds = [(20, 80), (25, 75), (30, 70), (35, 65)]
    stoch_thresholds = [(10, 90), (15, 85), (20, 80), (25, 75)]
    bb_thresholds = [(0.01, 0.99), (0.05, 0.95), (0.1, 0.9)]

    # Results storage
    results = []

    for rsi_lower, rsi_upper in rsi_thresholds:
        for stoch_lower, stoch_upper in stoch_thresholds:
            for bb_lower, bb_upper in bb_thresholds:
                # Create a copy of the data
                test_data = data.copy()

                # Apply the parameters
                # Reset the signal column
                test_data['Reversal_Signal'] = 0

                # Bullish signals
                test_data.loc[(test_data['RSI'] < rsi_lower) & (
                            test_data['RSI'] > test_data['RSI'].shift(1)), 'Reversal_Signal'] = 1
                test_data.loc[(test_data['Stoch_K'] < stoch_lower) & (
                            test_data['Stoch_K'] > test_data['Stoch_D']), 'Reversal_Signal'] = 1
                test_data.loc[(test_data['BB_PCT'] < bb_lower) & (
                            test_data['Close'] > test_data['Close'].shift(1)), 'Reversal_Signal'] = 1

                # Bearish signals
                test_data.loc[(test_data['RSI'] > rsi_upper) & (
                            test_data['RSI'] < test_data['RSI'].shift(1)), 'Reversal_Signal'] = -1
                test_data.loc[(test_data['Stoch_K'] > stoch_upper) & (
                            test_data['Stoch_K'] < test_data['Stoch_D']), 'Reversal_Signal'] = -1
                test_data.loc[(test_data['BB_PCT'] > bb_upper) & (
                            test_data['Close'] < test_data['Close'].shift(1)), 'Reversal_Signal'] = -1

                # Strategy returns
                test_data['Position'] = test_data['Reversal_Signal']
                test_data['Strategy_Return'] = test_data['Position'].shift(1) * test_data['Daily_Return']

                # Calculate performance
                total_return = (1 + test_data['Strategy_Return'].dropna()).prod() - 1
                win_rate = (test_data['Strategy_Return'] > 0).sum() / (test_data['Strategy_Return'] != 0).sum() if (
                                                                                                                               test_data[
                                                                                                                                   'Strategy_Return'] != 0).sum() > 0 else 0

                # Store results
                results.append({
                    'rsi_lower': rsi_lower,
                    'rsi_upper': rsi_upper,
                    'stoch_lower': stoch_lower,
                    'stoch_upper': stoch_upper,
                    'bb_lower': bb_lower,
                    'bb_upper': bb_upper,
                    'total_return': total_return,
                    'win_rate': win_rate
                })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Sort by total return
    optimal_params = results_df.sort_values('total_return', ascending=False).iloc[0]

    print("\nOptimal Parameters:")
    print(f"RSI Thresholds: {optimal_params['rsi_lower']}/{optimal_params['rsi_upper']}")
    print(f"Stochastic Thresholds: {optimal_params['stoch_lower']}/{optimal_params['stoch_upper']}")
    print(f"Bollinger Band Percentage Thresholds: {optimal_params['bb_lower']}/{optimal_params['bb_upper']}")
    print(f"Expected Return: {optimal_params['total_return']:.2%}")
    print(f"Expected Win Rate: {optimal_params['win_rate']:.2%}")

    # Save optimal parameters
    params_path = 'optimal_reversal_parameters.csv'
    results_df.to_csv(params_path)
    print(f"Parameter optimization results saved to {os.path.abspath(params_path)}")

    return optimal_params


def apply_to_0dte_options(strategy_data, metrics):
    """Apply the reversal strategy to 0DTE options trading."""
    print("Simulating application to 0DTE options trading...")

    # Create options simulator
    options_data = strategy_data.copy()

    # Options parameters
    average_iv = 0.2  # Average implied volatility for SPY options
    dte = 1 / 252  # 0DTE (1 day in annual terms)
    premium_factor = 0.4  # Option premium as percentage of expected move
    slippage = 0.05  # Slippage as percentage of premium
    commission = 0.01  # Commission as percentage of premium

    # Simple function to estimate option price based on moneyness and implied volatility
    def estimate_option_price(spot, strike, iv, dte, is_call):
        # Very simplified option pricing based on expected move
        expected_move = spot * iv * np.sqrt(dte)
        intrinsic = max(0, (spot - strike) if is_call else (strike - spot))
        time_value = expected_move * premium_factor
        return intrinsic + time_value

    # Initialize options columns
    options_data['Option_Type'] = np.where(options_data['Reversal_Signal'] == 1, 'Call',
                                           np.where(options_data['Reversal_Signal'] == -1, 'Put', 'None'))
    options_data['Strike'] = np.where(options_data['Option_Type'] == 'Call',
                                      options_data['Close'] * 1.01,  # 1% OTM for calls
                                      np.where(options_data['Option_Type'] == 'Put',
                                               options_data['Close'] * 0.99, 0))  # 1% OTM for puts

    # Calculate option prices
    options_data['Entry_Premium'] = options_data.apply(
        lambda row: estimate_option_price(
            row['Close'], row['Strike'],
            average_iv * (1 + 0.1 * row['ATR'] / row['Close']),
            dte, row['Option_Type'] == 'Call') if row['Option_Type'] != 'None' else 0,
        axis=1
    )

    # Account for slippage and commission
    options_data['Entry_Cost'] = options_data['Entry_Premium'] * (1 + slippage) + options_data[
        'Entry_Premium'] * commission

    # Simulate exit prices based on next day's movement
    options_data['Exit_Spot'] = options_data['Close'].shift(-1)  # Next day's close

    options_data['Exit_Premium'] = options_data.apply(
        lambda row: estimate_option_price(
            row['Exit_Spot'], row['Strike'],
            average_iv * (1 + 0.05 * row['ATR'] / row['Close']),  # Slightly lower IV for exit
            max(0.1 / 252, dte - 1 / 252),  # Remaining time
            row['Option_Type'] == 'Call') if row['Option_Type'] != 'None' else 0,
        axis=1
    )

    # Account for slippage and commission on exit
    options_data['Exit_Proceeds'] = options_data['Exit_Premium'] * (1 - slippage) - options_data[
        'Exit_Premium'] * commission

    # Calculate option returns
    options_data['Option_Return'] = np.where(
        options_data['Entry_Cost'] > 0,
        (options_data['Exit_Proceeds'] - options_data['Entry_Cost']) / options_data['Entry_Cost'],
        0
    )

    # Create cumulative options equity curve
    options_data['Option_Return'].fillna(0, inplace=True)
    options_data['Cumulative_Option_Return'] = (1 + options_data['Option_Return']).cumprod()

    # Calculate options performance metrics
    options_trades = options_data[options_data['Option_Type'] != 'None']
    options_win_rate = (options_trades['Option_Return'] > 0).sum() / len(options_trades) if len(
        options_trades) > 0 else 0
    options_total_return = options_data['Cumulative_Option_Return'].iloc[-1] - 1 if len(options_data) > 0 else 0

    # Calculate option strategy drawdowns
    options_data['Option_Peak'] = options_data['Cumulative_Option_Return'].cummax()
    options_data['Option_Drawdown'] = (options_data['Cumulative_Option_Return'] - options_data['Option_Peak']) / \
                                      options_data['Option_Peak']
    options_max_drawdown = options_data['Option_Drawdown'].min()

    # Compare stock vs options performance
    print("\nOptions Strategy Performance:")
    print(f"Total Return: {options_total_return:.2%}")
    print(f"Win Rate: {options_win_rate:.2%}")
    print(f"Max Drawdown: {options_max_drawdown:.2%}")
    print(f"Average Option Return: {options_trades['Option_Return'].mean():.2%}")

    # Plot strategy comparison
    plt.figure(figsize=(12, 8))
    plt.plot(options_data.index, options_data['Cumulative_Market_Return'], label='Buy & Hold SPY')
    plt.plot(options_data.index, options_data['Cumulative_Strategy_Return'], label='Stock Reversal Strategy')
    plt.plot(options_data.index, options_data['Cumulative_Option_Return'], label='Options Reversal Strategy')
    plt.title('Strategy Comparison: Stock vs Options')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)

    # Save comparison chart
    options_path = 'spy_reversal_options_comparison.png'
    plt.savefig(options_path)
    plt.close()
    print(f"Options comparison chart saved to {os.path.abspath(options_path)}")

    # Analyze options return distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(options_trades['Option_Return'], bins=30, kde=True)
    plt.title('Options Returns Distribution')
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.axvline(0, color='red', linestyle='--')
    plt.grid(True)

    # Save options distribution chart
    dist_path = 'spy_reversal_options_distribution.png'
    plt.savefig(dist_path)
    plt.close()
    print(f"Options return distribution saved to {os.path.abspath(dist_path)}")

    return options_data, {
        'options_total_return': options_total_return,
        'options_win_rate': options_win_rate,
        'options_max_drawdown': options_max_drawdown,
        'average_option_return': options_trades['Option_Return'].mean(),
        'options_trades': len(options_trades),
        'stock_strategy_return': metrics['total_return']
    }