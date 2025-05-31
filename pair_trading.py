import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
from tqdm import tqdm
from colorama import Fore

# Main script
train_start = '2021-01-01 00:00:00'
train_end = '2024-01-01 00:00:00'
test_start = '2024-01-01 01:00:00'
test_end = '2025-01-01 00:00:00'

window = 70
thres1 = 5
thres2 = -5

# Load and preprocess data
prices = pd.read_csv('ASSETS_1h_binance_candle_2.csv')
prices['datetime'] = pd.to_datetime(prices['datetime'])
prices = prices.set_index('datetime')

if (prices <= 0).any().any():
    raise ValueError("Price data contains zero or negative values, which are invalid for log transformation.")
prices = np.log(prices)
prices = prices.dropna()  # Remove any NaN or inf values after log transformation

train_price = prices[train_start:train_end]
test_price = prices[test_start:test_end]

# Define pairs to backtest (replace with your chosen pairs)
selected_pairs = [('ETHUSDT', 'SOLUSDT')]  # Example pair, modify as needed

# Define the pairs trading strategy function
def run_pairs_trading_strategy(train_price, asset1, asset2, window, thres1, thres2, beta):
    signals = pd.DataFrame(train_price[[asset1, asset2]].copy())
    signals['asset1_chg'] = train_price[asset1].pct_change()  # Percentage change on log prices
    signals['asset2_chg'] = train_price[asset2].pct_change()
    signals['spread'] = train_price[asset1] /( beta * train_price[asset2])  # Spread on log prices
    signals['spread_mean'] = signals['spread'].rolling(window=window).mean()
    signals['spread_std'] = signals['spread'].rolling(window=window).std()
    signals['zscore'] = (signals['spread'] - signals['spread_mean']) / signals['spread_std']
    signals['position'] = 0  # 1: long asset1/short asset2, -1: short asset1/long asset2, 0: no position
    signals['position'] = np.where(signals['zscore'] > thres1, -1, np.where(signals['zscore'] < thres2, 1, 0))
    signals['position'] = signals['position'].replace(0, np.nan).ffill()
    signals['trade'] = abs(signals['position'].shift(1) - signals['position']) 
    signals['returns'] = signals['position'].shift(1) * (signals['asset1_chg'] - beta * signals['asset2_chg']) - signals['trade'] * 0.0006
    signals['cumu_returns'] = signals['returns'].cumsum()
    signals['dd'] = signals['cumu_returns'] - signals['cumu_returns'].cummax()

    sr = round(signals['returns'].mean() / signals['returns'].std() * np.sqrt(365 * 24),4) if signals['returns'].std() != 0 else np.nan
    mdd = round(signals['dd'].min(),4) if not signals['dd'].empty else np.nan
    ar = round(signals['returns'].mean() * (365 * 24),4) if not signals['returns'].empty else np.nan
    cr = round(abs(ar / mdd),4) if mdd != 0 else np.nan
    cumu_returns = round(signals['cumu_returns'].iloc[-1],4) if not signals['cumu_returns'].empty else np.nan
    trades = int(signals['trade'].sum())

    metrics = {
        'Sharpe Ratio': sr,
        'Cumu Returns': cumu_returns,
        'MDD': mdd,
        'AR': ar,
        'CR': cr,
        'TRADES': trades,
    }

    return signals, metrics


for asset1, asset2 in selected_pairs:
    print(Fore.YELLOW + f"\nRunning backtesting for {asset1} - {asset2}")
    
    model = OLS(train_price[asset1], sm.add_constant(train_price[asset2])).fit()
    beta = model.params[asset2]
    
    signals, metrics = run_pairs_trading_strategy(train_price, asset1, asset2, window, thres1, thres2, beta)
    
    print(Fore.WHITE + f"Backtest Metrics for {asset1} - {asset2}:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    signals.to_csv(f'bt_{asset1}_{asset2}.csv')

# Optimization part
window_range = range(10, 200, 5)  # Window sizes from 10 to 200
thres1_range = np.arange(0.1, 5.1, 0.1)  # Thresholds from 0.1 to 5.0

optimization_results = []

for asset1, asset2 in selected_pairs:
    print(Fore.YELLOW + f"\nOptimizing parameters for {asset1} - {asset2}")
    pair_results = []
    
    # Fit cointegration model on log prices
    model = OLS(train_price[asset1], sm.add_constant(train_price[asset2])).fit()
    beta = model.params[asset2]
    
    # Grid search
    for window in tqdm(window_range, desc=f"Optimizing {asset1}-{asset2}"):
        for thres1 in thres1_range:
            thres2 = -thres1
            try:
                signals, metrics = run_pairs_trading_strategy(train_price, asset1, asset2, window, thres1, thres2, beta)
                pair_results.append({
                    'window': window,
                    'thres1': thres1,
                    'thres2': thres2,
                    **metrics
                })
            except Exception as e:
                print(Fore.RED + f"Error for {asset1}-{asset2}, window={window}, thres1={thres1}: {e}")
                pair_results.append({
                    'window': window,
                    'thres1': thres1,
                    'thres2': thres2,
                    'Sharpe Ratio': np.nan,
                    'Cumu Returns': np.nan,
                    'MDD': np.nan,
                    'AR': np.nan,
                    'CR': np.nan
                })
    
    # Convert to DataFrame and store
    pair_results_df = pd.DataFrame(pair_results)
    optimization_results.append({
        'pair': f"{asset1}-{asset2}",
        'results': pair_results_df
    })
    
    # Find best parameters
    valid_results = pair_results_df[
        pair_results_df['Sharpe Ratio'].notna() & 
        (pair_results_df['Sharpe Ratio'] != np.inf) & 
        (pair_results_df['Sharpe Ratio'] != -np.inf)
    ]
    
    
    if not valid_results.empty:
        best_params = valid_results.loc[valid_results['Sharpe Ratio'].idxmax()]
        print(Fore.WHITE + f"[ Best Params Optimized For {asset1} - {asset2} ]:")
        for key, value in best_params.items():
            print(f"{key}: {value}")
    
#Forward Test
    thres1 = best_params['thres1']
    thres2 = best_params['thres2']
    window = int(best_params['window'])

    print(Fore.YELLOW + f"\nRunning forward test for {asset1} - {asset2}")
    model = OLS(test_price[asset1], sm.add_constant(test_price[asset2])).fit()
    beta = model.params[asset2]

    signals, metrics = run_pairs_trading_strategy(test_price, asset1, asset2, window, thres1, thres2, beta)

    print(Fore.WHITE + f"Forward Metrics for {asset1} - {asset2}:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    signals.to_csv(f'ft_{asset1}_{asset2}.csv')


