import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
from tqdm import tqdm

train_start = '2021-01-01 00:00:00'
train_end = '2024-01-01 00:00:00'
test_start = '2024-01-01 01:00:00'
test_end = '2025-01-01 00:00:00'

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

assets = train_price.columns
pvalue_matrix = pd.DataFrame(np.nan, index=assets, columns=assets)

for i, asset1 in tqdm(enumerate(assets), total=len(assets), desc="Computing cointegration"):
    for j, asset2 in enumerate(assets):
        if i < j:  # Upper triangle to avoid redundancy
            try:
                score, pvalue, _ = coint(train_price[asset1], train_price[asset2])
                pvalue_matrix.loc[asset1, asset2] = pvalue
                pvalue_matrix.loc[asset2, asset1] = pvalue  # Symmetric matrix
            except Exception as e:
                print(f"Error computing cointegration for {asset1} and {asset2}: {e}")
                pvalue_matrix.loc[asset1, asset2] = np.nan
                pvalue_matrix.loc[asset2, asset1] = np.nan
        elif i == j:  # Diagonal
            pvalue_matrix.loc[asset1, asset2] = 0.0

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(pvalue_matrix, annot=True, fmt=".4f", cmap="YlGnBu", cbar_kws={'label': 'Cointegration P-value'})
plt.title("Cointegration P-value Matrix (Log Prices)")
plt.xlabel("Assets")
plt.ylabel("Assets")
plt.tight_layout()
plt.savefig('coint_heatmap_log.png')
plt.show()

print(pvalue_matrix)

# Select cointegrated pairs
selected_pairs = []
for i, asset1 in enumerate(assets):
    for j, asset2 in enumerate(assets):
        if i < j:  # Avoid duplicates
            pvalue = pvalue_matrix.loc[asset1, asset2]
            if not np.isnan(pvalue) and pvalue < 0.5:
                selected_pairs.append((asset1, asset2, pvalue))

print("\nSelected Cointegrated Pairs (p-value < 0.5):")
if selected_pairs:
    for asset1, asset2, pvalue in selected_pairs:
        print(f"Pair: {asset1} - {asset2}, p-value: {pvalue:.4f}")
else:
    print("No pairs found with good p-value.")