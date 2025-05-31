import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Define time periods
train_start = '2021-01-01 00:00:00'
train_end = '2024-01-01 00:00:00'
test_start = '2024-01-01 01:00:00'
test_end = '2025-01-01 00:00:00'

# Load and preprocess data
prices = pd.read_csv('ASSETS_1h_binance_candle_2.csv')
prices['datetime'] = pd.to_datetime(prices['datetime'])
prices = prices.set_index('datetime')

# Check for invalid price data
if (prices <= 0).any().any():
    raise ValueError("Price data contains zero or negative values, which are invalid for log transformation.")
prices = np.log(prices)
prices = prices.dropna()  # Remove any NaN or inf values after log transformation

# Split into train and test sets
train_price = prices[train_start:train_end]
test_price = prices[test_start:test_end]

# Compute correlation matrix
assets = train_price.columns
correlation_matrix = pd.DataFrame(np.nan, index=assets, columns=assets)

for i, asset1 in tqdm(enumerate(assets), total=len(assets), desc="Computing correlation"):
    for j, asset2 in enumerate(assets):
        if i <= j:  # Upper triangle including diagonal to avoid redundancy
            try:
                corr = train_price[asset1].corr(train_price[asset2], method='pearson')
                correlation_matrix.loc[asset1, asset2] = corr
                if i != j:  # Fill symmetric part, exclude diagonal
                    correlation_matrix.loc[asset2, asset1] = corr
            except Exception as e:
                print(f"Error computing correlation for {asset1} and {asset2}: {e}")
                correlation_matrix.loc[asset1, asset2] = np.nan
                if i != j:
                    correlation_matrix.loc[asset2, asset1] = np.nan
        if i == j:  # Diagonal
            correlation_matrix.loc[asset1, asset2] = 1.0  # Correlation of asset with itself is 1

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, fmt=".4f", cmap="RdBu_r", center=0, cbar_kws={'label': 'Correlation Coefficient'})
plt.title("Correlation Matrix (Log Prices)")
plt.xlabel("Assets")
plt.ylabel("Assets")
plt.tight_layout()
plt.savefig('corr_heatmap_log.png')
plt.show()

print(correlation_matrix)

# Select highly correlated pairs
selected_pairs = []
for i, asset1 in enumerate(assets):
    for j, asset2 in enumerate(assets):
        if i < j:  # Avoid duplicates
            corr = correlation_matrix.loc[asset1, asset2]
            if not np.isnan(corr) and abs(corr) > 0.5:  # Threshold for strong correlation
                selected_pairs.append((asset1, asset2, corr))

print("\nSelected Correlated Pairs (|correlation| > 0.5):")
if selected_pairs:
    for asset1, asset2, corr in selected_pairs:
        print(f"Pair: {asset1} - {asset2}, correlation: {corr:.4f}")
else:
    print("No pairs found with |correlation| > 0.5.")
