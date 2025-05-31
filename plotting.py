import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the backtest and forward test results
train_signals = pd.read_csv('bt_ETHUSDT_SOLUSDT.csv', index_col='datetime', parse_dates=True)
test_signals = pd.read_csv('ft_ETHUSDT_SOLUSDT.csv', index_col='datetime', parse_dates=True)

# Create a figure and axis
plt.figure(figsize=(12, 6))

# Plot equity curves
plt.plot(train_signals.index, train_signals['cumu_returns'], label='Training Period (Best Params, 2021-2023)', color='blue')
plt.plot(test_signals.index, test_signals['cumu_returns'], label='Forward Test Period (2024)', color='orange')

# Customize the plot
plt.title('Equity Curve for ETHUSDT-SOLUSDT Pairs Trading Strategy (Best Parameters)', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Cumulative Returns', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the plot
plt.savefig('equity_curve_best_params.png')
plt.show()


plt.figure(figsize=(12, 6))

# Plot KDE for z-scores
plt.hist(train_signals['zscore'].dropna(), bins=50, alpha=0.5, label='Training Period (2021-2023)', color='blue', density=True)
plt.hist(test_signals['zscore'].dropna(), bins=50, alpha=0.5, label='Forward Test Period (2024)', color='orange', density=True)

# Customize the plot
plt.title('Histogram of Z-Scores for ETHUSDT-SOLUSDT Pairs Trading Strategy', fontsize=14)
plt.xlabel('Z-Score', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the plot
plt.savefig('zscore_histogram.png')
plt.show()