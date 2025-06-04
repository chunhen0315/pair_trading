# Pair Trading
Pair trading backtest &amp; forward test framework

This project involved 2 part: candle and pair_trading. 

1. candle:
This is code to extract crypto assets close price through python binance api, involed datetime handling, pagination (receive data in ascending order) and specify assests to be retrieved.

2. pair_trading:
In a pair trading strategy, we need to decide which pair to trade. We can perform cointegraton test and correlation test between assets to identify assets'relationship. For the selected pair, we can perform backtesting using backtest.py and optimized it by grid search for best parameters. Lastly, we should perform forward test to validate stability of strategy.


![corr_heatmap_log](https://github.com/user-attachments/assets/5815ec18-e094-4483-ab82-6ecdf4588bd5)
![coint_heatmap_log](https://github.com/user-attachments/assets/5e36d49d-7c16-4c15-9dbf-afedf849af34)

