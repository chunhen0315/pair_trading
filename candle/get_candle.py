import pandas as pd
from binance.client import Client
import time
from datetime import datetime
import pytz
import util

api_key = "XDMzJMp2pSYue50iDgFvXMqxGhxzlWeFlraBNBi53lyw4WKDO3I8TNkA0H2Uw51R"
api_secret = "dlFdc3ZnSUBQOt6L1dWCBurhI5TOK88oT1JeUmeFW11jIolpifSoMC5Resm9NU8T"
client = Client(api_key, api_secret)

symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'TRXUSDT', 'DOGEUSDT']
interval = Client.KLINE_INTERVAL_1HOUR
limit = 1000
start_time_str = "2021-01-01 00:00:00"
end_time_str = "2025-01-01 00:00:00"

start_time = util.str_to_timestamp(start_time_str)
end_time = util.str_to_timestamp(end_time_str)

# Collect data for all symbols into a dictionary of DataFrames
dataframes = {}
for symbol in symbols:
    klines = util.get_kline_time(client, symbol, interval, start_time, end_time, limit)
    symbol_data = [{
        'datetime': util.timestamp_to_datetime(kline[0]),
        symbol: float(kline[4])  # Closing price
    } for kline in klines]
    dataframes[symbol] = pd.DataFrame(symbol_data)

# Merge DataFrames on 'datetime'
df = dataframes[symbols[0]]
for symbol in symbols[1:]:
    df = df.merge(dataframes[symbol], on='datetime', how='outer')

# Sort by datetime to ensure chronological order
df = df.sort_values('datetime').reset_index(drop=True)

print(df)

csv_filename = f'ASSETS_{interval}_binance_candle_2.csv'
df.to_csv(csv_filename, index=False)