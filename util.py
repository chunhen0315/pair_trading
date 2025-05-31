import pandas as pd
from binance.client import Client
import time
from datetime import datetime
import pytz

def str_to_timestamp(str):
    dt = datetime.strptime(str, '%Y-%m-%d %H:%M:%S')
    dt = pytz.utc.localize(dt)
    return int(dt.timestamp() * 1000)

def timestamp_to_datetime(timestamp):
    return datetime.fromtimestamp(timestamp/1000, tz=pytz.utc).strftime('%Y-%m-%d %H:%M:%S')

def get_kline_time(client, symbol, interval, start_time, end_time, limit):
    all_kline = []
    current_time = start_time

    while current_time < end_time:
        klines = client.get_klines(
            symbol=symbol,
            interval=interval,
            startTime=current_time,
            endTime=end_time,
            limit=limit
        )
        if not klines:
            break
        all_kline.extend(klines)
        current_time = klines[-1][0] + 1
        time.sleep(0.1)

    return all_kline