import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

from ..config.constant import MILLION, FPSCI_TIME_STR_FORMAT

def time_str_to_unix_microsec(time_string):
    """Convert single time string into UNIX microsecond"""
    ts = datetime.strptime(time_string, FPSCI_TIME_STR_FORMAT)
    ts = ts.replace(tzinfo=timezone.utc).timestamp()
    return int(ts * MILLION)

def time_list_to_unix_microsec(time_list):
    """Convert time string list into UNIX microsecond"""
    return list(map(time_str_to_unix_microsec, time_list))


def queryDB(db, command, args=(), convert_time=True):
    """SQL command query"""
    cur = db.cursor()
    q = cur.execute(command, args)
    cols = [column[0] for column in q.description]
    _d = pd.DataFrame.from_records(data=q.fetchall(), columns=cols)
    # Convert time to UNIX microsecond
    if convert_time:
        for col in cols:
            if 'time' in col:
                try:
                    _d[col] = pd.Series(time_list_to_unix_microsec(_d[col]))
                except: continue
    return _d

def filter_df_by_time_range(dataframe:pd.DataFrame, time_column, start_time, end_time):
    filtered_df = dataframe[(dataframe[time_column] >= start_time) & (dataframe[time_column] <= end_time)]
    return filtered_df


