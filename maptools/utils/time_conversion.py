import pandas as pd


def convert_timestamp_to_datetime64(series, timezone_offset=8, timestamp_unit='s', precision='s', drop_tz=True):
    """
    Converts a series of timestamps to datetime64 with the specified timezone and precision.

    Parameters:
    - series (pd.Series): Series containing timestamps.
    - timezone_offset (int): The timezone offset from UTC, e.g., 8 for UTC+8.
    - timestamp_unit (str): The unit of input timestamps ('s' for seconds, 'ms' for milliseconds).
    - precision (str): The precision of the output datetime64 ('s' for seconds, 'T' for minutes, 'H' for hours).
    - drop_tz (bool): If True, removes timezone information from the final datetime objects.

    Returns:
    - pd.Series: A Series containing datetime64 objects, adjusted to the specified timezone and precision.

    The function first converts timestamp units if necessary, then localizes the series to UTC, converts it to the specified timezone, and adjusts the precision. If 'drop_tz' is True, timezone information is removed from the final datetimes.
    """
    if timestamp_unit == 'ms':
        series = series / 1000
    dt_series = pd.to_datetime(series, unit='s')
    timezone_str = f"Etc/GMT{'+' if timezone_offset < 0 else '-'}{abs(timezone_offset)}"
    dt_series = dt_series.dt.tz_localize('UTC').dt.tz_convert(timezone_str)
    dt_series = dt_series.dt.floor(precision)
    if drop_tz:
        dt_series = dt_series.dt.tz_localize(None)
    
    return dt_series

def convert_str_to_datetime64(series, format, timezone_offset=8, precision='s', drop_tz=True):
    """
    Convert a series of datetime strings (already in a specific timezone) to datetime64.

    Parameters:
    - series (pd.Series): Series of datetime strings.
    - format (str): String format for output datetime objects. 
      Some common formats are:
      - '%Y%m%d%H%M%S%f' - input as '20231219181940879'
      - '%Y-%m-%d %H:%M:%S' - input as '2021-12-31 23:59:59'
      - '%Y-%m-%d' - input as '2021-12-31'
      - '%H:%M:%S' - input as '23:59:59'
    - timezone_offset (int): Timezone offset from UTC (e.g., 8 for UTC+8) of the input strings.
    - precision (str): Precision for the datetime ('s' for seconds, 'T' for minutes, 'H' for hours).
    - drop_tz (bool): If True, removes timezone information from the final datetime objects.
      
    Returns:
    - pd.Series: Series of datetime64 objects.
    """
    dt_series = pd.to_datetime(series, format=format)
    timezone_str = f"Etc/GMT{'+' if timezone_offset < 0 else '-'}{abs(timezone_offset)}"
    dt_series = dt_series.dt.tz_localize(timezone_str)
    dt_series = dt_series.dt.floor(precision)
    if drop_tz:
        dt_series = dt_series.dt.tz_localize(None)    
    
    return dt_series

def query_dataframe_by_time_range(df, column_name, start_time, end_time, additional_filter=None):
    """
    Query a DataFrame for a specific time range in a given datetime column, with an optional additional filter.

    Parameters:
    - df (pd.DataFrame): The DataFrame to query.
    - column_name (str): Name of the datetime column.
    - start_time (str): Start time of the range, in a format compatible with the datetime column.
    - end_time (str): End time of the range, in a format compatible with the datetime column.
    - additional_filter (str, optional): An additional SQL-like query string for further filtering.

    Returns:
    - pd.DataFrame: A DataFrame filtered by the specified time range and additional filter.
    """
    sql = f"('{start_time}' <= {column_name} <= '{end_time}')"
    if additional_filter:
        sql += f" and {additional_filter}"
    filtered_df = df.query(sql)
    
    return filtered_df


if __name__ == "__main__":
    timestamp_series = pd.Series([1609459200000, 1609545600000])
    print(convert_timestamp_to_datetime64(timestamp_series, timezone_offset=8, timestamp_unit='ms', precision='s'))


    str_series = pd.Series(['20231219181940879', '20231220081940879'])
    print(convert_str_to_datetime64(str_series, '%Y%m%d%H%M%S%f', timezone_offset=8, precision='s'))

    df = pd.DataFrame({
        'datetime': pd.to_datetime(['2021-01-01 12:00', '2021-06-01 12:00', '2021-12-31 12:00'])
    })
    print(query_dataframe_by_time_range(df, 'datetime', '2021-01-01', '2021-12-31'))
