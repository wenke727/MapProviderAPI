import pytest
import pandas as pd
from loguru import logger
from maptools.utils.time_conversion import convert_timestamp_to_datetime64, convert_str_to_datetime64, query_dataframe_by_time_range


def test_convert_timestamp_to_datetime64():
    # 测试时间戳转换函数
    timestamp_series = pd.Series([1609459200000, 1609545600000])
    result = convert_timestamp_to_datetime64(timestamp_series, timezone_offset=8, timestamp_unit='ms', precision='s', drop_tz=False)
    logger.debug(f"\n{result}")
    
    # 时区的信息会过滤掉
    expected_dates = ['2021-01-01 08:00:00', '2021-01-02 08:00:00']
    assert all(date in expected_dates for date in result.dt.strftime('%Y-%m-%d %H:%M:%S')), "Timestamp conversion failed"

def test_convert_str_to_datetime64():
    # 测试字符串转换函数
    str_series = pd.Series(['20231219181940879', '20231220081940879'])
    result = convert_str_to_datetime64(str_series, '%Y%m%d%H%M%S%f', timezone_offset=8, precision='s', drop_tz=False)
    assert (result.dt.tz == pd.Timestamp('2023-12-19', tz='Etc/GMT-8').tz), "Timezone conversion failed"
    
    assert all(result.apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S')).isin(['2023-12-19 18:19:40', '2023-12-20 08:19:40'])), "String conversion failed"

def test_query_dataframe_by_time_range():
    # 测试时间区间查询函数
    df = pd.DataFrame({
        'datetime': pd.to_datetime(['2021-01-01 12:00', '2021-06-01 12:00', '2021-12-31 12:00'])
    })
    result = query_dataframe_by_time_range(df, 'datetime', '2021-01-01', '2021-12-31 23:59')
    assert len(result) == 3, "Query time range failed"
