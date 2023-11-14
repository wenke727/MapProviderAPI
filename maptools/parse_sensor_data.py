#%%
import os
import json
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from scipy.optimize import linear_sum_assignment


def extract_data_from_zip(zip_filename: str, keyword: str) -> pd.DataFrame:
    """
    Extracts data from a zip file based on a keyword in filenames and returns a DataFrame.
    
    Parameters:
    - zip_filename (str): The path to the zip file.
    - keyword (str): The keyword to search for in filenames.
    
    Returns:
    - pd.DataFrame: A DataFrame containing the extracted data.
    """
    all_records = []

    with zipfile.ZipFile(zip_filename, 'r') as z:
        filenames = [name for name in z.namelist() if keyword in name and name.endswith(".txt")]
        
        for filename in filenames:
            with z.open(filename) as f:
                for line in f:
                    line = line.decode('utf-8').strip()  
                    if line:  
                        all_records.append(json.loads(line))

    # Convert the JSON records to a pandas DataFrame
    return pd.DataFrame(all_records)

def convert_timestamp_to_datetime_custom_unit(df: pd.DataFrame, column_name: str, unit: str = 'ms') -> pd.DataFrame:
    """
    Convert a timestamp column in a DataFrame to a datetime column in UTC+8 timezone with a specified unit.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing the timestamp column.
    - column_name (str): The name of the timestamp column to convert.
    - unit (str): The unit of the timestamp ('s', 'ms', 'us', etc.). Default is 'ms'.
    
    Returns:
    - pd.DataFrame: The DataFrame with the timestamp column converted to datetime in UTC+8 timezone.
    """
    df[column_name] = pd.to_datetime(df[column_name], unit=unit)
    
    if df[column_name].dt.tz is not None:
        df[column_name] = df[column_name].dt.tz_localize(None)
    
    df[column_name] = df[column_name].dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
    return df

def extract_data_from_directory(directory_path: str, keyword: str, unit: str = 'ms') -> pd.DataFrame:
    """
    Extract data from all zip files in a given directory based on a keyword in filenames and returns a DataFrame.
    
    Parameters:
    - directory_path (str): The path to the directory containing zip files.
    - keyword (str): The keyword to search for in filenames.
    - unit (str): The unit of the timestamp ('s', 'ms', 'us', etc.). Default is 'ms'.
    
    Returns:
    - pd.DataFrame: A DataFrame containing the extracted data from all zip files.
    """
    all_dataframes = []
    
    for filename in os.listdir(directory_path):
        if filename.endswith(".zip"):
            df = extract_data_from_zip(os.path.join(directory_path, filename), keyword)
            df.loc[:, 'fn'] = filename
            
            # df = convert_timestamp_to_datetime_custom_unit(df, 'timestamp', unit)
            
            all_dataframes.append(df)
    
    return pd.concat(all_dataframes, ignore_index=True)

def explode_sat_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explode the satDataList column of the DataFrame and return a new DataFrame with satellite data.
    
    Parameters:
    - df (pd.DataFrame): The original DataFrame containing the satDataList column.
    
    Returns:
    - pd.DataFrame: A DataFrame with satellite data.
    """
    # Explode the satDataList column
    exploded_df = df[['satDataList']].explode('satDataList').reset_index()
    
    # Extract the satellite data into separate columns
    sat_df = pd.json_normalize(exploded_df['satDataList'])
    
    # Add the original index as a column to ensure it can be matched with the original df
    sat_df['rid'] = exploded_df['index']
    
    return sat_df

def optimal_bipartite_matching(gnss_df, light_df):
    gnss_times = gnss_df['timestamp'].values#.dt.tz_localize(None).values.astype(np.int64)
    light_times = light_df['timestamp'].values#.dt.tz_localize(None).values.astype(np.int64)
    
    time_diff_matrix = np.abs(np.subtract.outer(gnss_times, light_times))
    
    gnss_indices, light_indices = linear_sum_assignment(time_diff_matrix)
    
    valid_matches = np.where(time_diff_matrix[gnss_indices, light_indices] <= 60 * 1e9)[0]  # 1e9 converts seconds to nanoseconds
    
    return gnss_indices[valid_matches], light_indices[valid_matches]

folder = Path("../../../7_Dataset/Sat")
df = extract_data_from_directory(folder, "GNSS")
df_sat = explode_sat_data(df)
df_sat



# %%
""" Add light """

test_df_custom_unit = extract_data_from_zip(folder / "z_深圳_上梅林站_轨迹1测试A-C_20231101-161712.zip", "GNSS")
light_df = extract_data_from_zip(folder / "z_深圳_上梅林站_轨迹1测试A-C_20231101-161712.zip", "light")

gnss_indices, light_indices = optimal_bipartite_matching(test_df_custom_unit, light_df)

merged_optimal_df_v3 = test_df_custom_unit.copy()
merged_optimal_df_v3['nearest_light_data'] = np.nan
merged_optimal_df_v3.loc[gnss_indices, 'nearest_light_data'] = light_df.loc[light_indices, 'values'].values

merged_optimal_df_v3.head()




# %%

def parallel_aggregate(df_chunk):
    df_chunk = df_chunk.apply(np.array)
    _df = pd.json_normalize(df_chunk.apply(agg_funcs))
    _df.index = df_chunk.index

    return _df

def npartition(df, n):
    chunk_size = len(df) // n + (len(df) % n > 0)
    for i in range(0, len(df), chunk_size):
        yield df.iloc[i:i + chunk_size]


# Calculate aggregated statistics
def agg_funcs(x, funcs=['mean', 'std', '50%']):
    _dict = {
        'mean': np.mean(x),
        'std': np.std(x),
        'min': np.min(x),
        'max': np.max(x),
        '25%': np.quantile(x, 0.25),
        '50%': np.quantile(x, 0.5),
        '75%': np.quantile(x, 0.75)
    }
    
    return {i: _dict[i] for i in funcs }
    
    
def process_data(data, group_keys, attrs, bin_att=None, intvl=25, filter_sql=None, n_partitions=4,feat_lst=[], tag_2_feats={}, desc=None):
    """
    Process the dataframe by grouping, binning, and aggregating.

    Parameters:
    df (DataFrame): The input dataframe.
    group_keys (list): List of columns to group by.
    attrs (list): List of attributes to aggregate.
    bin_att (str, optional): Attribute for binning. Defaults to None.
    bucket (bool, optional): Flag to indicate if binning is needed. Defaults to True.
    intvl (int, optional): Interval for binning. Defaults to 25.
    filter_sql (str, optional): SQL condition for filtering data. Defaults to None.

    Returns:
    DataFrame: A dataframe with aggregated statistics.
    """

    if filter_sql:
        data = data.query(filter_sql)

    if bin_att and bin_att in data.columns:
        # if bin_att not in group_keys:
            # group_keys.append(bin_att)
        bin_col = bin_att[:5] + "Bin"
        data[bin_col] = (data[bin_att] // intvl).astype(np.int8)
        group_keys.append(bin_col)

    records = data.groupby(group_keys).agg({attr: list for attr in attrs})

    # Apply aggregation function
    df_chunks = list(npartition(records, n_partitions))
    
    lst = []
    for attr in attrs:
        _df_chunks = [df[attr] for df in df_chunks]
        pool = Pool(4)
        results = pool.map(parallel_aggregate, _df_chunks)
        _df = pd.concat(results)
        # records[attr] = records[attr].apply(np.array)
        # _df = pd.json_normalize(records[attr].apply(agg_funcs))
        _df.index = records.index
        _df.columns = [f'{attr}_{i}' for i in _df.columns]
        lst.append(_df)
    dfs = pd.concat(lst, axis=1)

    for _ in group_keys[1:]:
        dfs = dfs.unstack()

    # modify columns
    columns = []
    for values in dfs.columns:
        lst = []
        for val, prefix in zip(values, dfs.columns.names):
            if prefix is None:
                lst.append(val)
            else:
                lst.append(f"{prefix}_{val}")
        columns.append("-".join(lst[::-1]))
    dfs.columns = columns
    
    dfs.fillna(-1, inplace=True)
    feat_lst.append(dfs)
    if desc is None:
        desc = '-'.join(group_keys)
    tag_2_feats[desc] = dfs.columns.tolist()

    return dfs, records


dfs, df_processed = process_data(df_sat, group_keys=['rid'], 
                                 attrs=['satDb', 'elevation'], 
                                 bin_att='satDb', 
                                 intvl=10, 
                                 filter_sql='satDb > 0 or isUsed == True'
                                 )

df_processed
dfs

# %%
import numpy as np
import pandas as pd
from pathos.multiprocessing import ProcessingPool as Pool

records = df_processed.copy()
n_partitions = 4

df_chunks = list(npartition(records, n_partitions))

# 分割 DataFrame
attr = 'satDb'
df_chunks = [df[attr] for df in df_chunks]

pool = Pool(4)
results = pool.map(parallel_aggregate, df_chunks)
dfs = pd.concat(results)
dfs

# %%
