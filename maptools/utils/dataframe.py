import numpy as np
import pandas as pd

def query_dataframe(df:pd.DataFrame, attr:str, val:str=None, contains:str=None):
    if val is None and contains is None:
        return df
    if contains is None:
        return df.query(f"{attr} == @val ")
    if val is None:
        return df.query(f"{attr}.str.contains('{contains}')", engine='python')

    return df

def filter_dataframe_columns(df, cols):
    cols = [i for i in cols if i in list(df)]
    
    return df[cols]

def reduce_mem_usage(df, verbose=True):
    start_mem = df.memory_usage().sum() / 1024**2
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
                    
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

def split_dataframe(df, n):
    """
    将 DataFrame 平均切分成 n 个子 DataFrame。
    
    参数:
    - df: 要切分的原始 DataFrame。
    - n: 要切分成的子 DataFrame 的数量。
    
    返回:
    - 一个包含 n 个子 DataFrame 的列表。
    """
    # 计算每个子 DataFrame 的大概行数
    chunk_size = int(np.ceil(len(df) / n))
    
    # 切分 DataFrame
    return [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]


if __name__ == "__main":
    # 示例使用
    # 创建一个示例 DataFrame
    df_example = pd.DataFrame({
        'A': range(1, 11),
        'B': range(11, 21)
    })

    # 将 DataFrame 平均切分成 3 个子 DataFrame
    sub_dfs = split_dataframe(df_example, 3)

    # 显示切分后的子 DataFrame
    for i, sub_df in enumerate(sub_dfs):
        print(f"子 DataFrame {i + 1}:\n{sub_df}\n")
