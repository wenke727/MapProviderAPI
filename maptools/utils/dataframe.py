import pandas as pd

def query_dataframe(df:pd.DataFrame, attr:str, val:str=None, contains:str=None):
    if val is None and contains is None:
        return df
    if contains is None:
        return df.query(f"{attr} == @val ")
    if val is None:
        return df.query(f"{attr}.str.contains('{contains}')", engine='python')

    return df

