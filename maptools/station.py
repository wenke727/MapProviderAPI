#%%
import re
import time
import urllib
import random
import pandas as pd
from loguru import logger
from utils.memory import create_memory_cache
MEMORY = create_memory_cache(cachedir="/Users/wenke/Documents/Cache")


def filter_by_row_content(df:pd.DataFrame):
    cond = df.apply(lambda x: x.nunique() > 1, axis=1)

    return df[cond].reset_index(drop=True)

def filter_by_col_name(dfs:pd.DataFrame, match:str):
    idx = None
    for i, dfs in enumerate(dfs):
        for col in list(dfs):
            if match in str(col):
                idx = i
                return idx

    return None

@MEMORY.cache()
def extract_tables_from_url(url, match=None, sleep=False):
    try:
        if match is not None and not isinstance(match, (str, re.Pattern)):
            raise ValueError("The match parameter must be a string or a compiled regular expression.")
        logger.info(f"URL: {url}")
        dfs = pd.read_html(url)
        if sleep:
            time.sleep(random.uniform(2, 15))

        return dfs
    except ValueError as ve:
        print(f"No tables found matching the pattern: {match}")
        raise
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

def get_table(url, match, col): 
    tables = extract_tables_from_url(url, match)
    idx = filter_by_col_name(tables, col)
    if idx is None:
        print("No mathcing result.")
        return pd.DataFrame()

    df = filter_by_row_content(tables[idx])
    return df

def get_station_info(url, city):
    df_lines = get_table(url, "线路", "线路名称")

    # step 2:
    lst = []
    for name in df_lines['线路名称']:
        name = name.split("（")[0]
        url = f"https://baike.baidu.com/item/{urllib.parse.quote(name)}"
        try:
            df = get_table(url, None, '')
            df.loc[:, 'line'] = name
            logger.debug(f"{name}, {len(df)} stations, columns: {df.columns.tolist()}, URL: {url}")
            print(name, )
            lst.append(df)
        except:
            logger.error(f"{name}: {url}")
        # time.sleep(random.uniform(1, 9))

    df = pd.concat(lst)
    
    return df

def postprocess(lst, city):
    df = pd.concat([i.rename(columns=ARR_2_KEY) for i in lst])
    df['站台形式'] = df['站台形式'].fillna('')
    df['敷设方式'] = df['敷设方式'].fillna('')
    df.loc[:, 'label'] = df.apply(lambda x: str(x['站台形式']) + str(x['敷设方式']), axis=1)

    type_2_str = {
        0: ['地下'],
        1: ['地面', '地上'],
        2: ['高架'],
    }

    str_2_type = {}
    for id, values in type_2_str.items():
        for val in values:
            str_2_type[val] = id

    def get_lable(arr, str_2_type):
        for key in str_2_type:
            if key in str(arr):
                return str_2_type[key]
        
        return -1

    df.loc[:, 'type'] = df.label.apply(lambda x: get_lable(x, str_2_type))


    atts = ['name', 'line', 'label', 'type']
    df[atts].to_excel(f'./{city}_station_type.xlsx')

    return df


if __name__ == "__main__":
    #! 获取地铁站的类型
    # Step 1:
    # url = "https://baike.baidu.com/item/%E6%B7%B1%E5%9C%B3%E5%9C%B0%E9%93%81/1945642" # shenzhen
    # url = "https://baike.baidu.com/item/%E5%8C%97%E4%BA%AC%E5%9C%B0%E9%93%81/408485" # beijing
    # url = "https://baike.baidu.com/item/%E4%B8%8A%E6%B5%B7%E5%9C%B0%E9%93%81" # Shanghai

    # stations =  get_station_info(url, 'shanghai')
    # stations
    # url = "https://baike.baidu.com/item/%E4%B8%8A%E6%B5%B7%E5%9C%B0%E9%93%812%E5%8F%B7%E7%BA%BF#2_1" # 上海地铁 2号线
    url = "https://baike.baidu.com/item/%E4%B8%8A%E6%B5%B7%E5%9C%B0%E9%93%813%E5%8F%B7%E7%BA%BF/22947837" # 上海地铁 2号线
    url = "https://baike.baidu.com/item/%E5%B9%BF%E5%B7%9E%E5%9C%B0%E9%93%8114%E5%8F%B7%E7%BA%BF/5041514" # 广州地铁 14 号线
    tables = extract_tables_from_url(url, match=None)
    # pd.read_html(url)
    # tables.to_csv('')


# %%
df = tables[0].copy()
import numpy as np

RENAME_DICT = {
    'idx': ['编号', '序号'],
    'name': [ '车站', '车站站名', '车站名称', '站点名称', '站名'],
    'name_en': ['英文名称'],
    'district': [ '所在行政区', '所属区域', '所属行政区'],
    'location': ['车站位置', '地理位置'],
    "type": ['车站形式', '站台形式', '车站类型', '站台结构'],
    "exchange": ['换乘线路', '可换乘线路']
} 

ARR_2_KEY = {}
for key, values in RENAME_DICT.items():
    for val in values:
        ARR_2_KEY[val] = key


def is_digit_header(df):
    return np.array([str(i).isdigit() for i in df.columns]).all()

def get_station_label(df):
    label_cols = []
    for col in df.columns:
        ratio = df[col].str.contains('高架|地下|地面').sum() / df.shape[0]
        if ratio > 0.5:
            label_cols.append((ratio, col))

    if len(col) == 0:
        return None

    col = sorted(label_cols, reverse=True)[0][1]

    def _judge_label(arr):
        if '地下' in arr:
            return 0
        elif '地面' in arr:
            return 1
        elif '高架' in arr:
            return 2
        else:
            return -1

    return df[col].apply(_judge_label)


if is_digit_header(df):
    # FIXME
    df = pd.DataFrame(df.iloc[2:].values, columns=df.iloc[1].values)
    
if df.iloc[-1][0] == '注' or df.iloc[-1].nunique() == 1:
    df = df.iloc[:-1]

label = get_station_label(df) 
df.rename(columns=ARR_2_KEY, inplace=True)
df.loc[:, 'label'] = label
df

# %%


# %%

pd.read_html(url)[0]

# %%
import requests
from bs4 import BeautifulSoup

url = "https://baike.baidu.com/item/%E5%B9%BF%E5%B7%9E%E5%9C%B0%E9%93%8114%E5%8F%B7%E7%BA%BF/5041514" # 广州地铁 14 号线

response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
table = soup.find('table') # 或者使用更具体的选择器

for row in table.find_all('tr'):
    for cell in row.find_all('td'):
        print(cell.text)

# %%
