#%%
import os
import time 
import random
import requests
import pandas as pd
import geopandas as gpd
from loguru import logger
from pathlib import Path

from geo.visualize import plot_geodata
from utils.misc import read_json_file
from utils.logger import make_logger
from utils.serialization import save_checkpoint, load_checkpoint, to_geojson
from geo.coords_utils import str_to_point, xsys_str_to_linestring
from provider.busline import get_bus_line, parse_line_and_stops
from cfg import KEY, MEMORY, LL_SYS

logger = make_logger('../cache', 'subway', console=True)


SPECIAL_CASE = {
    ('1100', '大兴国际机场线'): '北京大兴国际机场线',
    ('3201', "S6号线"): "南京S6号线",
    ('3201', "S7号线"): "南京S7号线",
    ('5000', "江跳线"): "轨道交通江跳线", # 重庆
    ('4101','郑许线'): '郑州地铁17号线', # 郑州
    ('2102', '保税区线'): "地铁保税区线", # 大连
    ('3303', 'S1线'): "温州S1线",
    ('3310', 'S1线'): "台州S1线",
    ('8100', "迪士尼"): "迪士尼线", # 香港
}

BAD_CASE ={
    ('4301', '长株潭西环线'): '长沙',
    ('4303', '长株潭西环线'): '湘潭',
    ('3411', '宁滁线'): '滁州',
    ('8100', "屯马线"): "香港",
    ('4331', "凤凰磁浮观光快线"): "xiangxi", # 湘西
    ('3307', "金义东线义东段"): "jinhua", # 金华
    ('3307', "金义东线金义段"): "jinhua", # 金华
}

def get_subway_cities_list(timestamp=None):
    """
    Fetch the list of cities with subway systems from AMAP (https://map.amap.com/subway/index.html)
    and parse the result into a DataFrame.

    Parameters:
    - timestamp (int, optional): A timestamp in milliseconds. If not provided, the current timestamp is used.

    Returns:
    - pandas.DataFrame: A DataFrame containing the city spellings, administrative codes, and city names.
    
    Raises:
    Exception: If the request does not succeed or if the JSON response cannot be parsed.
    ValueError: If the data from 'citylist' cannot be transformed into a DataFrame.
    """
    if timestamp is None:
        timestamp = int(time.time() * 1000)
    
    url = f"https://map.amap.com/service/subway?_{timestamp}&srhdata=citylist.json"
    logger.info(url)
    headers = {
        'Cookie': 'connect.sess=s%3Aj%3A%7B%7D.DffclZ%2FN%2BAiqU5kXMjqg3VQHapScLmBFjbTUDpqgPVQ'
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data['citylist'])

    except requests.RequestException as e:
        raise Exception(f"Request failed: {e}")
    except ValueError as e:
        raise ValueError(f"Error parsing JSON response: {e}")
    except KeyError as e:
        raise KeyError(f"The key 'citylist' was not found in the JSON response: {e}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {e}")

    return df

def get_city_subway_lines(city_info, timestamp=None):
    """
    Retrieves a list of subway lines for a specified city from AMAP and converts it to a DataFrame.

    Parameters:
    - city_info (dict): A dictionary containing the 'city' (administrative code) and 'spell' (city name's spelling) for which to fetch the subway lines.
    - timestamp (int, optional): A timestamp in milliseconds. If not provided, the current timestamp is used.

    Returns:
    pandas.DataFrame: A DataFrame containing the details of the subway lines in the specified city.
    
    Raises:
    AssertionError: If the necessary keys are not found in the input dictionary.
    HTTPError: If the HTTP request to fetch the subway lines is unsuccessful.
    JSONDecodeError: If the response body does not contain valid JSON.
    ValueError: If the data cannot be transformed into a DataFrame.
    """
    adcode = city_info.get('adcode', None)
    spell = city_info.get('spell', None)
    assert adcode is not None and spell is not None, "The input dictionary must contain 'city' and 'spell' keys."
    
    if timestamp is None:
        timestamp = int(time.time() * 1000)
    
    url = f"https://map.amap.com/service/subway?_{timestamp}&srhdata={adcode}_drw_{spell}.json"
    headers = {
        'Cookie': 'connect.sess=s%3Aj%3A%7B%7D.DffclZ%2FN%2BAiqU5kXMjqg3VQHapScLmBFjbTUDpqgPVQ'
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.HTTPError as e:
        raise requests.HTTPError(f"HTTP request failed with status code {response.status_code}: {e}")

    try:
        data = response.json()
    except ValueError as e:
        raise ValueError(f"Error parsing JSON response: {e}")

    try:
        df = pd.DataFrame(data['l'])  # Assuming 'l' key contains the line data
    except (KeyError, ValueError) as e:
        raise e

    logger.debug(f"City: {city_info.get('cityname', None)}, Lines: {df.ln.values.tolist()}, url: {url}")
    
    return df

def get_national_subways(filename, refresh=False, auto_save=False):
    if os.path.exists(filename) and not refresh:
        return load_checkpoint(filename)
    
    # timestamp = int(time.time() * 1000)
    cities = get_subway_cities_list()

    res = {}
    for i, city in cities.iterrows():
        lines = get_city_subway_lines(city)
        res[city['spell']] = {
            "df": lines, 
            "lines": lines['ln'].values.tolist(), 
            **city.to_dict()
        }
        time.sleep(random.uniform(.5, 6))

    if auto_save:
        save_checkpoint(res, filename)

    return res

def fetch_and_parse_subway_lines(subway_line_names, cityname, citycode, included_line_types, ll, key):
    lines = []
    stations = []
    error_lst = []
    
    for _name in subway_line_names:
        if (citycode, _name) in SPECIAL_CASE:
            _name = SPECIAL_CASE[(citycode, _name)]
        if (citycode, _name) in BAD_CASE:
            logger.warning(f"Skip {citycode} {_name} due to lack of data.")
            continue

        line = get_bus_line(_name, citycode, key=key)
        gdf_geometry, gdf_stops = parse_line_and_stops(line, included_line_types, ll)
        if not gdf_geometry.empty: 
            lines.append(gdf_geometry)
        if not gdf_stops.empty: 
            stations.append(gdf_stops)
        
        level = 'debug' if len(gdf_geometry) > 0 else 'error'
        getattr(logger, level)(f"{cityname}({citycode}), {_name}")
        time.sleep(random.uniform(1, 3))

    if len(lines) == 0:
        logger.warning(f"Check city {citycode}, for there is no subway lines.")
        return pd.DataFrame(), pd.DataFrame()
    
    # Combine all the GeoDataFrames and reset index
    gdf_lines = gpd.GeoDataFrame(pd.concat(lines, ignore_index=True))
    gdf_stations = gpd.GeoDataFrame(pd.concat(stations, ignore_index=True))
    
    # Remove duplicate stations by 'id'
    gdf_lines.drop_duplicates('id', inplace=True)
    gdf_stations.id = gdf_stations.id.astype(str)
    gdf_stations.drop_duplicates(['id', 'name', 'line_name'], inplace=True)

    return gdf_lines, gdf_stations

def parse_amap_web_search_response(fn, ll='wgs'):
    """_summary_

    Args:
        fn (function): _description_
        ll (str, optional): _description_. Defaults to 'wgs'.

    Returns:
        _type_: _description_
    Refs:
        - https://www.amap.com/search?query=屯马线&city=810000&geoobj=114.06381%7C22.304433%7C114.158902%7C22.452082&zoom=12.79
    """
    data = read_json_file(fn)

    # lines
    lines = data.get('data', {}).get("busline_list", [])
    df_lines = pd.DataFrame(lines)
    geoms = df_lines.apply(lambda x: xsys_str_to_linestring(x['xs'], x['ys'], ll), axis=1)
    df_lines = gpd.GeoDataFrame(df_lines, geometry=geoms, crs=4326)
       
    line_rename_dict = {
        'name': 'name',
        'code': 'citycode',
        'front_name': 'start_stop',
        'terminal_name': 'end_stop',
        'stations': 'busstops',
    }
    
    _df_lines = df_lines.copy()
    df_lines = _df_lines[line_rename_dict.keys()].rename(columns=line_rename_dict)
    df_lines.loc[:, 'loop'] = False
    df_lines.loc[df_lines.start_stop == df_lines.end_stop, 'loop'] = False
    df_lines.loc[:, 'status'] = 1 # FIXME 这里没有对应的字段
    df_lines = gpd.GeoDataFrame(df_lines, geometry=_df_lines.geometry)

    # stations
    stops = df_lines['busstops'].explode()
    df_stations = pd.json_normalize(stops)

    df_stations.loc[:, 'location'] = df_stations.xy_coords.apply(lambda x: x.replace(";", ','))
    df_stations.loc[:, 'order'] = stops.index
    df_stations.loc[:, 'id'] = df_stations.station_id
    df_stations.loc[:, 'sequence'] = range(len(df_stations))
    geoms = df_stations.location.apply(lambda x: str_to_point(x, ll))
    df_stations = gpd.GeoDataFrame(df_stations, geometry=geoms)
    
    busstops = df_stations[['id', 'name', 'location']].fillna('').to_dict(orient='records')
    busstops_lst = [[busstops[0]]]
    prev_idx = df_stations.iloc[0].order

    for i, stop in zip(df_stations.iloc[1:].order, busstops[1:]):
        if i == prev_idx:
            lst = busstops_lst[-1]
            lst.append(stop)
        else:
            busstops_lst.append([stop])
        prev_idx = i

    df_lines.loc[:, 'busstops'] = [str(i) for i in busstops_lst]
    # df_lines.loc[:, 'busstops'] = [json.dumps(i) for i in busstops_lst]

    return df_lines, df_stations

def pipeline_subway_data(folder="../data/subway/", round=1):
    folder = Path(folder)
    df_lines = get_national_subways(folder / "China_subways.pkl")
    citycode_2_cityname = {val['adcode']: key for key, val in df_lines.items()}
    
    COUNT = 0
    if round == 1:
        for cityname, city_info in df_lines.items():
            lines = city_info['lines']
            logger.info(f"City: [{city_info['cityname']}, {city_info['spell']}, "
                        f"{city_info['adcode']}], Lines: {lines}")
            
            df_lines, df_stations = fetch_and_parse_subway_lines(
                subway_line_names=lines, 
                cityname=city_info['cityname'],
                citycode=city_info['adcode'], 
                included_line_types=set(['地铁', '磁悬浮列车']), 
                ll=LL_SYS,
                key=KEY
            )
            
            if not df_lines.empty:
                to_geojson(df_lines, folder / LL_SYS / f"{cityname}_subway_lines_{LL_SYS}")
            if not df_stations.empty:
                to_geojson(df_stations, folder / LL_SYS / f"{cityname}_subway_station_{LL_SYS}")
        return True
    elif round == 2:
        for dirpath, dirnames, filenames in os.walk(folder / 'bad_cases'):
            for file_name in filenames:
                if 'json' not in file_name:
                    continue
                
                citycode, line_name = file_name.replace('.json', '').split('_')
                cityname = citycode_2_cityname[citycode]
                json_fn = os.path.join(dirpath, file_name)
                df_lines, df_stations =  parse_amap_web_search_response(json_fn)
                
                line_fn = folder / LL_SYS / f"{cityname}_subway_lines_{LL_SYS}.geojson"
                if not os.path.exists(line_fn):
                    to_geojson(df_lines, line_fn)
                    continue
                
                ori_gdf = gpd.read_file(line_fn)
                gdf = gpd.GeoDataFrame(pd.concat([ori_gdf, df_lines]), crs=4326)
                to_geojson(gdf, line_fn)
        
        return True
    
    return False


#%%
if __name__ == "__main__":
    pipeline_subway_data(folder="../data/subway/", round=1)
    pipeline_subway_data(folder="../data/subway/", round=2)

#%%
folder = "../data/subway/"
folder = Path(folder)
df_lines = get_national_subways(folder / "China_subways-240116.pkl", refresh=False, auto_save=True)

# 输出统计指标
cities = [val['cityname'] for key, val in df_lines.items()]
lines = pd.concat([val['df'] for key, val in df_lines.items()])
stations = pd.json_normalize(lines.st.explode())


#%%
city = 'shenzhen'
# city = 'beijing'

df_lines[city].keys()
df  = df_lines[city]['df']
df.head(1)

# %%
line_rename_dict = {
   'ln': "line_name",
   'kn': 'full_name',
   'ls': 'line_id', # key
   'su': 'status',
   'cl': 'color',
   'li': 'line_id_lst',
   'st': 'stations',
   'x': 'x'
}

station_rename_dict = {
 'n': 'name',
 'sid': 'nid',
 'poiid': 'bvid',
} # bvid 用于检测是否存在

df = df[line_rename_dict.keys()].rename(columns=line_rename_dict)
df

# %%


# TODO 转换成`点`和`边`的关系
stops = df['stations'].explode()
df_stations = pd.json_normalize(stops)

df_stations = df_stations[station_rename_dict.keys()].rename(columns=station_rename_dict)
df_stations.loc[:, 'line_id'] = stops.index
df_stations

# %%
