#%%
import os
import re
import time 
import random
import pandas as pd
import geopandas as gpd
from loguru import logger
from pathlib import Path

from geo.visualize import plot_geodata
from utils.logger import make_logger
from utils.serialization import to_geojson
from provider.busline import get_bus_line, parse_line_and_stops
from provider.subwayAmap import get_national_subways
from provider.subwayAmap import parse_amap_web_search_response
from cfg import KEY, MEMORY, LL_SYS

logger = make_logger('../cache', 'subway', console=True)


SPECIAL_CASE = {
    ('1100', '大兴国际机场线'): '北京大兴国际机场线',
    # ('3201', "S6号线"): "南京S6号线",
    # ('3201', "S7号线"): "南京S7号线",
    ('5000', "江跳线"): "轨道交通江跳线", # 重庆
    ('4101','郑许线'): '郑州地铁17号线', # 郑州
    ('2102', '保税区线'): "地铁保税区线", # 大连
    # ('3303', 'S1线'): "温州S1线",
    # ('3310', 'S1线'): "台州S1线",
    ('8100', "迪士尼"): "迪士尼线", # 香港
}

BAD_CASE = {
    ('4301', '长株潭西环线'): '长沙',
    ('4303', '长株潭西环线'): '湘潭',
    ('3411', '宁滁线'): '滁州',
    ('8100', "屯马线"): "香港",
    ('4331', "凤凰磁浮观光快线"): "xiangxi", # 湘西
    ('3307', "金义东线义东段"): "jinhua", # 金华
    ('3307', "金义东线金义段"): "jinhua", # 金华
}

def line_name_match_pattern(s):
    # 正则表达式模式 "Sxx线"，其中xx为1到2位任意数字
    pattern = r'S\d{1,2}线'
    # 使用 re.match() 检查字符串是否从头开始匹配给定的模式
    if re.match(pattern, s):
        return True
    else:
        return False

def fetch_subway_data_by_city(city_info, folder, ll, key=KEY, included_line_types = set(['地铁', '磁悬浮列车'])):
    """
    Fetch subway line and station data for a specified city, filtering by included line types.

    This function retrieves subway line and station information for a given city, filtering the data based on predefined conditions and included line types (e.g., '地铁', '磁悬浮列车'). It processes the data to normalize line names, filter out unwanted cases based on certain criteria (BAD_CASE, SPECIAL_CASE), and then constructs GeoDataFrames for both subway lines and stations. Finally, it exports these GeoDataFrames to GeoJSON files for further use.

    Parameters:
    - cityname: The name of the city for which subway data is being fetched.
    - city_info: A dictionary containing detailed information about the city, including city name, spelling, administrative code, and lines.
    - folder: The base folder path where the output GeoJSON files will be saved.
    - ll: The coordinate system used for the GeoDataFrames.
    - key: The API key used for fetching data (default: KEY).
    - included_line_types: A set of strings representing the types of lines to include in the output (default: {'地铁', '磁悬浮列车'}).

    Returns:
    A tuple of GeoDataFrames: (gdf_lines, gdf_stations), representing subway lines and stations, respectively. If no subway lines are found, returns two empty GeoDataFrames.

    Notes:
    The function logs various messages, including warnings for skipped lines and errors for lines with no data. It also ensures uniqueness of stations and lines by removing duplicates.
    """
    logger.info(
        f"City: [{city_info['cityname']}, {city_info['spell']}, "
        f"{city_info['adcode']}], Lines: {city_info['lines']}"
    )

    lines = []
    stations = []
    for _name in city_info['lines']:
        # check the name of subway
        if (city_info['adcode'], _name) in BAD_CASE:
            logger.warning(f"Skip {city_info['adcode']} {_name} due to lack of data.")
            continue
        if (city_info['adcode'], _name) in SPECIAL_CASE:
            _name = SPECIAL_CASE[(city_info['adcode'], _name)]
        if line_name_match_pattern(_name):
            _name = city_info['cityname'].replace("市", '') + _name[:-1] + "号线"

        line = get_bus_line(_name, city_info['adcode'], key=key)
        gdf_geometry, gdf_stops = parse_line_and_stops(line, included_line_types, ll)
        if not gdf_geometry.empty: lines.append(gdf_geometry)
        if not gdf_stops.empty: stations.append(gdf_stops)
        
        getattr(logger, 'debug' if len(gdf_geometry) > 0 else 'error')(
            f"{city_info['cityname']}({city_info['adcode']}), {_name}")
        time.sleep(random.uniform(1, 3))

    if len(lines) == 0:
        logger.warning(f"Check city {city_info['adcode']}, for there is no subway lines.")
        return gpd.GeoDataFrame(), gpd.GeoDataFrame()
    
    # Combine all the GeoDataFrames and reset index
    gdf_lines = gpd.GeoDataFrame(pd.concat(lines, ignore_index=True))
    gdf_stations = gpd.GeoDataFrame(pd.concat(stations, ignore_index=True))
    
    # Remove duplicate stations by 'id'
    gdf_lines.drop_duplicates('id', inplace=True)
    gdf_stations.id = gdf_stations.id.astype(str)
    gdf_stations.drop_duplicates(['id', 'name', 'line_name'], inplace=True)
    
    to_geojson(gdf_lines, folder / LL_SYS / f"{city_info['cityname']}_subway_lines_{LL_SYS}")
    to_geojson(gdf_stations, folder / LL_SYS / f"{city_info['cityname']}_subway_station_{LL_SYS}")

    return gdf_lines, gdf_stations 

def pipeline_subway_data(folder="../data/subway/"):
    folder = Path(folder)
    df_lines_lst = get_national_subways(folder / "China_subways-240316.pkl")
    citycode_2_cityname = {val['adcode']: key for key, val in df_lines.items()}
    
    for cityname, city_info in df_lines_lst.items():
        # if cityname != 'wuxi':
            # continue
        fetch_subway_data_by_city(city_info, folder, ll=LL_SYS)
    
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
    

#%%
if __name__ == "__main__":
    # pipeline_subway_data(folder="../data/subway/", round=1)
    # pipeline_subway_data(folder="../data/subway/", round=2)
    pass

#%%
# TODO 检查更新的地铁信息
folder = "../data/subway/"
folder = Path(folder)
city_2_info = get_national_subways(folder / "China_subways-240316.pkl", refresh = False, auto_save = True)

# 输出统计指标
cities = [val['cityname'] for key, val in city_2_info.items()]
lines = pd.concat([val['df'] for key, val in city_2_info.items()])
stations = pd.json_normalize(lines.st.explode())


#%%
city_name = 'shenzhen'
# city_name = 'beijing'
city_name = 'wuxi'

city_2_info[city_name].keys()
df  = city_2_info[city_name]['df']

gdf_lines, gdf_stations = fetch_subway_data_by_city(city_2_info[city_name], folder, ll='gcj')
gdf_lines

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
