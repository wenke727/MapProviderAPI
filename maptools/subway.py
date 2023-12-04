#%%
import os
import json
import time 
import random
import requests
import pandas as pd
import geopandas as gpd
from loguru import logger
from copy import deepcopy
from pathlib import Path
from utils.vis import plot_geodata
from utils.misc import read_json_file
from utils.serialization import save_checkpoint, load_checkpoint, to_geojson
from utils.coords_utils import str_to_point, xyxy_str_to_linestring, xsys_str_to_linestring
from cfg import KEY, MEMORY, LL_SYS


SPECIAL_CASE = {
  '大兴国际机场线': '北京大兴国际机场线',
  "S6号线": "南京S6号线",
  "S7号线": "南京S7号线",
  "江跳线": "轨道交通江跳线", # 重庆
  '郑许线': '郑州地铁17号线', # 郑州
  'S1线': "温州S1线",
  '保税区线': "地铁保税区线", #
  "迪士尼": "迪士尼线", # 香港
}

BAD_CASE ={
    '宁滁线': '滁州',
    "屯马线": "香港",
    "凤凰磁浮观光快线": "xiangxi", # 湘西
    "金义东线义东段": "jinhua", # 金华
    "金义东线金义段": "jinhua", # 金华
}

@MEMORY.cache
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

@MEMORY.cache
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

    logger.debug(f"{url}\n\tLines: {df.ln.values.tolist()}")
    
    return df

def get_all_subways_in_China(filename):
    if os.path.exists(filename):
        return load_checkpoint(filename)
    
    cities = get_subway_cities_list()

    res = {}
    for i, city in cities.iterrows():
        lines = get_city_subway_lines(city)
        res[city['spell']] = {"df": lines, 
                              "lines": lines['ln'].values.tolist(), 
                              **city.to_dict()}
        # time.sleep(random.uniform(2, 8))

    save_checkpoint(res, filename)

    return res

@MEMORY.cache()
def get_bus_line(keywords, city='0755', output='json', extensions='all', key=None):
    if key is None:
        raise ValueError("An AMap API key is required to use this function.")

    api_url = "https://restapi.amap.com/v3/bus/linename"
    params = {
        'key': key,
        'city': city,
        'keywords': keywords,
        'output': output,
        'extensions': extensions
    }
    _url = api_url + "?" + "&".join([f"{key}={val}" for key, val in params.items()])
    logger.debug(_url)
    
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()

        if output == 'json':
            data = response.json()
            if data['status'] != '1':
                raise Exception(f"API Error: {data.get('info', 'No info available')} (infocode: {data.get('infocode', 'No infocode')})")
            return data
        else:
            return response.text
    except requests.exceptions.HTTPError as http_err:
        raise SystemError(f"HTTP error occurred: {http_err}") 
    except Exception as err:
        raise SystemError(f"An error occurred: {err}")

def parse_line_and_stops(json_data, included_line_types: set=None, ll='wgs'):
    if ll not in ['wgs', 'gcj']:
        raise ValueError("The 'll' parameter must be either 'wgs' or 'gcj'.")

    bus_lines = json_data.get('buslines', [])
    lines_data = []
    stops_data = []

    for line in bus_lines:
        coords_str = deepcopy(line.get('polyline', ''))
        del line['polyline']

        line_name = line.get('name', '')
        if included_line_types is not None:
            if str(line.get('type')) not in included_line_types:
                continue
        
        lines_data.append({**line, 'geometry': xyxy_str_to_linestring(coords_str, ll)})

        stops = line.get('busstops', [])
        for stop in stops:
            stop_location_str = stop.get('location', '')
            stops_data.append({
                **stop,
                'line_name': line_name,
                'geometry': str_to_point(stop_location_str, ll)
            })

    # Convert to GeoDataFrame
    geometry_gdf = gpd.GeoDataFrame(lines_data, geometry='geometry', crs=4326) \
        if lines_data else gpd.GeoDataFrame()
    stops_gdf = gpd.GeoDataFrame(stops_data, geometry='geometry', crs=4326) \
        if stops_data else gpd.GeoDataFrame()

    return geometry_gdf, stops_gdf

def fetch_and_parse_subway_lines(subway_line_names, citycode, included_line_types, ll, key):
    lines = []
    stations = []
    error_lst = []
    
    for _name in subway_line_names:
        if _name in SPECIAL_CASE:
            _name = SPECIAL_CASE[_name]

        line = get_bus_line(_name, citycode, key=key)
        if line.get('count', 0) == 0:
            error_lst.append((citycode, _name))
            # FIXME 为啥没有更新
            logger.warning(f"Fetching {_name}, {citycode} failed!")
            continue
        
        gdf_geometry, gdf_stops = parse_line_and_stops(line, included_line_types, ll)
        lines.append(gdf_geometry)
        stations.append(gdf_stops)
        
        level = 'debug' if len(gdf_geometry) > 0 else 'error'
        getattr(logger, level)(f"{_name}: {len(gdf_geometry)} lines, {len(gdf_stops)} stations.")
        time.sleep(random.uniform(1, 5))

    # Combine all the GeoDataFrames and reset index
    gdf_lines = gpd.GeoDataFrame(pd.concat(lines, ignore_index=True))
    gdf_stations = gpd.GeoDataFrame(pd.concat(stations, ignore_index=True))
    
    # Remove duplicate stations by 'id'
    gdf_lines.drop_duplicates('id', inplace=True)
    gdf_stations.drop_duplicates('id', inplace=True)

    return gdf_lines, gdf_stations

def parse_amap_web_search_response(fn, ll='wgs'):
    # TODO align attribute
    data = read_json_file(fn)

    # lines
    lines = data.get('busMoreData', {}).get("busline_list", [])
    df_lines = pd.DataFrame(lines)
    geoms = df_lines.apply(lambda x: xsys_str_to_linestring(x['xs'], x['ys'], ll), axis=1)
    df_lines = gpd.GeoDataFrame(df_lines, geometry=geoms, crs=4326)

    # stations
    stops = df_lines['via_stops'].explode()
    df_stations = pd.json_normalize(stops)

    df_stations.loc[:, 'location'] = df_stations.apply(lambda x: f"{x['location.lng']:.6f},{x['location.lat']:.6f}",axis=1)
    df_stations.loc[:, 'order'] = stops.index
    geoms = df_stations.location.apply(lambda x: str_to_point(x, ll))
    df_stations = gpd.GeoDataFrame(
        df_stations,
        geometry=geoms
    )
       
    # post process for `lines`
    line_rename_dict = {
        'name': 'name',
        'code': 'citycode',
        'front_name': 'start_stop',
        'terminal_name': 'end_stop',
        'via_stops': 'busstops',
    }
    
    _df_lines = df_lines.copy()
    df_lines = _df_lines[line_rename_dict.keys()].rename(columns=line_rename_dict)
    df_lines.loc[:, 'loop'] = False
    df_lines.loc[df_lines.start_stop == df_lines.end_stop, 'loop'] = False
    df_lines.loc[:, 'status'] = 1 # FIXME 这里没有对应的字段
    df_lines.loc[:, 'geometry'] = _df_lines.geometry
    # df_lines.plot()

    # update `busstops`
    busstops = df_stations[['id', 'name', 'sequence', 'location']].fillna('').to_dict(orient='records')
    busstops_lst = [[busstops[0]]]
    prev_idx = df_stations.iloc[0].order

    for i, stop in zip(df_stations.iloc[1:].order, busstops[1:]):
        # print(prev_idx, i, stop)
        if i == prev_idx:
            lst = busstops_lst[-1]
            lst.append(stop)
        else:
            busstops_lst.append([stop])
            # logger.info(f"{i}, {len(busstops[-1])}, {stop}")
        prev_idx = i

    df_lines.loc[:, 'busstops'] = [str(i) for i in busstops_lst]
    # df_lines.loc[:, 'busstops'] = [json.dumps(i) for i in busstops_lst]

    return df_lines, df_stations


#%%
if __name__ == "__main__":
    """ China subways """
    ll = 'wgs'
    save_folder = Path("../data/subway/") / LL_SYS
    df_lines = get_all_subways_in_China("../data/subway/China_subways.pkl")
    
    for city, city_info in df_lines.items():
        if city != 'shanghai':
            continue
        
        lines = city_info['lines']
        logger.info(f"City: {city_info['cityname']} / {city_info['spell']} / {city_info['adcode']}\n\tlines: {lines}")
        
        df_lines, df_stations = fetch_and_parse_subway_lines(
            subway_line_names=lines, 
            citycode=city_info['adcode'], 
            included_line_types=set(['地铁', '磁悬浮列车']), 
            ll=ll,
            key=KEY
        )
        
        to_geojson(df_lines, save_folder / f"{city}_subway_lines_{LL_SYS}")
        to_geojson(df_stations, save_folder / f"{city}_subway_station_{LL_SYS}")
        

# %%
""" 补充特殊的线路 """
fn = "../data/subway/bad_cases/宁滁线.json"
df_lines, df_stations = parse_amap_web_search_response(fn, LL_SYS)
df_lines
# to_geojson(df_lines, 'test_lines')
# to_geojson(df_stations, 'test_stations')

# %%
