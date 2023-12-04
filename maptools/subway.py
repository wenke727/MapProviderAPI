#%%
import os
import time 
import random
import requests
import pandas as pd
import geopandas as gpd
from loguru import logger
from copy import deepcopy
from pathlib import Path

from utils.vis import plot_geodata
from utils.coords_utils import str_to_point, str_to_linestring
from utils.serialization import save_checkpoint, load_checkpoint, to_geojson

from cfg import KEY

from utils.memory import create_memory_cache
MEMORY = create_memory_cache(cachedir="/Users/wenke/Documents/Cache")

SPECIAL_CASE = {
    '大兴国际机场线': ' 北京大兴国际机场线',
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
    lines = []
    stops_data = []

    for line in bus_lines:
        coords_str = deepcopy(line.get('polyline', ''))
        del line['polyline']

        line_name = line.get('name', '')
        if included_line_types is not None:
            if str(line.get('type')) not in included_line_types:
                continue
        
        lines.append({**line, 'geometry': str_to_linestring(coords_str, ll)})

        stops = line.get('busstops', [])
        for stop in stops:
            stop_location_str = stop.get('location', '')
            stops_data.append({
                **stop,
                'line_name': line_name,
                'geometry': str_to_point(stop_location_str, ll)
            })

    # Convert to GeoDataFrame
    geometry_gdf = gpd.GeoDataFrame(lines, geometry='geometry', crs=4326)
    stops_gdf = gpd.GeoDataFrame(stops_data, geometry='geometry', crs=4326)

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
            logger.warning(f"Fetching {_name}, {citycode} failed!")
            continue
        
        gdf_geometry, gdf_stops = parse_line_and_stops(line, included_line_types, ll)
        lines.append(gdf_geometry)
        stations.append(gdf_stops)
        
        level = 'debug' if len(gdf_geometry) > 0 else 'warning'
        getattr(logger, level)(f"{_name}: {len(gdf_geometry)} lines, {len(gdf_stops)} stations.")
        time.sleep(random.uniform(1, 5))

    # Combine all the GeoDataFrames and reset index
    gdf_lines = gpd.GeoDataFrame(pd.concat(lines, ignore_index=True))
    gdf_stations = gpd.GeoDataFrame(pd.concat(stations, ignore_index=True))
    
    # Remove duplicate stations by 'id'
    gdf_lines.drop_duplicates('id', inplace=True)
    gdf_stations.drop_duplicates('id', inplace=True)

    return gdf_lines, gdf_stations


#%%
if __name__ == "__main__":
    """ China subways """
    ll = 'wgs'
    save_folder = Path("../data/subway/")
    res = get_all_subways_in_China("../data/subway/China_subways.pkl")
    
    for city, city_info in res.items():
        if city != 'nanjing':
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
        
        to_geojson(df_lines, save_folder / f"{city}_subway_lines_{ll}")
        to_geojson(df_stations, save_folder / f"{city}_subway_station_{ll}")
        

# %%
