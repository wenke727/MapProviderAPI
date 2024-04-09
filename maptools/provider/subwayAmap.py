import os
import time 
import random
import requests
import pandas as pd
import geopandas as gpd
from loguru import logger

from geo.coords_utils import str_to_point, xsys_str_to_linestring
from utils.misc import read_json_file
from utils.serialization import save_checkpoint, load_checkpoint


""" 获取有地铁的城市列别和线路列表 """
def fetch_subway_cities_dataframe(timestamp=None):
    """
    Fetch the list of cities with subway systems from AMAP (https://map.amap.com/subway/index.html)
    and parse the result into a DataFrame.

    Parameters:
    - timestamp (int, optional): A timestamp in milliseconds. If not provided, the current timestamp is used.

    Returns:
    - pandas.DataFrame: A DataFrame containing the city spellings, administrative codes, and city names.
    
    Raises:
    - Exception: If the request does not succeed or if the JSON response cannot be parsed.
    - ValueError: If the data from 'citylist' cannot be transformed into a DataFrame.
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

def fetch_subway_lines_as_dataframe(city_info, timestamp=None):
    """
    Retrieves a list of subway lines for a specified city from AMAP and converts it to a DataFrame.

    Parameters:
    - city_info (dict): A dictionary containing the 'city' (administrative code) and 'spell' (city name's spelling) for which to fetch the subway lines.
    - timestamp (int, optional): A timestamp in milliseconds. If not provided, the current timestamp is used.

    Returns:
    pandas.DataFrame: A DataFrame containing the details of the subway lines in the specified city.
    
    Raises:
    - AssertionError: If the necessary keys are not found in the input dictionary.
    - HTTPError: If the HTTP request to fetch the subway lines is unsuccessful.
    - JSONDecodeError: If the response body does not contain valid JSON.
    - ValueError: If the data cannot be transformed into a DataFrame.
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

def fetach_national_subway_data(filename, refresh=False, auto_save=False):
    """
    Compiles subway data from all cities into a structured format, with an option to save the data.

    This function retrieves the list of cities with subway systems and then fetches the subway line data for each city. The data is compiled into a dictionary, where each key is the city's spelling and the value is a dictionary containing both the DataFrame of subway lines and additional city details. If a file containing previously compiled data exists and the 'refresh' option is not set to True, the function will load and return this data instead of fetching new data.

    Parameters:
    - filename (str): The path of the file where the compiled data is stored or will be saved.
    - refresh (bool, optional): If True, the function will fetch new data even if a file with compiled data already exists. Defaults to False.
    - auto_save (bool, optional): If True, the newly compiled data will be automatically saved to the specified filename. Defaults to False.

    Returns:
    - dict: A dictionary with each city's spelling as keys and values containing a DataFrame of subway lines, list of line names, and city details.

    Raises:
    - Various exceptions may be raised depending on the success of data fetching, file operations, and the integrity of the data.

    Note:
    The function includes a delay between requests to avoid overwhelming the data source and to comply with usage policies.
    """
    if os.path.exists(filename) and not refresh:
        return load_checkpoint(filename)
    
    cities = fetch_subway_cities_dataframe()

    res = {}
    for _, city in cities.iterrows():
        lines = fetch_subway_lines_as_dataframe(city)
        res[city['spell']] = {"df": lines,  "lines": lines['ln'].values.tolist(), **city.to_dict()}
        time.sleep(random.uniform(.5, 6))

    if auto_save:
        save_checkpoint(res, filename)

    return res


def parse_amap_web_search_response(fn: str, ll: str = 'wgs') -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Parses the response of an Amap web search.

    Args:
        fn (str): Filename or file path to read JSON data.
        ll (str, optional): Coordinate system. Defaults to 'wgs'.

    Returns:
        tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]: Tuple containing GeoDataFrames for bus lines and stations.

    Refs:
        - https://www.amap.com/search?query=屯马线&city=810000&geoobj=114.06381%7C22.304433%7C114.158902%7C22.452082&zoom=12.79
    """
    # Assuming `read_json_file` is a custom function to read JSON data
    data = read_json_file(fn)

    # Extracting bus lines data
    lines = data.get('data', {}).get("busline_list", [])
    df_lines = pd.DataFrame(lines)

    # Converting bus lines data to GeoDataFrame
    geoms = df_lines.apply(lambda x: xsys_str_to_linestring(x['xs'], x['ys'], ll), axis=1)
    df_lines = gpd.GeoDataFrame(df_lines, geometry=geoms, crs=4326)
       
    line_rename_dict = {
        'name': 'name',
        'code': 'citycode',
        'front_name': 'start_stop',
        'terminal_name': 'end_stop',
        'stations': 'busstops',
    }

    # Renaming columns of bus lines DataFrame
    _df_lines = df_lines.copy()
    df_lines = _df_lines[line_rename_dict.keys()].rename(columns=line_rename_dict)
    df_lines.loc[:, 'loop'] = False
    df_lines.loc[df_lines.start_stop == df_lines.end_stop, 'loop'] = False
    df_lines.loc[:, 'status'] = 1  # FIXME: This line seems to be a placeholder for status value, needs to be revised
    df_lines = gpd.GeoDataFrame(df_lines, geometry=_df_lines.geometry)

    # Extracting bus stations data
    stops = df_lines['busstops'].explode()
    df_stations = pd.json_normalize(stops)

    # Processing bus stations data
    df_stations.loc[:, 'location'] = df_stations.xy_coords.apply(lambda x: x.replace(";", ','))
    df_stations.loc[:, 'order'] = stops.index
    df_stations.loc[:, 'id'] = df_stations.station_id
    df_stations.loc[:, 'sequence'] = range(len(df_stations))
    geoms = df_stations.location.apply(lambda x: str_to_point(x, ll))
    df_stations = gpd.GeoDataFrame(df_stations, geometry=geoms)

    # Reorganizing bus stations data by bus stops
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

    # Updating bus lines DataFrame with reorganized bus stops data
    df_lines.loc[:, 'busstops'] = [str(i) for i in busstops_lst]

    return df_lines, df_stations

