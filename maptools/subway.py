#%%
import time 
import random
import requests
import pandas as pd
import geopandas as gpd
from loguru import logger
from copy import deepcopy
from shapely import Point, LineString

from utils.vis import plot_geodata
from coordtransform import gcj_to_wgs
from utils.coords_utils import check_ll, str_to_point, str
from utils.memory import MEMORY

from cfg import KEY



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

    return df

@MEMORY.cache()
def get_bus_line(keywords, city='0755', output='json', extensions='all', key=None):
    """
    Retrieve bus or subway line information from AMap API.

    Parameters
    ----------
    keywords : str
        The line name to search for. For example, '地铁1号线'.
    city : str, optional
        The city code or city name. Default is '0755' for Shenzhen.
    output : str, optional
        The format of the output, either 'json' or 'xml'. Default is 'json'.
    extensions : str, optional
        The level of detail for the response, either 'base' or 'all'. Default is 'all'.
    key : str
        Your AMap API Key. This must be provided for the function to work.

    Returns
    -------
    dict or str
        The response from the AMap API as a dictionary if output is 'json', 
        otherwise as a raw text string.

    Raises
    ------
    requests.exceptions.HTTPError
        If the request to the AMap API fails.

    Notes
    -----
    - The API Key must be provided by the user.
    - This function does not handle pagination. For keywords that may return multiple
      results, it will only retrieve the first page.

    References
    ----------
    - API usage: https://www.jianshu.com/p/ef0a31803bc9
    """
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
    
    
    # FIXME: https://www.cnblogs.com/giserliu/p/8251114.html
    # BaseUrl = "https://ditu.amap.com/service/poiInfo?query_type=TQUERY&pagesize=20&pagenum=1&qii=true&cluster_state=5&need_utd=true&utd_sceneid=1000&div=PC1000&addr_poi_merge=true&is_classify=true&"
    # params = {
    #     'keywords':'11路',
    #     'zoom': '11',
    #     'city':'610100',
    #     'geoobj':'107.623|33.696|109.817|34.745'
    # }

    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()  
        return response.json() if output == 'json' else response.text
    except requests.exceptions.HTTPError as http_err:
        raise SystemError(f"HTTP error occurred: {http_err}") 
    except Exception as err:
        raise SystemError(f"An error occurred: {err}")

def parse_line_and_stops(json_data, included_line_types: set=None, ll='wgs'):
    """
    Parses the JSON data from an AMap API response to extract bus line geometries
    and corresponding bus stops. The geometries and stops are converted into 
    GeoDataFrames which are suitable for spatial analysis. Optionally filters the 
    bus lines by type if a set of included line types is provided.

    Parameters
    ----------
    json_data : dict
        The JSON data obtained from the AMap API response.
    included_line_types : set, optional
        A set of bus line types to include in the parsing. If provided, only bus lines 
        whose 'type' matches one of the entries in this set will be included.
    ll : str, optional
        The coordinate system to use. Can be 'wgs' for WGS-84 or 'gcj' for GCJ-02. 
        If 'gcj', a coordinate transformation is applied to convert the coordinates 
        to WGS-84. Default is 'wgs'.

    Returns
    -------
    tuple of (geopandas.GeoDataFrame, geopandas.GeoDataFrame)
        Returns two GeoDataFrames: one for the geometries of the bus lines and 
        one for the stops. Each geometry is represented as a LineString, and each 
        stop is represented as a Point.

    Raises
    ------
    ValueError
        If the 'll' parameter is not one of the accepted values ('wgs', 'gcj').
    Exception
        If any error occurs during the parsing of the JSON data or during the 
        creation of the GeoDataFrames.

    Notes
    -----
    The function assumes that the 'polyline' field in the JSON contains the bus line 
    geometries as a semicolon-separated list of comma-separated longitude and latitude 
    pairs. The 'busstops' field is assumed to contain a list of stops, each with a 
    'location' field that provides the longitude and latitude as a comma-separated 
    string.
    
    The filtering of bus lines is based on the 'type' field present in each line's 
    information within the JSON data. If 'included_line_types' is not None, only lines 
    with a type present in this set are processed.
    """
    if ll not in ['wgs', 'gcj']:
        raise ValueError("The 'll' parameter must be either 'wgs' or 'gcj'.")

    bus_lines = json_data.get('buslines', [])
    lines = []
    stops_data = []

    for line in bus_lines:
        coords_str = deepcopy(line.get('polyline', ''))
        del line['polyline']

        # FIXME
        coords = [tuple(map(float, p.split(','))) for p in coords_str.split(';') if p]
        if ll == 'wgs':
            coords = [gcj_to_wgs(*coord) for coord in coords]

        line_name = line.get('name', '')
        if included_line_types is not None:
            if line.get('type') not in included_line_types:
                continue
        
        lines.append({**line, 'geometry': LineString(coords)})

        stops = line.get('busstops', [])
        for stop in stops:
            stop_location_str = stop.get('location', '')

            # FIXME
            stop_location = tuple(map(float, stop_location_str.split(',')))
            if ll == 'wgs':
                stop_location = gcj_to_wgs(*stop_location)
            
            stops_data.append({
                **stop,
                'line_name': line_name,
                'geometry': Point(stop_location)
            })

    # Convert to GeoDataFrame
    geometry_gdf = gpd.GeoDataFrame(lines, geometry='geometry', crs=4326)
    stops_gdf = gpd.GeoDataFrame(stops_data, geometry='geometry', crs=4326)

    return geometry_gdf, stops_gdf

def fetch_and_parse_subway_lines(subway_line_names, citycode, included_line_types, key):
    """
    For a given citycode, retrieves and parses the subway line geometries and stations,
    then compiles them into GeoDataFrames.

    Parameters
    ----------
    subway_line_names : list
        A list of subway line names for which to fetch and parse the data.
    citycode : str
        The code of the city for which to fetch the subway lines.
    included_line_types : set
        A set of line types to include in the parsing.
    key : str
        API key for the service from which the subway line data is being fetched.

    Returns
    -------
    tuple
        A tuple containing two GeoDataFrames: one for the subway lines and one for the stations.

    Notes
    -----
    The function calls 'get_bus_line' to fetch each line's data and 'parse_line_and_stops' to parse it.
    There is a delay between API calls to avoid rate-limiting issues.
    """
    lines = []
    stations = []
    
    for _name in subway_line_names:
        line = get_bus_line(_name, citycode, key=key)
        gdf_geometry, gdf_stops = parse_line_and_stops(line, included_line_types)
        lines.append(gdf_geometry)
        stations.append(gdf_stops)
        # Delay to avoid rate limiting, randomized to mimic non-automated access patterns
        time.sleep(random.uniform(2, 15))

    # Combine all the GeoDataFrames and reset index
    gdf_lines = gpd.GeoDataFrame(pd.concat(lines, ignore_index=True))
    gdf_stations = gpd.GeoDataFrame(pd.concat(stations, ignore_index=True))
    # Remove duplicate stations by 'id'
    gdf_stations.drop_duplicates('id', inplace=True)

    return gdf_lines, gdf_stations


if __name__ == "__main__":
    """ single lines """
    lines = ['1号线/罗宝线', '坪山云巴1号线', '2号线/8号线', '3号线/龙岗线', '4号线/龙华线', '5号线/环中线', 
             '6号线/光明线', '6号线支线', '7号线/西丽线', '9号线/梅林线', '10号线/坂田线', '11号线/机场线', 
             '12号线/南宝线', '14号线/东部快线', '16号线/龙坪线', '20号线']
    line = lines[15]
    result = get_bus_line(line, city='0755', key=KEY)
    geometry_df, stops_df = parse_line_and_stops(result)

    plot_geodata(geometry_df)
    # geometry_df[['id', 'name', 'geometry']].to_file(f"../data/line_{line_num}.geojson", driver="GeoJSON")
    # stops_df.to_file(f"../data/line_{line_num}_stops.geojson", driver="GeoJSON")

    """ whole city """
    df_subway_cities = get_subway_cities_list()
    df_shenzhen_subway = get_city_subway_lines(df_subway_cities.loc[38])
    logger.info(f"City: {df_subway_cities.cityname.values}")

    lines_list = df_shenzhen_subway.ln.values
    logger.info(f"Lines: {lines_list}")

    lines, stations = fetch_and_parse_subway_lines(
        subway_line_names=lines_list, 
        citycode='4403', 
        included_line_types=set(['地铁']), 
        key=KEY
    )

# %%
