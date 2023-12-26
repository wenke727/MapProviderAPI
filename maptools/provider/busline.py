import requests
import geopandas as gpd
from loguru import logger
from copy import deepcopy

from geo.coords_utils import str_to_point, xyxy_str_to_linestring
from cfg import MEMORY

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
