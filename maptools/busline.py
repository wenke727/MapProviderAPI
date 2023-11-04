#%%
import shapely
import requests
import pandas as pd
import geopandas as gpd
from shapely import Point, LineString

from coordtransform import gcj_to_wgs
from utils.vis import plot_geodata
from cfg import KEY


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


def parse_line_and_stops(json_data, ll='wgs'):
    """
    Parses the JSON data from an AMap API response to extract bus line geometries
    and corresponding bus stops. The geometries and stops are converted into 
    GeoDataFrames which are suitable for spatial analysis.

    Parameters
    ----------
    json_data : dict
        The JSON data obtained from the AMap API response.
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

    """
    if ll not in ['wgs', 'gcj']:
        raise ValueError("The 'll' parameter must be either 'wgs' or 'gcj'.")

    try:
        bus_lines = json_data.get('buslines', [])
        lines = []
        stops_data = []

        for line in bus_lines:
            coords_str = line.get('polyline', '')
            del line['polyline']
            coords = [tuple(map(float, p.split(','))) for p in coords_str.split(';') if p]
            
            if ll == 'wgs':
                coords = [gcj_to_wgs(*coord) for coord in coords]

            line_name = line.get('name', '')
            lines.append({**line, 'geometry': LineString(coords)})

            stops = line.get('busstops', [])
            for stop in stops:
                stop_location_str = stop.get('location', '')
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

    except Exception as e:
        print(f"An error occurred: {e}")
        return gpd.GeoDataFrame(), gpd.GeoDataFrame()


if __name__ == "__main__":
    line_num = 11
    result = get_bus_line(f'地铁{line_num}号线', city='0755', key=KEY)
    geometry_df, stops_df = parse_line_and_stops(result)

    plot_geodata(geometry_df)
    geometry_df[['id', 'name', 'geometry']].to_file(f"../data/line_{line_num}.geojson", driver="GeoJSON")
    stops_df.to_file(f"../data/line_{line_num}_stops.geojson", driver="GeoJSON")

# %%
