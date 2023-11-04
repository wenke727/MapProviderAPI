import re
import shapely
import requests
import pandas as pd
import geopandas as gpd
from loguru import logger
import matplotlib.pyplot as plt
from utils.vis import plot_geodata
from coordtransform import gcj_to_wgs

from cfg import KEY


def convert_to_geom(df: pd.DataFrame, ll_sys: str = 'wgs') -> gpd.GeoDataFrame:
    """
    Convert DataFrame locations to geometries based on the coordinate system.
    """
    if ll_sys == "gcj":
        geoms = df.location.apply(lambda x: 
            shapely.Point([float(i) for i in x.split(",")]))
    elif ll_sys == "wgs":
        geoms = df.location.apply(lambda x: 
            shapely.Point(gcj_to_wgs(*[float(i) for i in x.split(",")])))
    
    return gpd.GeoDataFrame(df, geometry=geoms, crs=4326)

def plot_buffered_zones(gdf: gpd.GeoDataFrame, buffer_sizes: list, title: str) -> plt.Figure:
    """
    Creates a plot of buffered entrances for a given GeoDataFrame.

    Parameters
    ----------
    df_entrances : gpd.GeoDataFrame
        A GeoDataFrame containing the entrances to plot.
    buffer_sizes : list
        A list of integers representing buffer sizes in meters.
    title : str
        The title to be used for the plot.

    Returns
    -------
    plt.Figure
        The matplotlib Figure object with the plotted data.
    """
    buffer_sizes.sort(reverse=True)

    ori_crs = gdf.crs
    utm_crs = gdf.estimate_utm_crs()

    df = gdf.copy()
    df.to_crs(utm_crs, inplace=True)

    # Create buffers and store them as GeoDataFrames
    geoms = []
    for size in buffer_sizes:
        geoms.append({"geometry": df.buffer(size).unary_union, "buffer_size": size})
    buffers = gpd.GeoDataFrame(geoms, crs=utm_crs).to_crs(ori_crs)

    # Create a plot
    params = {'facecolor': 'none', 'linestyle': '--'} 
    fig, ax = plot_geodata(buffers, **params, tile_alpha=.8, alpha=0, legend=True) # provider=1, zoom=19
    buffers.plot(ax=ax, alpha=.6, categorical=True, legend=True, **params)
    gdf.plot(color='r', ax=ax)

    # Extract and annotate entrance names
    pattern = r"地铁站(\w*)口"
    for x, item in gdf.iterrows():
        match = re.search(pattern, item['name'])
        name = match.group(1) if match else item['name']
        ax.text(item.geometry.x, item.geometry.y, f" {name}", transform=ax.transData) 

    ax.set_title(title)

    return fig

def search_API(keywords: str, types: str = None, citycode: str = None,
                         show_fields = None, page_size: int = 20, page_num: int = 1,
                         to_geom: bool = True, ll_sys: str = 'wgs', key: str = None) -> pd.DataFrame:
    """
    Search locations using the AMap API with pagination.

    Parameters:
        - keywords (str): The search keyword.
        - types (str, optional): Filter by type. Defaults to None.
        - citycode (str, optional): The code of the city to search within. Defaults to None.
        - show_fields (str or list, optional): Specific fields to be returned. Defaults to None.
        - page_size (int, optional): Number of results to return per page. Defaults to 20.
        - page_num (int, optional): The current page number to fetch. Defaults to 1.
        - to_geom (bool, optional): If True, converts locations to geometries. Defaults to True.
        - ll_sys (str, optional): The coordinate system. Can be 'gcj' or 'wgs'. Defaults to 'wgs'.
        - key (str, optional): The API key for the AMap API. Must be provided.

    Returns:
    - DataFrame or GeoDataFrame: If to_geom is True, a GeoDataFrame with geometries is returned.
                                 Otherwise, a regular DataFrame is returned.
    Refs:
    - Usage API: https://lbs.amap.com/api/webservice/guide/api/newpoisearch#t5
    """
    if ll_sys not in ["gcj", "wgs"]:
        raise ValueError(f"Unsupported coordinate system: {ll_sys}")

    if key is None:
        raise ValueError("API key is required.")

    if isinstance(keywords, list):
        keywords = '|'.join(keywords)

    url = f"https://restapi.amap.com/v5/place/text"
    params = {
        'keywords': keywords,
        'key': key,
        'types': types,
        'citycode': citycode,
        'page_size': page_size,
        'page_num': page_num,
        'show_fields': show_fields
    }
    
    try:
        response = requests.get(url, params=params)
        logger.debug(f"{url}, {params}")
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        print(f"Error occurred: {e}")
        return pd.DataFrame()

    if data.get('status') != '1' or 'pois' not in data:
        return pd.DataFrame()

    df = pd.DataFrame(data['pois'])
    
    # Check if more data is available by comparing the number of returned results with the page size
    if len(df) == page_size:
        # Recursively call the function to get the next page
        next_page_df = search_API(
            keywords, types, citycode, show_fields, page_size, page_num + 1, to_geom, ll_sys, key
        )
        df = pd.concat([df, next_page_df], ignore_index=True)

    if to_geom and not df.empty:
        df = convert_to_geom(df, ll_sys)

    return df


if __name__ == "__main__":
    station_name = "下梅林地铁站" # 景田, 上梅林，车公庙
    show_fields = ['children', 'business', 'indoor']
    df_entrances = search_API(station_name, 150501, "0755", page_size=10, ll_sys='wgs', show_fields=show_fields, key=KEY)
    
    # FIXME 定位原则
    max_idx = df_entrances.parent.value_counts().index[0]
    plot_buffered_zones(df_entrances.query("parent == @max_idx"), [50, 100, 200], station_name);

