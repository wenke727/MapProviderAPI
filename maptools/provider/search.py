import requests
import pandas as pd
from loguru import logger

from ..geo.coords_utils import convert_to_geom


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

