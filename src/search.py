import shapely
import requests
import pandas as pd
import geopandas as gpd
from eviltransform import gcj2wgs

from cfg import KEY


def search_API(keyword: str, types: str = None, citycode: str = None,
               to_geom: bool = True, ll_sys: str = 'wgs', key: str = KEY) -> pd.DataFrame:
    """
    Search locations using the AMap API.

    Parameters:
    - keyword (str): The search keyword.
    - types (str, optional): Filter by type. Defaults to None.
    - citycode (str, optional): The code of the city to search within. Defaults to None.
    - key (str, optional): The API key for the AMap API. Defaults to the one in the cfg module.
    - to_geom (bool, optional): If True, converts locations to geometries. Defaults to True.
    - ll_sys (str, optional): The coordinate system. Can be 'gcj' or 'wgs'. Defaults to 'wgs'.

    Returns:
    - DataFrame or GeoDataFrame: If to_geom is True, a GeoDataFrame with geometries is returned. 
                                  Otherwise, a regular DataFrame is returned.
    """
    # 检查坐标系统是否有效
    if ll_sys not in ["gcj", "wgs"]:
        raise ValueError(f"Unsupported coordinate system: {ll_sys}")

    url = f"https://restapi.amap.com/v5/place/text?keywords={keyword}&key={key or ''}"

    if types:
        url += f"&types={types}"
    if citycode:
        url += f"&citycode={citycode}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        # 可以考虑记录错误，或者返回一个空的DataFrame
        print(f"Error occurred: {e}")
        return pd.DataFrame()

    if data.get('status') != '1' or 'pois' not in data:
        return pd.DataFrame()

    df = pd.DataFrame(data['pois'])
    
    if to_geom:
        df = convert_to_geom(df, ll_sys)

    return df

def convert_to_geom(df: pd.DataFrame, ll_sys: str) -> gpd.GeoDataFrame:
    """
    Convert DataFrame locations to geometries based on the coordinate system.
    """
    if ll_sys == "gcj":
        geoms = df.location.apply(lambda x: shapely.Point([float(i) for i in x.split(",")]))
    elif ll_sys == "wgs":
        geoms = df.location.apply(lambda x: shapely.Point(gcj2wgs(*[float(i) for i in x.split(",")])))
    return gpd.GeoDataFrame(df, geometry=geoms)

if __name__ == "__main__":
    df_1 = search_API("上梅林地铁站", 150501, "0755", ll_sys='gcj')
    df_1.plot()

