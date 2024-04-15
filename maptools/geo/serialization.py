import pandas as pd
import geopandas as gpd
from shapely import wkt


def read_csv_to_geodataframe(file_path, crs="EPSG:4326"):
    df = pd.read_csv(file_path)
    
    df['geometry'] = df['geometry'].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=crs)
    
    if "Unnamed: 0" in gdf.columns:
        gdf.drop(columns=['Unnamed: 0'], inplace=True)
    
    return gdf

def to_geojson(gdf, fn):
    if not isinstance(gdf, gpd.GeoDataFrame):
        print('Check the format of the gdf.')
        return False

    if 'geojson' not in str(fn):
        fn = f'{fn}.geojson'
    
    gdf.to_file(fn, driver="GeoJSON")

    return 

def csv_to_geodf(filepath, longitude='longitude', latitude='latitude', crs=4326, drop_xy=True):
    """
    Converts a CSV file into a GeoDataFrame with point geometries.

    Parameters:
    - filepath (str): The path to the CSV file to be converted.
    - longitude (str): The name of the column in the CSV that contains longitude values. Defaults to 'longitude'.
    - latitude (str): The name of the column in the CSV that contains latitude values. Defaults to 'latitude'.
    - crs (int): The coordinate reference system to be used for the GeoDataFrame. Defaults to 4326 (WGS84).
    - drop_xy (bool): Whether to drop the original longitude and latitude columns from the GeoDataFrame. Defaults to True.

    Returns:
    - gpd.GeoDataFrame: A GeoDataFrame containing point geometries constructed from the CSV data.

    Example:
    >>> geodf = csv_to_geodf('./data/trajs/sample.csv', longitude='lon', latitude='lat', crs=4326, drop_xy=True)
    """
    df = pd.read_csv(filepath)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[longitude], df[latitude], crs=crs)
    )
    
    if drop_xy:
        gdf.drop(columns=[longitude, latitude], inplace=True)
        
    return gdf


if __name__ == "__main__":
    # Usage Example:
    gdf = read_csv_to_geodataframe('your_csv_file.csv')
    print(gdf)
