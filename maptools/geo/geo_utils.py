import geopandas as gpd
from shapely import LineString, Point

from math import atan2, cos, degrees, pi, radians, sin, sqrt

def _is_point(input):
    if not isinstance(input, Point):
        raise TypeError(
            f"Only Points are supported as arguments, got {input} {type(input)}"
        )

def azimuth(point1, point2):
    """
    Calculates euclidean bearing of line between two points.
    """
    _is_point(point1)
    _is_point(point2)
    angle = atan2(point2.x - point1.x, point2.y - point1.y)
    azimuth = degrees(angle)
    if angle < 0:
        azimuth += 360

    return azimuth


def point_gdf_to_linestring(df, geom_col_name):
    """
    Convert GeoDataFrame of Points to shapely LineString
    """
    if len(df) > 1:
        return LineString(df[geom_col_name].tolist())
    else:
        raise RuntimeError("DataFrame needs at least two points to make line!")
    
def convert_geom_to_utm_crs(gdf:gpd.GeoDataFrame, crs=None, inplace=True):
    if crs is None:
        crs = gdf.estimate_utm_crs().to_epsg()
    
    gdf.to_crs(crs, inplace=inplace)
    
    return gdf

def convert_geom_to_wgs(gdf, crs='epsg:4326', inplace=True):
    gdf.to_crs(crs, inplace=inplace)
    
    return gdf
