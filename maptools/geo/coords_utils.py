import shapely
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import Point, LineString

from .coordtransform import gcj_to_wgs


def check_ll(ll):
    if ll not in ['wgs', 'gcj']:
        raise ValueError("The 'll' parameter must be either 'wgs' or 'gcj'.")
    
    return True

def convert_to_geom(df: pd.DataFrame, ll_sys: str = 'wgs'):
    """
    Convert DataFrame locations to geometries based on the coordinate system.
    """
    check_ll(ll_sys)
    if ll_sys == "gcj":
        geoms = df.location.apply(lambda x: 
            shapely.Point([float(i) for i in x.split(",")]))
    elif ll_sys == "wgs":
        geoms = df.location.apply(lambda x: 
            shapely.Point(gcj_to_wgs(*[float(i) for i in x.split(",")])))
    
    return gpd.GeoDataFrame(df, geometry=geoms, crs=4326)


def str_to_point(coords_str, ll_sys, sep=',', decimals=6):
    check_ll(ll_sys)
    stop_location = tuple(map(float, coords_str.split(sep)))
    if ll_sys == 'wgs':
        stop_location = gcj_to_wgs(*stop_location)

    return Point(*np.round(stop_location, decimals))


def xyxy_str_to_linestring(coords_str, ll_sys, sep=';', decimals=6):
    check_ll(ll_sys)
    coords = [tuple(map(float, p.split(','))) 
                for p in coords_str.split(sep) if p]
        
    if ll_sys == 'wgs':
        coords = [gcj_to_wgs(*coord) for coord in coords]
    
    return LineString(np.round(coords, decimals))


def xsys_str_to_linestring(xs, ys, ll_sys, sep=',', decimals=6):
    check_ll(ll_sys)

    xs = map(float, xs.split(sep))
    ys = map(float, ys.split(sep))
    coords = [[x, y] for x, y in zip(xs, ys)]

    if ll_sys == 'wgs':
        coords = [gcj_to_wgs(*coord) for coord in coords]
    
    return LineString(np.round(coords, decimals))


