import numpy as np
from shapely import Point, LineString

from coordtransform import gcj_to_wgs


def check_ll(ll):
    if ll not in ['wgs', 'gcj']:
        raise ValueError("The 'll' parameter must be either 'wgs' or 'gcj'.")
    
    return True

def str_to_point(coords_str, ll, sep=',', decimals=6):
    check_ll(ll)
    stop_location = tuple(map(float, coords_str.split(sep)))
    if ll == 'wgs':
        stop_location = gcj_to_wgs(*stop_location)

    return Point(*np.round(stop_location, decimals))

def xyxy_str_to_linestring(coords_str, ll, sep=';', decimals=6):
    check_ll(ll)
    coords = [tuple(map(float, p.split(','))) 
                for p in coords_str.split(sep) if p]
        
    if ll == 'wgs':
        coords = [gcj_to_wgs(*coord) for coord in coords]
    
    return LineString(np.round(coords, decimals))

def xsys_str_to_linestring(xs, ys, ll, sep=',', decimals=6):
    check_ll(ll)

    xs = map(float, xs.split(sep))
    ys = map(float, ys.split(sep))
    coords = [[x, y] for x, y in zip(xs, ys)]

    if ll == 'wgs':
        coords = [gcj_to_wgs(*coord) for coord in coords]
    
    return LineString(np.round(coords, decimals))

