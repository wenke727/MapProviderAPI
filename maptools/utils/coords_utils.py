import numpy as np
from shapely import Point, LineString

from ..coordtransform import gcj_to_wgs


def check_ll(ll):
    if ll not in ['wgs', 'gcj']:
        raise ValueError("The 'll' parameter must be either 'wgs' or 'gcj'.")
    
    return True

def str_to_point(coords_str, ll):
    check_ll(ll)
    stop_location = tuple(map(float, coords_str.split(',')))
    if ll == 'wgs':
        stop_location = gcj_to_wgs(*stop_location)

    return Point(*stop_location)

def str_to_point(coords_str, ll):
    check_ll(ll)
    coords = [tuple(map(float, p.split(','))) 
                for p in coords_str.split(';') if p]
        
    if ll == 'wgs':
        coords = [gcj_to_wgs(*coord) for coord in coords]
    
    return LineString(coords)

