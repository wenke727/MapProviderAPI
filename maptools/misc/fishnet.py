# the default cordination system is wgs84

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon
import sys
sys.path.append('map_factory')
import CoordTransform_utils as ct
import haversine

def fishnet_shp( gdf,  x_step = 0.05, y_step = 0.05, coords_in_sys = 'wgs',coords_out_sys = 'gcj', shp_type='point' ):
    # only consider the first polygon 
    if isinstance(gdf, str):
        gdf = gpd.read_file(gdf)
    [x0,y0,x1,y1] = gdf.bounds.values[0]

    coords = gpd.GeoDataFrame( columns=['x', 'y'] )
    for x in np.arange( x0, x1, x_step ):
        for y in np.arange( y0, y1, y_step ):
            coords = coords.append( {'x':x, 'y':y}, ignore_index=True )

    if coords_in_sys != coords_out_sys:
        if coords_in_sys == 'wgs':
            x_temp = coords.apply( lambda x: ct.wgs84_to_gcj02( x.x, x.y )[0], axis =1 )
            y_temp = coords.apply( lambda x: ct.wgs84_to_gcj02( x.x, x.y )[1], axis =1 )
        else:
            x_temp = coords.apply( lambda x: ct.gcj02_to_wgs84( x.x, x.y )[0], axis =1 )
            y_temp = coords.apply( lambda x: ct.gcj02_to_wgs84( x.x, x.y )[1], axis =1 )
        coords.loc[:,'x'] = x_temp
        coords.loc[:,'y'] = y_temp

    coords.loc[:,'geometry'] = coords.apply( lambda x: Point( x.x, x.y ), axis=1 )
    coords = coords [ coords.within( gdf.geometry[0] ) ]
    bb = pd.DataFrame({'x_0':coords.x - x_step/2, 'x_1':coords.x + x_step/2, 'y_0':coords.y - y_step/2, 'y_1':coords.y + y_step/2})
    coords = pd.concat( [coords, bb], axis=1 )
    coords.loc[:,'dis'] = coords.apply( lambda x: haversine.haversine((x.y_0, x.x_0), (x.y_1, x.x_1)), axis =1 )
    if shp_type=='polygon':
        coords.geometry = coords.apply( lambda i: Polygon( [ (i.x_0, i.y_0), (i.x_0, i.y_1), (i.x_1, i.y_1), (i.x_1, i.y_0), (i.x_0, i.y_0) ] ), axis=1 )
    # coords.plot()
    return coords.reset_index(drop=True)


if __name__ == "__main__":
    boundary_fn = 'E:\GIS_Data\ShenzhenBoundary_wgs_citylevel.shp'
    coords = fishnet_shp( boundary_fn, coords_in_sys = 'wgs',coords_out_sys = 'wgs' )
    print(coords)
pass
