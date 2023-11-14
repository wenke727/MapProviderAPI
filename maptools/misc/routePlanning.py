import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, LineString
import urllib
import json
import time
import pinyin
import random
import multiprocessing as mp
from multiprocessing import Pool
from math import ceil
# import matplotlib.pyplot as plt

import sys
# sys.path.append('map')
import CoordTransform_utils as ct
import coordTransfrom_shp as ct_shp

def get_key( fn = 'input/amap_key_personal.xlsx' ):
    return pd.read_excel( fn ).key.values


def route_planning(o, d, i=-1, mode='bicycling', keys=['71646457c61fe244423c40254833dc22']):
    if mode == 'driving':
        url_base = 'https://restapi.amap.com/v3/direction/driving'
    else:
        url_base = 'https://restapi.amap.com/v4/direction/bicycling'

    url = url_base + f"?key={keys[random.randint(0, len(keys)-1)]}&origin={o}&destination={d}"

    first_level = 'data' if mode in( ['bicycling'] ) else 'route'
    req = urllib.request.Request(url)
    html_raw = urllib.request.urlopen(req).read()
    json_data = json.loads(html_raw)

    polyline = ';'.join([  x['polyline'] for x in json_data[first_level]['paths'][0]['steps'] ])
    polyline = LineString( [ np.array(x.split(',')).astype(np.float) for x in polyline.split(';') ] )
    
    # base attributes
    result = {'o':o, 
              'd':d, 
              'distance': int(json_data[first_level]['paths'][0]['distance']), 
              'duration': int(json_data[first_level]['paths'][0]['duration']),
              'geometry':polyline
            }
    # add driving attributes
    if mode == 'driving':
        atts = ['strategy','toll_distance','tolls','traffic_lights']
        for att in atts:
            result[att] = json_data[first_level]['paths'][0][att]

        tmcs = []
        for x in json_data[first_level]['paths'][0]['steps']:
            tmcs += x['tmcs']
        tmcs = pd.DataFrame(tmcs)
        tmcs.loc[:, 'distance'] = tmcs.distance.astype(np.int16)
        tmcs_percentage = tmcs.groupby('status').sum()
        atts = ['smooth','slow','congested','verycongested','unknown']
        for i, status in  enumerate(['??','??','??','????','??']):
            try:
                result[atts[i]] = tmcs_percentage.loc[status].values[0] / tmcs.distance.sum()
            except:
                pass
    
    if i >= 0:   result['index'] = i; print(f'{i}, {url}' )
    return result


def route_planning_batch( odi_ls, keys ):
    res = []
    for index, item in enumerate(odi_ls):
        res.append( route_planning(*item, keys=keys ) )   
    return res

def multi_process_start(fn):
    ods = pd.read_excel(fn)
    origins = ods.apply( lambda i:  ','.join([str(x) for x in ct.wgs84_to_gcj02(i.lon_0, i.lat_0)]), axis=1 )
    destinations = ods.apply( lambda i:  ','.join([str(x) for x in ct.wgs84_to_gcj02(i.lon_1, i.lat_1)]), axis=1 )
    i = range(0,len(origins))
    odi_ls = list(zip(origins, destinations,i))

    # start
    flist = odi_ls
    num = (int(mp.cpu_count()) if int(mp.cpu_count()) < 12 else 12 )
    n = int(ceil(len(flist) / num  ))
    print(f"process: cpu {mp.cpu_count()}, total length: {len(flist)}, {n} records each core")
    time.sleep(3)
    pool = mp.Pool( processes = num )
    result = []

    # keys = get_key()
    for i in range( num ):
        # result.append( pool.apply_async( route_planning_batch, args= (flist[n*i: n*(i+1)], keys, ) ) )
        result.append( pool.apply_async( route_planning_batch, args= (flist[n*i: n*(i+1)], ['71646457c61fe244423c40254833dc22'], ))) 
    pool.close()
    pool.join()
    print( 'DONE' )
    # end

    df = gpd.GeoDataFrame()
    for res in result:
        if len(res.get()) >0:
            df = df.append( res.get(), ignore_index = True )
    df.geometry = df.geometry
    df = gpd.GeoDataFrame(df, geometry = df.geometry)
    # df = ct_shp.gdf_gcj_to_wgs(df)
    print( df.shape, type(df) )
    df.to_file('d:/route_planning.shp', encoding = 'utf-8')



if __name__ == "__main__":
    print('start')
    # road_level = multi_process_start(fn ='d:/Route_0328.xlsx')
    road_level = multi_process_start(fn ='E:/Data/GBA/cities_poi.xlsx')
    pass


#%%
# # test
# # TODO ???????

# o = '113.941704,22.572923'
# d = '113.903148,22.887382'

# res = [route_planning(o, d, mode='driving')]
# res.append( route_planning(d, o, mode='driving'))

# gpd.GeoDataFrame(res)

# def multi_process( fun, *args ):
#     # start
#     num = (int(mp.cpu_count()) if int(mp.cpu_count()) < 12 else 12 )
#     n = int(ceil(len(flist) / num  ))
#     print(f"process: cpu {mp.cpu_count()}, total length: {len(flist)}, {n} records each core")
#     time.sleep(3)
#     pool = mp.Pool( processes = num )
#     result = []
#     for i in range( num ):
#         result.append( pool.apply_async( fun, args= (flist[n*i: n*(i+1)], keys, ) ) )
#     pool.close()
#     pool.join()
#     print( "Sub-process(es) done." )
#     # end
#     return result

