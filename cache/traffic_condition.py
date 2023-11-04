#%%
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
from math import ceil
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

import os, sys
sys.path.append( os.path.dirname(__file__) )
from fishnet import fishnet_shp
import coordTransfrom_shp as ct_shp
from area_boundary import get_boundary_city_level


def get_key( fn = os.path.join( os.path.dirname(__file__), 'input/amap_key_personal.xlsx' )):
    return pd.read_excel( fn ).key.values

def get_roads_conditions_city(city, level=6, save_shp=False):
    df_shp, _ = get_boundary_city_level(city)
    coords = fishnet_shp( df_shp, x_step = 0.065, y_step = 0.055, coords_in_sys = 'gcj', coords_out_sys = 'gcj' )
    df_tc = get_roads_conditions_batch(coords, keys=['71646457c61fe244423c40254833dc22'], level=level)
    df_tc.loc[:, 'time'] = time.strftime("%Y%m%d %H:%M",time.localtime())
    if save_shp: df_tc.to_file( f'output/{pinyin.get(city,format="strip" )}_level_{level}_{time.strftime("%Y%m%d_%H%M",time.localtime())}.geojson', driver='GeoJSON', encoding='utf-8' )
    return df_tc

def get_roads_conditions(  area, keys=['71646457c61fe244423c40254833dc22'], level=6,  coords_sys_output='wgs'):
    '''
    url: 矩形对角线不能超过10公里, 输入坐标为国测局坐标体系
    TODO 增加每个点的采集情况，后续若是没有的话，可以去掉, level 为道路等级
    keys=['71646457c61fe244423c40254833dc22']; level=6; coords_sys_output='wgs'
    '''
    url = f'https://restapi.amap.com/v3/traffic/status/rectangle?rectangle={area}&key={keys[random.randint(0, len(keys)-1)]}&extensions=all&level={level}'
    time_str = time.strftime('%Y-%m-%dT%H:%M'+":00",time.localtime())

    req = urllib.request.Request(url)
    html_raw = urllib.request.urlopen(req).read()
    json_data = json.loads(html_raw)

    if json_data['status'] =='0' or len(json_data['trafficinfo']['roads'])==0:
        df_roads =  gpd.GeoDataFrame()
    else:
        df_roads             = gpd.GeoDataFrame(json_data['trafficinfo']['roads'], crs={'init': 'epsg:4326'} ).reset_index().rename(columns={'index':'road_id'})
        df_roads.speed       = df_roads.speed.astype(float)
        df_roads.status      = df_roads.status.astype(int)
        df_roads['geometry'] = df_roads.polyline.apply( lambda a: LineString( np.array( [  x.split(',') for x in a.split(';') ] ).astype(np.float) ) )
        if coords_sys_output=='wgs':    
            df_roads         = ct_shp.polyline_gcj_to_wgs( df_roads )
    return df_roads

def get_roads_conditions_batch(coords, keys, level = 6 ):
    # print( f'coords: {coords.shape[0]}' )
    df_tc = gpd.GeoDataFrame(  )
    log = []
    for index, item in coords.iterrows():
        area = '%.6f,%.6f;%.6f,%.6f'%( item.x_0, item.y_0, item.x_1, item.y_1  )
        tc_info = get_roads_conditions( area, keys=keys, level=level )
        log.append( {'id':index, 'len':len(tc_info)} )
        # print( f'{index},area[{area}]: {len(tc_info)}' )
        df_tc = df_tc.append( tc_info, ignore_index=True )
        time.sleep( 0.5 )
    return df_tc, pd.DataFrame(log)

def multi_process_start( city, x_step = 0.06, y_step = 0.04, add_buffer =True, save_shp_path = None, overlay_intersction=True ):
    '''
    多线程获取交通态势，坐标系默认 wgs84
    Parameter:
    city='深圳'; x_step = 0.02; y_step = 0.02; add_buffer = True; save_shp_path = './output/tc/'
    '''
    keys        = get_key()
    area_shp_fn = f'{os.path.dirname(__file__)}/input/{city}.geojson'
    coords_fn   = f'{os.path.dirname(__file__)}/input/coords_{x_step}_{y_step}.geojson'

    if not os.path.exists( area_shp_fn ):
        df_shp, _ = get_boundary_city_level(city)
        df_shp.to_file( area_shp_fn, driver='GeoJSON' )
    else:
        df_shp = gpd.read_file( area_shp_fn )

    if add_buffer:
        df_shp.geometry = df_shp.buffer( max( x_step, y_step )*1.05 )

    if not os.path.exists( coords_fn ):
        coords = fishnet_shp( df_shp, x_step, y_step, coords_in_sys = 'gcj', coords_out_sys = 'gcj' )
        print(f'creat fishnet({x_step}, {y_step})')
    else:
        coords = gpd.read_file( coords_fn )

    # multi
    print( f'coords: {coords.shape[0]}' )
    n_jobs = int(mp.cpu_count()//2) 
    df = coords.copy()
    df.loc[:,'group'] = df.index % n_jobs
    df_group = df.groupby('group')
    results = Parallel(n_jobs=n_jobs, verbose=0)(delayed(get_roads_conditions_batch)( group, keys ) for name, group in df_group) # TODO: 访问速度不宜过快，会导致无结果反馈的

    roads = pd.concat(np.array(results)[:,0])
    if overlay_intersction:
        roads = gpd.overlay( roads, df_shp, how='intersection' )
    # roads.plot()
    
    print(f'len: {len(roads)}')
    log =  pd.concat(np.array(results)[:,1]).sort_values('id')
    coords.query( f"index in {list(log.query('len>0').id.values)}", inplace=True )
    coords.to_file( coords_fn, driver='GeoJSON' )
    roads.loc[:, 'time'] =  time.strftime('%Y-%m-%dT%H:%M'+":00",time.localtime())
    roads.loc[(roads.speed < 10) & (roads.status==0), 'status'] = 3

    if save_shp_path:
        roads.to_file( os.path.join( save_shp_path, f'{pinyin.get(city,format="strip" )}_{time.strftime("%Y%m%d_%H%M",time.localtime())}.geojson'), driver='GeoJSON', encoding='utf-8' )
    return roads

def obtain_road_network_level(city='深圳', save_csv=True):
    '''
    指定道路等级，下面各值代表的含义：
    1：高速（京藏高速）
    2：城市快速路、国道(西三环、103国道)
    3：高速辅路（G6辅路）
    4：主要道路（长安街、三环辅路路）
    5：一般道路（彩和坊路）
    6：无名道路
    '''
    ls_road_level = []
    ls_road_level_name = []
    for i in range( 1, 7 ):
        ls_road_level.append( get_roads_conditions_city(city, level=i) )
        ls_road_level_name.append( ls_road_level[i-1].name.unique() )

    result = [ls_road_level_name[0]]
    df_road_level = pd.DataFrame( {'name':ls_road_level_name[0],'level':1} )
    for i in range( 1, 6 ):
        roads = np.setdiff1d( ls_road_level_name[i],  ls_road_level_name[i-1]  )
        result.append(roads)
        df = pd.DataFrame( {'name':roads, 'level': i+1} )
        df_road_level = df_road_level.append( df, ignore_index=True )
    # df_road_level.groupby('level').count()
    if save_csv: df_road_level.to_csv( f'output/{pinyin.get(city,format="strip" )}_road_level.csv',  encoding='utf-8' )
    return df_road_level

#%%
if __name__ == "__main__":
    roads_sz = multi_process_start(city='深圳', x_step = 0.04, y_step = 0.03, save_shp_path = './output/tc/')
    roads_sz.plot()
    roads_sz
    # roads.to_file( 'Shenzhen_roads_traffic_condition_wgs.geojson', driver='GeoJSON' )
    # road_level = obtain_road_network_level(city='深圳')
    pass


