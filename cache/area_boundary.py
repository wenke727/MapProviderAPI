import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon
import pinyin

import urllib
import json

# coords_sys_output 还没有实现
def parse_polygon( polyline ):
    coords = []
    for index, points in enumerate( [ x.split(';') for x in polyline.split('|')] ):
        coords.append( Polygon(np.array( [x.split(',') for x in points]).astype(np.float)) )
    return MultiPolygon( coords )

def get_boundary_city_level( city, coords_sys_output='wgs', save_shp=False ):
    '''
    return area and sub_distict
    '''
    key = '71646457c61fe244423c40254833dc22'
    url = f'https://restapi.amap.com/v3/config/district?key={key}&keywords={urllib.parse.quote(city)}&subdistrict=1&extensions=all'
    print(url)
    req = urllib.request.Request(url)
    html_raw = urllib.request.urlopen(req).read()
    json_data = json.loads(html_raw)

    df = gpd.GeoDataFrame(json_data['districts'], crs={'init': 'epsg:4326'}  )
    print(list(df))
    df.loc[:, 'geometry'] = df.polyline.apply( lambda x: parse_polygon( x ) )
    df.drop(columns=['districts','polyline'], inplace=True)
    if save_shp:     df.to_file(f'input/{pinyin.get(city,format="strip" )}_boundary_city_level.shp', encoding='utf-8')
    df_sub = gpd.GeoDataFrame(json_data['districts'][0]['districts'])
    return df, df_sub

def get_boundary_district_level(city, coords_sys_output='wgs', save_shp=False):
    df_shp = gpd.GeoDataFrame()
    df, df_sub = get_boundary_city_level( city )
    for adcode in df_sub.adcode.values:
        shp,_ = get_boundary_city_level(adcode)
        df_shp = df_shp.append( shp, ignore_index=True )
    if save_shp: df_shp.to_file(f'input/{pinyin.get(city,format="strip" )}_boundary_district_level.shp', encoding='utf-8')
    return df_shp
    
if __name__ == "__main__":
    df_shp = get_boundary_district_level('广州')
    df_shp.plot()
    pass

