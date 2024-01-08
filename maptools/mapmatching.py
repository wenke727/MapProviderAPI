import geopandas as gpd


def get_fake_cell_traj():
    from shapely import Point
    geoms = [
        Point(113.923483, 22.524037), # 南山
        Point(114.055636,22.539872), # 福田（11）
        Point(114.06017,22.548627), # 少年宫（3）
        # Point(114.059415,22.570459), # 上梅林（4）
        Point(114.060432,22.56844), # 上梅林（9）
    ]
    
    return gpd.GeoDataFrame({'geometry': geoms}, crs=4326)

