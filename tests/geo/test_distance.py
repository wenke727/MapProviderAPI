import pytest
import numpy as np
import geopandas as gpd
from shapely import Point

from maptools.geo.distance import cal_distance_matrix_geoseries


def test_cal_distance_matrix_geoseries():

    # 创建两个测试 GeoSeries
    points1 = gpd.GeoSeries([Point(0, 0), Point(1, 1)])
    points2 = gpd.GeoSeries([Point(1, 1), Point(0, 0), Point(1, 1)])

    # 确保两个 GeoSeries 使用相同的 CRS
    points1.set_crs(epsg=4326, inplace=True)
    points2.set_crs(epsg=4326, inplace=True)

    ans = np.array(
        [[156989.23339984,      0.        , 156989.23339984],
         [     0.        , 156989.23339984,      0.        ]]
    )
    
    # 计算两个 GeoSeries 之间的距离矩阵
    distance_matrix = cal_distance_matrix_geoseries(points1, points2)
    distance_matrix


    assert np.allclose(distance_matrix, ans)

    # 如果无法确定预期结果，可以编写其他断言来验证结果的一些性质
    # assert len(result) == len(geoseries), "结果长度不符合预期"
    # assert isinstance(result[0], (int, float)), "结果类型不符合预期"
