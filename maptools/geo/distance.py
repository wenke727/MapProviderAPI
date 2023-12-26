import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import Point


def cal_pointwise_distance_geoseries(arr1, arr2, align=True):
    """calculate two geoseries distance

    Args:
        arr1 (gpd.GeoSeries): Geom array 1.
        arr2 (gpd.GeoSeries): Geom array 2.
        align (bool, optional): Align the two  Geom arrays. Defaults to True.

    Returns:
        pd.Series: Distance array
    """
    if isinstance(arr1, pd.Series):
        arr1 = gpd.GeoSeries(arr1)
    if isinstance(arr2, pd.Series):
        arr2 = gpd.GeoSeries(arr2)
    arr1.reset_index(drop=True)
    arr2.reset_index(drop=True)

    crs_1 = arr1.crs
    crs_2 = arr2.crs
    assert crs_1 is not None or crs_2 is not None, "arr1 and arr2 must have one has crs"
    
    if align:
        if crs_1 is None:
            arr1.set_crs(crs_2, inplace=True)
        if crs_2 is None:
            arr2.set_crs(crs_1, inplace=True)
    else:
        assert crs_1 is not None and crs_2 is not None, "Turn `align` on to align geom1 and geom2"

    if arr1.crs.to_epsg() == 4326:
        crs = arr1.estimate_utm_crs()
        dist = arr1.to_crs(crs).distance(arr2.to_crs(crs))
    else:
        dist = arr1.distance(arr2)

    return dist

def cal_distance_matrix_geoseries(points1, points2, align=True):
    """Generate a pairwise distance matrix between two GeoSeries.

    Args:
        arr1 (gpd.GeoSeries): Geom array 1.
        arr2 (gpd.GeoSeries): Geom array 2.

    Returns:
        pd.DataFrame: A distance matrix of size n x m
    """
    n, m = len(points1), len(points2)

    # Replicate arr1 and arr2
    repeated_arr1 = points1.repeat(m).reset_index(drop=True)
    tiled_arr2 = np.tile(points2, n)
    repeated_arr2 = gpd.GeoSeries(tiled_arr2, crs=points2.crs).reset_index(drop=True)
    
    # Calculate distances
    distances = cal_pointwise_distance_geoseries(repeated_arr1, repeated_arr2, align=align)

    # Reshape into matrix
    distance_matrix = distances.values.reshape(n, m)
    
    return pd.DataFrame(distance_matrix, index=points1.index, columns=points2.index)


if __name__ == "__main__":
    # matrix = haversine_matrix(traj_points, points_, xy=True)
    
    # 创建两个测试 GeoSeries
    points1 = gpd.GeoSeries([Point(0, 0), Point(1, 1)])
    points2 = gpd.GeoSeries([Point(1, 1), Point(0, 0), Point(1, 1)])

    # 确保两个 GeoSeries 使用相同的 CRS
    points1.set_crs(epsg=4326, inplace=True)
    points2.set_crs(epsg=4326, inplace=True)

    # 计算两个 GeoSeries 之间的距离矩阵
    distance_matrix = cal_distance_matrix_geoseries(points1, points2)
    distance_matrix

