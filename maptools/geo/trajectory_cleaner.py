import numpy as np
import pandas as pd
import geopandas as gpd

from geo.distance import cal_pointwise_distance_geoseries as cal_distance


def calculate_angle_between_sides(adjacent_side1, adjacent_side2, opposite_side):
    """
    根据相邻两边和对边长度计算夹角（以度为单位）, 余弦定理计算夹角。

    参数:
    adjacent_side1 (array-like): 第一条相邻边的长度。
    adjacent_side2 (array-like): 第二条相邻边的长度。
    opposite_side (array-like): 对边的长度。

    返回:
    angles (array-like): 计算出的夹角值（以度为单位）。
    """
    # 使用余弦定理计算夹角的余弦值
    angle_cos = (adjacent_side1 ** 2 + adjacent_side2 ** 2 - opposite_side ** 2) / (2 * adjacent_side1 * adjacent_side2)
    angle_cos = np.clip(angle_cos, -1, 1)  # 确保值在[-1, 1]范围内

    # 计算夹角并转换为度
    angles = np.degrees(np.arccos(angle_cos))

    return angles

def traj_clean_drift(data, col=['rid', 'dt', 'geometry'],
                     method='twoside',
                     speedlimit=None,
                     dislimit=5000,
                     anglelimit=30):
    [Rid, Time, Geometry] = col
    df = data.copy()
    df = df.drop_duplicates(subset=[Rid, Time])
    df = df.sort_values(by=[Rid, Time])

    # 计算前后点距离、时间差、速度
    for i in [Rid, Geometry, Time]:
        df[i + '_pre'] = df[i].shift()
        df[i + '_next'] = df[i].shift(-1)

    # TODO: 转换
    df['dis_pre'] = cal_distance(df[Geometry], df[Geometry + '_pre'])
    df['dis_next'] = cal_distance(df[Geometry], df[Geometry + '_next'])
    df['dis_prenext'] = cal_distance(df[Geometry + '_pre'], df[Geometry + '_next'])

    oneside_mask = (df[Rid + '_pre'] == df[Rid])
    twoside_mask = oneside_mask & (df[Rid + '_next'] == df[Rid])

    if speedlimit:
        df['timegap_pre'] = df[Time] - df[Time + '_pre']
        df['timegap_next'] = df[Time + '_next'] - df[Time]
        df['timegap_prenext'] = df[Time + '_next'] - df[Time + '_pre']

        df['speed_pre'] = df['dis_pre'] / df['timegap_pre'] * 3.6
        df['speed_next'] = df['dis_next'] / df['timegap_next'] * 3.6
        df['speed_prenext'] = df['dis_prenext'] / df['timegap_prenext'] * 3.6
        
        if method == 'oneside':
            df = df[-(oneside_mask & (df['speed_pre'] > speedlimit))]
        elif method == 'twoside':
            df = df[
                -(twoside_mask &
                    (df['speed_pre'] > speedlimit) &
                    (df['speed_next'] > speedlimit) &
                    (df['speed_prenext'] < speedlimit))]

    if dislimit:
        if method == 'oneside':
            df = df[
                -(oneside_mask &
                  (df['dis_pre'] > dislimit))]
        elif method == 'twoside':
            df = df[
                -(twoside_mask &
                    (df['dis_pre'] > dislimit) &
                    (df['dis_next'] > dislimit) &
                    (df['dis_prenext'] < dislimit))]

    if anglelimit:
        df['angle'] = calculate_angle_between_sides(df['dis_pre'], df['dis_next'], df['dis_prenext'])
        df = df[-(twoside_mask & (df['angle'] < anglelimit))]

    # df = df[data.columns]
    return df


if __name__ == "__main__":
    _records = traj_clean_drift(records)
    _records
    