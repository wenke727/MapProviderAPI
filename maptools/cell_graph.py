# %%
from geo.distance import cal_pointwise_distance_geoseries as cal_distance
import pandas as pd
import numpy as np
import sys
import geopandas as gpd
from shapely import Point

records = gpd.read_file("../data/cells/traj_00011.geojson")
records.geometry = records.geometry.fillna(Point())
records.lac = records.lac.fillna(-1)
records.duration = records.duration.fillna(0)
records.loc[:, 'rid'] = 1

records.loc[80:90]

# %%


# %%

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

    df['dis_pre'] = cal_distance(df[Geometry], df[Geometry + '_pre'])
    df['dis_next'] = cal_distance(df[Geometry], df[Geometry + '_next'])
    df['dis_prenext'] = cal_distance(df[Geometry + '_pre'], df[Geometry + '_next'])

    prev_cond = (df[Rid + '_pre'] == df[Rid])
    next_cond = (df[Rid + '_next'] == df[Rid])

    # 以速度限制删除异常点
    if speedlimit:
        # 计算前后点时间差
        df['timegap_pre'] = df[Time + ''] - df[Time + '_pre']
        df['timegap_next'] = df[Time + '_next'] - df[Time + '']
        df['timegap_prenext'] = df[Time + '_next'] - df[Time + '_pre']

        # 计算前后点速度
        df['speed_pre'] = df['dis_pre'] / df['timegap_pre'] * 3.6
        df['speed_next'] = df['dis_next'] / df['timegap_next'] * 3.6
        df['speed_prenext'] = df['dis_prenext'] / df['timegap_prenext'] * 3.6
        if method == 'oneside':
            df = df[-(prev_cond & (df['speed_pre'] > speedlimit))]
        elif method == 'twoside':
            df = df[
                -(prev_cond & next_cond &
                    (df['speed_pre'] > speedlimit) &
                    (df['speed_next'] > speedlimit) &
                    (df['speed_prenext'] < speedlimit))]

    # 以距离限制删除异常点
    if dislimit:
        if method == 'oneside':
            df = df[
                -(prev_cond &
                  (df['dis_pre'] > dislimit))]
        elif method == 'twoside':
            df = df[
                -(prev_cond & next_cond &
                    (df['dis_pre'] > dislimit) &
                    (df['dis_next'] > dislimit) &
                    (df['dis_prenext'] < dislimit))]

    # 以角度限制删除异常点
    if anglelimit:
        # 余弦定理计算夹角
        angle_cos = (df['dis_pre'] ** 2+df['dis_next'] ** 2 -
                    df['dis_prenext'] **2 ) / (2 * df['dis_pre'] * df['dis_next'])
        angle_cos = np.maximum(np.minimum(angle_cos, 1), -1)
        df['angle'] = np.degrees(np.arccos(angle_cos))

        df = df[-(prev_cond & next_cond & (df['angle'] < anglelimit))]

    # df = df[data.columns]
    return df


_records = traj_clean_drift(records)
_records

# %%


# %%
