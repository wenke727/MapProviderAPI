#%%
import shapely
import numpy as np
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
from loguru import logger

from base import BaseTrajectory
from utils import convert_geom_to_utm_crs, convert_geom_to_wgs
from serialization import read_csv_to_geodataframe

TRAJ_ID_COL = "tid"

#%%

def logger_dataframe(df, desc="", level='debug'):
    getattr(logger, level)(f"{desc}/n{df}")


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

def get_outliers_thred_by_iqr(series, alpha=3):
    """
    Returns a series of indexes of row that are to be considered outliers
    using the quantiles of the data.
    """
    q25, q75 = series.quantile((0.25, 0.75))
    iqr = q75 - q25
    q_high = q75 + alpha * iqr
    q_low = q25 - alpha * iqr
    # return the row indexes that are over/under the calculated threshold
    
    return q_low, q_high


def traj_clean_drift(data:GeoDataFrame, col=[TRAJ_ID_COL, 'dt', 'geometry'],
                     method='twoside',
                     speedlimit=None,
                     dislimit=None,
                     anglelimit=30, alpha=1):
    def _preprocess(df):
        for i in [Rid, Geometry, Time]:
            df[i + '_pre'] = df[i].shift()
            df[i + '_next'] = df[i].shift(-1)
            
        df['dis_pre'] = df[Geometry].distance(df[Geometry + '_pre'])
        df['dis_next'] = df[Geometry].distance(df[Geometry + '_next'])
        df['dis_prenext'] = df[Geometry + '_pre'].distance(df[Geometry + '_next'])

        df['timegap_pre'] = df[Time] - df[Time + '_pre']
        df['timegap_next'] = df[Time + '_next'] - df[Time]
        df['timegap_prenext'] = df[Time + '_next'] - df[Time + '_pre']

        df['speed_pre'] = df['dis_pre'] / df['timegap_pre'] * 3.6
        df['speed_next'] = df['dis_next'] / df['timegap_next'] * 3.6
        df['speed_prenext'] = df['dis_prenext'] / df['timegap_prenext'] * 3.6
        
        return df

    [Rid, Time, Geometry] = col
    df = data[col].copy()
    df = df.drop_duplicates(subset=[Rid, Time])
    df = df.sort_values(by=[Rid, Time])

    if data.crs.to_epsg() == 4326:
        convert_geom_to_utm_crs(df, inplace=True)

    df = _preprocess(df)

    # traj mask
    traj_mask = (df[Rid + '_pre'] == df[Rid])
    if method == 'twoside':
        traj_mask = traj_mask & (df[Rid + '_next'] == df[Rid])

    if speedlimit is None:
        _, speedlimit = get_outliers_thred_by_iqr(df['speed_pre'], alpha)
    if method == 'oneside':
        df = df[-(traj_mask & (df['speed_pre'] > speedlimit))]
    elif method == 'twoside':
        df = df[~(traj_mask
                    & (df['speed_pre'] > speedlimit) | (df['speed_next'] > speedlimit)
                    )] # & (df['speed_prenext'] < speedlimit)

    if dislimit:
        _, speedlimit = get_outliers_thred_by_iqr(df['dis_pre'], alpha)
    if method == 'oneside':
        df = df[
            ~(traj_mask
                & (df['dis_pre'] > dislimit))]
    elif method == 'twoside':
        df = df[
            ~(traj_mask
                & ((df['dis_pre'] > dislimit) | (df['dis_next'] > dislimit))
                )] # & (df['dis_prenext'] < dislimit)

    if anglelimit:
        df['angle'] = calculate_angle_between_sides(df['dis_pre'], df['dis_next'], df['dis_prenext'])
        df = df[-(traj_mask & (df['angle'] < anglelimit))]

    # df = df[data.columns]
    return df


class Trajectory(BaseTrajectory):
    def __init__(self, df:gpd.GeoDataFrame, traj_id:int, traj_id_col=TRAJ_ID_COL, obj_id=None, 
                 t=None, x=None, y=None, geometry='geometry', crs="epsg:4326", parent=None,
                 latlon=False):
        
        assert not (x is None and y is None) or geometry is not None, "Check Coordination"
        self.raw_df = df.copy()
        self.points = df
        self.latlon = latlon
        
        if self.latlon is False:
            self.points = convert_geom_to_utm_crs(self.points)
            
        if traj_id_col not in self.points.columns:
            self.points.loc[:, traj_id_col] = traj_id
    
    def clear_drift_points(self):
        return NotImplementedError

    @property
    def crs(self):
        return self.points.crs

    @property
    def get_epsg(self):
        return self.crs.to_epsg()
    
    def plot(self, *args, **kwargs):
        return self.points.plot()
    
    def get_points(self, ll=True):
        return convert_geom_to_wgs(self.points)

# if __name__ == "__main__":

# fn = "../../../ST-MapMatching/data/cells/004.csv"
# fn = "../../../ST-MapMatching/data/cells/014.csv"
fn = "../../../ST-MapMatching/data/cells/420.csv"
pts = read_csv_to_geodataframe(fn)

traj = Trajectory(pts, traj_id=1, )
traj.points

_traj = traj_clean_drift((traj.points), anglelimit=None)
# logger_dataframe(_traj)

ax = pts.plot(color='r')
_traj.plot(ax=ax, color='b')



# %%
