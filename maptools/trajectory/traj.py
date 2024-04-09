import sys
sys.path.append('../')

import shapely
import numpy as np
import pandas as pd
import geopandas as gpd
from loguru import logger
from geopandas import GeoDataFrame
import matplotlib.pyplot as plt

from tilemap import plot_geodata

from .base import BaseTrajectory
from .cleaner import simplify_traj_points
from .cleaner import clean_drift_traj_points
from .cleaner import filter_by_point_update_policy
from ..geo.serialization import read_csv_to_geodataframe, to_geojson
from ..geo.geo_utils import convert_geom_to_utm_crs, convert_geom_to_wgs
from ..utils.logger import logger_dataframe, make_logger


TRAJ_ID_COL = "tid"


class Trajectory(BaseTrajectory):
    def __init__(self, df:gpd.GeoDataFrame, traj_id:int, traj_id_col=TRAJ_ID_COL, obj_id=None, 
                 t=None, x=None, y=None, geometry='geometry', utm_crs=None, parent=None,
                 latlon=False):
        assert not (x is None and y is None) or geometry is not None, "Check Coordination"
        self.raw_df = df
        self.latlon = latlon
        self.utm_crs = utm_crs
        if self.latlon is False:
            self.raw_df = convert_geom_to_utm_crs(self.raw_df, utm_crs)

        self.points = self.raw_df.copy()

        self.traj_id_col = traj_id_col
        if traj_id_col not in self.points.columns:
            self.points.loc[:, traj_id_col] = traj_id

    def __str__(self):
        return (
            f"Trajectory {self.id} ({self.get_start_time()} to {self.get_end_time()}) "
            f"| Size: {self.size()} | Length: {round(self.get_length(), 1)}m\n"
        )

    def is_valid(self):
        return self.points.shape[0] > 1

    def clean_drift_points(self, method='twoside', speed_limit=None, dis_limit=None,
                           angle_limit=30, alpha=1, strict=False, verbose=False):
        """
        Clean drift in trajectory data by filtering out points based on speed, distance, 
        and angle thresholds.
        """

        if verbose:
            ori_size = len(self.points)

        self.points, mask = clean_drift_traj_points(
            self.points, col=[self.traj_id_col, 'dt', 'geometry'],
            method=method, speed_limit=speed_limit, dis_limit=dis_limit,
            angle_limit=angle_limit, alpha=alpha, strict=strict)

        if verbose:
            # logger_dataframe(mask, desc="clean_drift_traj_points mask:")
            cur_size = len(self.points)
            logger.debug(f"Clean drift points {ori_size} -> {cur_size}, cut down {(ori_size - cur_size) / ori_size * 100:.1f}%")

        return self.points

    def filter_by_point_update_policy(self, radius=500, verbose=False):
        """
        This function identifies points that are beyond a specified radius from the 
        last updated point, considering them as new updates and thereby reducing the 
        number of updates sent to a server in a tracking system.
        """
        if self.latlon:
            radius /= 110,000

        if verbose:
            ori_size = len(self.points)

        self.points = filter_by_point_update_policy(self.points, radius)

        if verbose:
            cur_size = len(self.points)
            logger.debug(f"Filter points {ori_size} -> {cur_size}, cut down {(ori_size - cur_size) / ori_size * 100:.1f}%")

        return self.points

    def simplify(self, tolerance=100, precision=6, verbose=False):
        if verbose:
            ori_size = len(self.points)
        self.points = simplify_traj_points(self.points, tolerance, precision)
        if verbose:
            cur_size = len(self.points)
            logger.debug(f"Simplify points {ori_size} -> {cur_size}, cut down {(ori_size - cur_size) / ori_size * 100:.1f}%")

        return self.points

    def preprocess(
        self,
        radius=500,
        speed_limit=0,
        dis_limit=0,
        angle_limit=45,
        alpha=3,
        strict=False,
        tolerance=None,
        verbose=True,
        plot=True,
    ):
        self.filter_by_point_update_policy(radius=radius, verbose=verbose)
        self.clean_drift_points(
            speed_limit=speed_limit,
            dis_limit=dis_limit,
            angle_limit=angle_limit,
            alpha=alpha,
            strict=strict,
            verbose=verbose,
        )

        if tolerance:
            self.points = self.simplify(tolerance, verbose=verbose)

        if plot:
            fig, ax = self.plot_preprocess_result()

    @property
    def crs(self):
        return f"EPSG:{self.points.crs.to_epsg()}"

    @property
    def get_epsg(self):
        return self.crs.to_epsg()

    def plot(self, *args, **kwargs):
        return self.points.plot(*args, **kwargs)

    def get_points(self, latlon=True):
        if latlon:
            return convert_geom_to_wgs(self.points)

        return self.points

    def plot_preprocess_result(self):
        """轨迹数据预处理结果可视化"""
        fig, ax = plot_geodata(self.raw_df.to_crs(4326), 
                            tile_alpha=.5, color='r', alpha=.7, marker='x', label='Remove')

        segs = self.to_line_gdf()
        segs.to_crs(4326).plot(ax=ax, color='b', alpha=.6)

        segs = self.to_line_gdf(self.raw_df)
        segs.to_crs(4326).plot(linestyle=':', ax=ax, color='red', alpha=.4)

        _pts = self.points.to_crs(4326)
        _pts.iloc[:-1].plot(color='b', ax=ax, facecolor='white', zorder=4, label='Keep')
        _pts.iloc[[-1]].plot(ax=ax, color='b', zorder=5, label='Dest')

        ax.legend(loc='best')

        return fig, ax

    def distance(self, other):
        # other = self.align_crs(other)
        return self.raw_df.distance(other)

    def align_crs(self, gdf:gpd.GeoDataFrame):
        if gdf.crs == self.crs:
            return gdf

        return gdf.to_crs(self.crs)

    def to_file(self, fn, raw=False):
        df = self.points if not raw else self.raw_df

        return to_geojson(df, fn)

if __name__ == "__main__":
    idx = 0
    fn = f"../../../ST-MapMatching/data/cells/{idx:03d}.csv"

    pts = read_csv_to_geodataframe(fn)
    self = traj = Trajectory(pts, traj_id=1)

    traj.preprocess(
        radius=300, 
        speed_limit=0, dis_limit=0, angle_limit=45, alpha=2, strict=False, 
        tolerance=300,
        verbose=True, plot=True
    )
