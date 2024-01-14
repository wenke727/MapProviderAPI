#%%
import sys
sys.path.append('../')

import shapely
import numpy as np
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
from loguru import logger

from base import BaseTrajectory
from geo.geo_utils import convert_geom_to_utm_crs, convert_geom_to_wgs
from geo.serialization import read_csv_to_geodataframe
from cleaner import clean_drift_traj_points, filter_by_point_update_policy

from utils.logger import logger_dataframe, make_logger

TRAJ_ID_COL = "tid"

#%%

class Trajectory(BaseTrajectory):
    def __init__(self, df:gpd.GeoDataFrame, traj_id:int, traj_id_col=TRAJ_ID_COL, obj_id=None, 
                 t=None, x=None, y=None, geometry='geometry', crs="epsg:4326", parent=None,
                 latlon=False):
        assert not (x is None and y is None) or geometry is not None, "Check Coordination"
        self.raw_df = df
        self.latlon = latlon
        if self.latlon is False:
            self.raw_df = convert_geom_to_utm_crs(self.raw_df)

        self.points = self.raw_df.copy()
                
        self.traj_id_col = traj_id_col
        if traj_id_col not in self.points.columns:
            self.points.loc[:, traj_id_col] = traj_id

    def clean_drift_points(self, method='twoside', speed_limit=None, dis_limit=None,
                           angle_limit=30, alpha=1, strict=False, verbose=False):
        """
        Clean drift in trajectory data by filtering out points based on speed, distance, 
        and angle thresholds.
        """
   
        if verbose:
            ori_size = len(self.points)
     
        self.points = clean_drift_traj_points(
            self.points, col=[self.traj_id_col, 'dt', 'geometry'],
            method=method, speed_limit=speed_limit, dis_limit=dis_limit,
            angle_limit=angle_limit, alpha=alpha, strict=strict)
        
        if verbose:
            cur_size = len(self.points)
            logger.debug(f"Clean drift points {ori_size} -> {cur_size}, cut down {(ori_size - cur_size) / ori_size * 100:.1f}%")
        
        return pts

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
        
    @property
    def crs(self):
        return self.points.crs

    @property
    def get_epsg(self):
        return self.crs.to_epsg()
    
    def plot(self, *args, **kwargs):
        return self.points.plot()
    
    def get_points(self, latlon=True):
        if latlon:
            return convert_geom_to_wgs(self.points)
        
        return self.points


# if __name__ == "__main__":
idx = 420
fn = f"../../../ST-MapMatching/data/cells/{idx:03d}.csv"

pts = read_csv_to_geodataframe(fn)
traj = Trajectory(pts, traj_id=1)

traj.filter_by_point_update_policy(radius=500, verbose=True)
traj.clean_drift_points(speed_limit=0, dis_limit=0, angle_limit=45, alpha=2, 
                        strict=False, verbose=True)

# logger_dataframe(traj.points)

ax = traj.raw_df.plot(color='b')
traj.points.plot(color='r', ax=ax)

# _traj.index



# %%
