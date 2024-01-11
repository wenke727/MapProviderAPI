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
from geo_utils import convert_geom_to_utm_crs, convert_geom_to_wgs
from serialization import read_csv_to_geodataframe
from trajectory_cleaner import clean_drift_traj_points

from utils.logger import logger_dataframe, make_logger

TRAJ_ID_COL = "tid"

#%%

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
        
        self.traj_id_col = traj_id_col
        if traj_id_col not in self.points.columns:
            self.points.loc[:, traj_id_col] = traj_id

    def clean_drift_points(self, method='twoside', speed_limit=None, dis_limit=None,
                           angle_limit=30, alpha=1, strict=False):
        """
        Clean drift in trajectory data by filtering out points based on speed, distance, 
        and angle thresholds.
        """
        pts = clean_drift_traj_points(self.points, col=[self.traj_id_col, 'dt', 'geometry'],
                                      method=method, speed_limit=speed_limit, dis_limit=dis_limit,
                                      angle_limit=angle_limit, alpha=alpha, strict=strict)
        self.points = pts
        
        return pts

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

fn = "../../../ST-MapMatching/data/cells/004.csv"
fn = "../../../ST-MapMatching/data/cells/014.csv"
# fn = "../../../ST-MapMatching/data/cells/420.csv"

pts = read_csv_to_geodataframe(fn)

traj = Trajectory(pts, traj_id=1, )
traj.points

_traj = traj.clean_drift_points(speed_limit=0, dis_limit=0, angle_limit=45, alpha=1, strict=False)
logger_dataframe(_traj)

ax = pts.plot(color='r')
_traj.plot(ax=ax, color='b')

_traj.index


# %%
