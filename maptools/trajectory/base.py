import pandas as pd
import geopandas as gpd
from shapely import Point, LineString
from shapely.affinity import translate


ACCELERATION_COL_NAME = "acceleration"
ANGULAR_DIFFERENCE_COL_NAME = "angular_difference"
DIRECTION_COL_NAME = "direction"
DISTANCE_COL_NAME = "distance"
SPEED_COL_NAME = "speed"
TIMEDELTA_COL_NAME = "timedelta"
TRAJ_ID_COL_NAME = "traj_id"


class BaseTrajectory:
    def __init__(
        self, df, traj_id, traj_id_col=None, obj_id=None,
        t=None, x=None, y=None, crs="epsg:4326", parent=None,):
        self.points = None
        
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError

    def size(self):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

    def drop(self, **kwargs):
        raise NotImplementedError

    def plot(self, *args, **kwargs):
        raise NotImplementedError

    def hvplot(self, *args, **kwargs):
        raise NotImplementedError

    def is_valid(self):
        raise NotImplementedError

    def get_crs(self):
        raise NotImplementedError

    def to_crs(self, crs):
        raise NotImplementedError

    def is_latlon(self):
        raise NotImplementedError

    def get_min(self, column):
        raise NotImplementedError

    def get_max(self, column):
        raise NotImplementedError

    def get_column_names(self):
        raise NotImplementedError

    def get_traj_id_col(self):
        raise NotImplementedError

    def get_speed_col(self):
        raise NotImplementedError

    def get_distance_col(self):
        raise NotImplementedError

    def get_direction_col(self):
        raise NotImplementedError

    def get_angular_difference_col(self):
        raise NotImplementedError

    def get_timedelta_col(self):
        raise NotImplementedError

    def get_geom_col(self):
        return self.points.geometry.name

    def to_linestring(self):
        raise NotImplementedError

    def to_linestringm_wkt(self):
        raise NotImplementedError

    def to_point_gdf(self, return_orig_tz=False):
        raise NotImplementedError

    def to_traj_gdf(self, wkt=False, agg=False):
        raise NotImplementedError

    def get_start_location(self):
        raise NotImplementedError

    def get_end_location(self):
        raise NotImplementedError

    def get_bbox(self):
        raise NotImplementedError

    def get_start_time(self):
        raise NotImplementedError

    def get_end_time(self):
        raise NotImplementedError

    def get_duration(self):
        raise NotImplementedError

    def get_row_at(self, t, method="nearest"):
        raise NotImplementedError

    def interpolate_position_at(self, t):
        raise NotImplementedError

    def get_position_at(self, t, method="interpolated"):
        raise NotImplementedError

    def get_linestring_between(self, t1, t2, method="interpolated"):
        raise NotImplementedError

    def get_segment_between(self, t1, t2):
        raise NotImplementedError

    def _compute_distance(self, row, conversion):
        raise NotImplementedError

    def _add_prev_pt(self, force=True):
        raise NotImplementedError

    def get_length(self):
        raise NotImplementedError

    def get_direction(self):
        raise NotImplementedError

    def get_sampling_interval(self):
        raise NotImplementedError

    def _compute_heading(self, row):
        raise NotImplementedError

    def _compute_angular_difference(self, row):
        raise NotImplementedError

    def _compute_speed(self, row, conversion):
        raise NotImplementedError

    def add_traj_id(self, overwrite=False):
        raise NotImplementedError

    def add_direction(self, overwrite=False):
        raise NotImplementedError

    def add_angular_difference(
        self,
    ):
        raise NotImplementedError

    def add_distance(self, overwrite=False, name=DISTANCE_COL_NAME, units=None):
        raise NotImplementedError

    def add_speed(self, overwrite=False, name=SPEED_COL_NAME, units=None):
        raise NotImplementedError

    def add_acceleration(
        self, overwrite=False, name=ACCELERATION_COL_NAME, units=None
    ):
        raise NotImplementedError

    def add_timedelta(self, overwrite=False, name=TIMEDELTA_COL_NAME):
        raise NotImplementedError

    def _get_df_with_timedelta(self, name=TIMEDELTA_COL_NAME):
        raise NotImplementedError

    def _get_df_with_distance(self, conversion, name=DISTANCE_COL_NAME):
        raise NotImplementedError

    def _get_df_with_speed(self, conversion, name=SPEED_COL_NAME):
        raise NotImplementedError

    def _get_df_with_acceleration(self, conversion, name=ACCELERATION_COL_NAME):
        raise NotImplementedError

    def intersects(self, polygon):
        raise NotImplementedError

    def distance(self, other, units=None):
        raise NotImplementedError

    def hausdorff_distance(self, other, units=None):
        raise NotImplementedError

    def clip(self, polygon, point_based=False):
        raise NotImplementedError

    def intersection(self, feature, point_based=False):
        raise NotImplementedError

    def apply_offset_seconds(self, column, offset):
        raise NotImplementedError

    def apply_offset_minutes(self, column, offset):
        raise NotImplementedError

    def _connect_prev_pt_and_geometry(self, row, eps=1e-8):
        pt0 = row["prev_pt"]
        pt1 = row[self.get_geom_col()]
        if not isinstance(pt0, Point):
            return None
        if not isinstance(pt1, Point):
            raise ValueError(f"Invalid trajectory! Got {pt1} instead of point!")
        if pt0 == pt1:
            # to avoid intersection issues with zero length lines
            pt1 = translate(pt1, eps, eps)
        
        return LineString(list(pt0.coords) + list(pt1.coords))
    
    def _to_line_df(self, points, columns=None):
        """
        Convert trajectory data GeoDataFrame of points to GeoDataFrame of lines
        that connect consecutive points.

        Returns
        -------
        line_df : GeoDataFrame
            GeoDataFrame of line segments
        """
        if columns is None:
            line_df = points.copy()
        else:
            line_df = points[columns].copy()
        line_df["prev_pt"] = line_df.geometry.shift()
        line_df["t"] = points.index
        line_df["prev_t"] = line_df["t"].shift()
        line_df["line"] = line_df.apply(self._connect_prev_pt_and_geometry, axis=1)
        line_df = line_df.set_geometry("line")[1:]
        return line_df

    def to_line_gdf(self, points=None, columns=None):
        """
        Return the trajectory's line segments as GeoDataFrame.

        Returns
        -------
        GeoDataFrame
        """
        if points is None:
            points = self.points
            
        line_gdf = self._to_line_df(points, columns)
        line_gdf.drop(columns=[self.get_geom_col(), "prev_pt"], inplace=True)
        line_gdf.reset_index(drop=True, inplace=True)
        line_gdf.rename(columns={"line": "geometry"}, inplace=True)
        line_gdf.set_geometry("geometry", inplace=True)
        if self.crs:
            line_gdf.set_crs(self.crs, inplace=True)
            
        return line_gdf

    def get_mcp(self):
        raise NotImplementedError
