import numpy as np
from numba import njit
from loguru import logger
from geopandas import GeoDataFrame
from shapely import LineString, Point

from ..geo.geo_utils import convert_geom_to_utm_crs

TRAJ_ID_COL = 'tid'


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
    Calculate and return the lower and upper threshold values for outlier detection
    in a given series, using the Interquartile Range (IQR) method.

    Parameters:
    series (pd.Series): A Pandas Series for which to calculate the outlier thresholds.
    alpha (int, optional): The multiplier for IQR to set the thresholds. Default is 3.

    Returns:
    tuple: A tuple containing two elements:
           - The lower threshold for outlier detection.
           - The upper threshold for outlier detection.
    """
    q25, q75 = series.quantile((0.25, 0.75))
    iqr = q75 - q25
    q_high = q75 + alpha * iqr
    q_low = q25 - alpha * iqr
    
    return q_low, q_high

def clean_drift_traj_points(data: GeoDataFrame, col=[TRAJ_ID_COL, 'dt', 'geometry'],
                     method='twoside', speed_limit=None, dis_limit=None,
                     angle_limit=30, alpha=1, strict=False, verbose=False):
    """
    Clean drift in trajectory data by filtering out points based on speed, distance, 
    and angle thresholds.

    This function processes a GeoDataFrame containing trajectory data and removes 
    points that are considered drifts based on specified criteria. The filtering 
    conditions include speed and distance limits and angular constraints. It 
    supports both one-sided and two-sided filtering methods and offers a 'strict' 
    mode for more stringent filtering.

    Parameters:
    - data (GeoDataFrame): The trajectory data to be processed.
    - col (list, optional): The columns to use, default is [TRAJ_ID_COL, 'dt', 'geometry'].
    - method (str, optional): The filtering method, either 'oneside' or 'twoside', default is 'twoside'.
    - speed_limit (float, optional): The speed threshold for filtering, default is None.
    - dis_limit (float, optional): The distance threshold for filtering, default is None.
    - angle_limit (float, optional): The angle threshold for filtering, default is 30 degrees.
    - alpha (float, optional): The multiplier for calculating IQR based thresholds, default is 1.
    - strict (bool, optional): If set to True, applies stricter filtering conditions, default is False.

    Returns:
    GeoDataFrame: The cleaned trajectory data with drift points removed based on the specified criteria.

    Examples:
    >>> cleaned_data = traj_clean_drift(trajectory_data, speed_limit=50, dis_limit=100, strict=True)
    >>> cleaned_data = traj_clean_drift(trajectory_data, method='oneside', angle_limit=45, alpha=2)
    """
    [Rid, Time, Geometry] = col
    data.drop_duplicates(subset=[Rid, Time], inplace=True)
    
    def _preprocess(data):
        df = data[col].copy()
        df = df.sort_values(by=[Rid, Time])

        if df.crs.to_epsg() == 4326:
            convert_geom_to_utm_crs(df, inplace=True)
        
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

    def op(a, b):
        return a | b if not strict else a & b

    df = _preprocess(data)
    
    # traj mask
    traj_mask = (df[Rid + '_pre'] == df[Rid])
    if method == 'twoside':
        traj_mask = traj_mask & (df[Rid + '_next'] == df[Rid])

    # speed limit
    speed_mask = False
    if speed_limit is not None:
        if speed_limit <= 0:
            _, speed_limit = get_outliers_thred_by_iqr(df['speed_pre'], alpha)
            if verbose: 
                logger.debug(f"Speed limit: {speed_limit:.2f} m/s")
        
        if method == 'oneside':
            speed_mask = df['speed_pre'] > speed_limit
        elif method == 'twoside':
            speed_mask = op(df['speed_pre'] > speed_limit, df['speed_next'] > speed_limit)
            if strict:
                speed_mask &= (df['speed_prenext'] < speed_limit)

    # distance limit
    dis_mask = False
    if dis_limit is not None:
        if dis_limit <= 0:
            _, dis_limit = get_outliers_thred_by_iqr(df['dis_pre'], alpha)
            if verbose: 
                logger.debug(f"Distance limit: {dis_limit:.2f} m")
        if method == 'oneside':
            dis_mask = df['dis_pre'] > dis_limit
        elif method == 'twoside':
            dis_mask = op(df['dis_pre'] > dis_limit, df['dis_next'] > dis_limit)
            if strict:
                dis_mask &= (df['dis_prenext'] < dis_limit)

    # angle limit
    angle_mask = False
    if angle_limit is not None:
        df['angle'] = 180 - calculate_angle_between_sides(df['dis_pre'], df['dis_next'], df['dis_prenext'])
        angle_mask = (df['angle'] > angle_limit).fillna(False)
        if verbose: 
            logger.debug(f"\nAngels: {list(np.round(df['angle'].values, 0))}, \nmask: {list(angle_mask)}")
    mask = (traj_mask & (speed_mask | dis_mask | angle_mask))
    
    return data[~mask], df

@njit
def find_updates(xy, radius):
    update_indices = [0]
    last_update_x = xy[0, 0]
    last_update_y = xy[0, 1]
    
    for i in range(1, xy.shape[0]):
        dx = xy[i, 0] - last_update_x
        dy = xy[i, 1] - last_update_y
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance >= radius:
            update_indices.append(i)
            last_update_x = xy[i, 0]
            last_update_y = xy[i, 1]
            
    return np.array(update_indices)

def filter_by_point_update_policy(gdf:GeoDataFrame, radius:float, keep_last=True):
    """
    Optimized point update policy for GeoDataFrame with projected coordinates.
    
    This function identifies points that are beyond a specified radius from the 
    last updated point, considering them as new updates and thereby reducing the 
    number of updates sent to a server in a tracking system.
    
    Parameters:
    - gdf (GeoDataFrame): The input GeoDataFrame with a 'geometry' column. 
        It should be projected and indexed by time.
    - radius (float): The radius threshold in the same units as the GeoDataFrame's 
        projection (e.g., meters).
    
    Returns:
    - GeoDataFrame: A new GeoDataFrame with points that qualify as updates based on 
    the specified radius.
    
    Refs: 《Computing with Spatial Trajectories》
    """
    
    gdf = gdf.sort_index()
    # last_update_index = gdf.index[0]
    # update_indices = [last_update_index]
    
    # for idx, row in gdf.iterrows():
    #     if idx == last_update_index:
    #         continue

    #     distance = row.geometry.distance(gdf.at[last_update_index, 'geometry'])
    #     if distance < radius:
    #         continue
        
    #     # Record index if distance exceeds the threshold
    #     last_update_index = idx
    #     update_indices.append(idx)
    
    xy = np.array([(geom.x, geom.y) for geom in gdf.geometry])
    update_indices = find_updates(xy, radius)
    
    # the last point
    idx = len(gdf) - 1
    if keep_last and update_indices[-1] != idx:
        p_0 = gdf.iloc[update_indices[-1]].geometry
        p_1 = gdf.iloc[-1].geometry
        if not p_0.equals(p_1):
            update_indices = np.append(update_indices, idx)
        
    return gdf.loc[update_indices]

def simplify_traj_points(gdf, tolerance, precision=6):
    """
    Simplify a trajectory represented as a collection of points in a GeoDataFrame,
    while keeping track of the indices of the points that are retained.
    This function uses the Douglas-Peucker algorithm.

    Parameters:
    ----------
    gdf : GeoDataFrame
        Input GeoDataFrame with 'dt' for time, 'geometry' for points, 
        and other attributes. The DataFrame is expected to be sorted by 'dt'.
    tolerance : float
        Tolerance parameter for the Douglas-Peucker algorithm. 
        It determines the degree of simplification.
    precision : int, optional
        The number of decimal places to consider for comparing coordinates.
        Defaults to 6.

    Returns:
    -------
    GeoDataFrame
        A simplified GeoDataFrame containing only the points that are within
        'precision' decimal places of the points on the simplified trajectory.
    """
    if gdf.shape[0] <= 2:
        return gdf
    
    gdf.sort_values(by='dt', inplace=True)
    simplified_line = LineString(gdf['geometry'].values).simplify(tolerance)
    simplified_points = [tuple(np.round(p, precision)) for p in simplified_line.coords]

    def is_near_simplified_line(point):
        point_rounded = tuple(np.round(point.coords[0], precision))
        return point_rounded in simplified_points
    # FIXME 存在重叠点的问题
    mask = gdf['geometry'].apply(is_near_simplified_line)

    return gdf[mask]


if __name__ == "__main__":
    from ..geo.serialization import read_csv_to_geodataframe
    traj = read_csv_to_geodataframe('./data/cells/004.csv')
    _records = clean_drift_traj_points(traj)
    _records
    