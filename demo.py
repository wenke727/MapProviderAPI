#%%
import numpy as np
import pandas as pd
import seaborn as sns
from copy import copy
import geopandas as gpd
from shapely.ops import linemerge
from shapely import LineString, MultiLineString

from maptools.trajectory import Trajectory
from maptools.geo.serialization import read_csv_to_geodataframe

from mapmatching.graph import GeoDigraph
from mapmatching import ST_Matching
from mapmatching.utils.logger_helper import logger_dataframe, make_logger
from mapmatching.geo.io import read_csv_to_geodataframe, to_geojson


#%%

def test_shortest_path(net, src, dst):
    """ 最短路径测试 """
    res = net.search(src, dst)
    net.df_edges.loc[res['epath']].plot()

    return res

def merge_linestrings(linestrings, to_multilinestring=False):
    """
    Merges a list of LineString objects into a MultiLineString or a single LineString
    using Shapely's linemerge function.
    
    Args:
    - linestrings (list): A list of LineString objects.
    - to_multilinestring (bool): If True, force output to be a MultiLineString.

    Returns:
    - LineString/MultiLineString: The merged LineString or MultiLineString object.
    """

    valid_linestrings = [ls for ls in linestrings if not ls.is_empty]

    if not valid_linestrings:
        return LineString()

    merged = linemerge(valid_linestrings)
    if to_multilinestring and not isinstance(merged, MultiLineString):
        return MultiLineString([merged])
    
    return merged

def process_path_data(df):
    df = df.copy()
    special_cases_mask = df['dst_name'].isin(['exchange', 'inner_link'])
    
    # step id
    step_id = 0
    _len = len(df)
    arr_steps = np.zeros(_len)
    df.loc[:, 'order'] = range(_len)
    prev_way_id = df.iloc[0].way_id
    for i, (_, item) in enumerate(df.iloc[1:].iterrows(), start=1):
        if item['way_id'] == prev_way_id and not special_cases_mask.iloc[i]:
            arr_steps[i] = step_id
            continue
        step_id += 1
        arr_steps[i] = step_id
        prev_way_id = item['way_id']
    df.loc[:, 'step'] = arr_steps

    # Separate records where dst_name is 'exchange' or 'inner_link'
    special_cases = df[special_cases_mask]
    df = df[~special_cases_mask]
    
    # Group by eid and aggregate
    grouped = df.groupby(['way_id', 'step']).agg({
        'src': 'first',
        'dst': 'last',
        'src_name': 'first',
        'dst_name': 'last',
        'eid': lambda x: list(x),
        'dist': 'sum',
        # 'distance': 'sum',
        'duration': 'sum',
        'walking_duration': 'sum',
        'speed': 'mean',
        'geometry': merge_linestrings,
        'order': 'first',
    }).reset_index()

    # Handle missing values in walking_duration
    grouped['walking_duration'] = grouped['walking_duration'].replace({0: np.nan})

    # Combine the grouped data with the special cases
    result = pd.concat([grouped, special_cases], ignore_index=True)\
               .sort_values(['order', 'step'])\
               .drop(columns=['step', 'order'])\
               .reset_index(drop=True)

    return gpd.GeoDataFrame(result)

def load_mapmather():
    df_nodes = gpd.read_file('../MapTools/exp/shezhen_subway_nodes.geojson')
    df_edges = gpd.read_file('../MapTools/exp/shezhen_subway_edges.geojson')

    df_edges = df_edges.assign(
        dist = df_edges['distance'],
        geometry = df_edges.geometry.fillna(LineString())
    )

    net = GeoDigraph(df_edges, df_nodes.set_index('nid'), weight='duration')
    matcher = ST_Matching(net=net, ll=False, loc_deviaction=200)
    
    return matcher, net, df_edges, df_nodes

def plot_helper(traj:Trajectory, matcher: ST_Matching):
    fig, ax = matcher.plot_result(traj.points.to_crs(4326), res)
    traj.raw_df.to_crs(4326).plot(ax=ax, color='b', alpha=.5, marker='x', zorder=1)


    segs = self.to_line_gdf().to_crs(4326)
    segs.plot(ax=ax, color='b', alpha=.6, linestyle=':', zorder=2)

    _pts = traj.points.to_crs(4326)
    _pts.iloc[1:-1].plot(ax=ax, color='b', facecolor='white', zorder=5)
    _pts.iloc[[-1]].plot(ax=ax, color='b', zorder=6)

    ax.set_title(f"traj: {rid}")


matcher, net, df_edges, df_nodes = load_mapmather()

lineid_2_waitingtime = df_edges.query(" dst_name == 'inner_link' ")[['way_id', 'duration']]\
                               .drop_duplicates().set_index('way_id').to_dict()['duration']


#%%

# 5, 12, 23
# 轨迹中断： 32， 43， 48

rid = 5
fn = f"./exp/231206/0800/{rid:03d}.csv"
# fn = f"../ST-MapMatching/data/cells/{idx:03d}.csv"

# read
pts = read_csv_to_geodataframe(fn)

# preprocess
self = traj = Trajectory(pts, traj_id=1)
traj.preprocess(
    radius=500, 
    speed_limit=0, dis_limit=None, angle_limit=60, alpha=2, strict=False, 
    tolerance=200,
    verbose=False, 
    plot=False
)

# map-matching
res = matcher.matching(
    traj.points.to_crs(4326), 
    search_radius=500, top_k=8,
    dir_trans=False, details=True, plot=False, 
    simplify=False, tolerance=500, debug_in_levels=False
)

# visualize
plot_helper(traj, matcher)

# metric, 计算轨迹分数：时间、空间 以及 Cell
df_path = df_edges.loc[res['epath']]
route = merge_linestrings(df_path.geometry)

dists = traj.distance(route)
# sns.boxplot(traj.distance(route))
dist_dict = dists.describe().to_dict()


probs = copy(res['probs'])
pd.DataFrame([{**probs, **dist_dict}])

#%%
# 裁剪首尾段
eps = 0.1
start = 0 if res['step_0'] < eps else 1
end = -1 if res['step_n'] < eps else len(res['epath'])
df_path = df_path.iloc[start: end]

df_combined_path = process_path_data(df_path)
df_combined_path

# %%
#! travel time probs
df_path.duration.sum()


# %%
actual_duration = traj.raw_df.dt.max() - traj.raw_df.dt.min()

excahnge_links = df_combined_path.query("dst_name in ['exchange', 'inner_link']")
waiting_time = excahnge_links.duration.sum() - excahnge_links.walking_duration.sum()
first_watiting_time = lineid_2_waitingtime[df_combined_path.iloc[0].way_id]

_sum = df_combined_path.duration.sum()
min_duration = _sum - waiting_time
avg_duration = _sum + first_watiting_time


actual_duration, min_duration, avg_duration, waiting_time + first_watiting_time
# (1875.9, 1517.0, 1757.0, 240.0)


# %%
