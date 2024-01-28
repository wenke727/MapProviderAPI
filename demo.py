#%%
import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
from copy import copy
import geopandas as gpd
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
from shapely import LineString, MultiLineString

from tilemap import plot_geodata
from maptools.trajectory import Trajectory
from maptools.geo.serialization import read_csv_to_geodataframe

from mapmatching import ST_Matching
from mapmatching.graph import GeoDigraph
from mapmatching.geo.io import read_csv_to_geodataframe, to_geojson

from maptools.utils.misc import set_chinese_font_style
from maptools.utils.logger import make_logger, logger_dataframe
from maptools.utils.serialization import save_fig
from maptools.geo.linestring import merge_linestrings

set_chinese_font_style()
UTM_CRS = None

#%%

def _test_shortest_path(net, src, dst):
    """ 最短路径测试 """
    res = net.search(src, dst)
    net.df_edges.loc[res['epath']].plot()

    return res

def _judge_not_subway(probs:dict, eps=.6):
    for key, val in probs.items():
        if val < eps:
            return False
    return True

def _plot_temporal_prob_dist(actual_duration, avg_duration, avg_waiting):
    # 生成一系列的实际通行时间用于绘图
    min_duration = avg_duration - avg_waiting
    durations = np.linspace(min_duration*0.8, actual_duration * 1.2, 1000)
    probabilities = [cal_temporal_prob(duration, avg_duration, min_duration) for duration in durations]

    # 绘制概率图
    plt.figure(figsize=(10, 6))
    plt.plot(durations, probabilities, label="Probability")
    plt.axvline(x=actual_duration, color='r', linestyle='--', label='Actual Duration')
    plt.axvline(x=min_duration, color='g', linestyle='--', label='Min Duration')
    plt.axvline(x=avg_duration, color='b', linestyle='--', label='Avg Duration')
    plt.xlabel('Duration')
    plt.ylabel('Probability')
    plt.title('Probability of Actual Duration')
    plt.legend()
    plt.grid(True)
    plt.show()

def cal_temporal_prob(actual_duration, avg_duration, avg_waiting, factor=2, bias=60):
    """
    Calculate the probability of a given actual subway transit time based on a normal distribution.

    This function estimates the likelihood that a given actual duration of subway transit is typical,
    based on average expected transit time and waiting time. It assumes a normal distribution of transit times.

    Parameters:
    actual_duration : float
        The actual duration of the subway transit in minutes.
    avg_duration : float
        The average predicted duration of the subway transit in minutes.
    avg_waiting : float
        The average waiting time for the subway in minutes, used to adjust the minimum expected transit time and standard deviation.
    factor : float, optional
        A factor used to scale the standard deviation. Default is 5.
    bias : float, optional
        A bias term added to the average waiting time to calculate the standard deviation. Default is 120.

    Returns:
    float
        The probability of the given actual_duration being typical, calculated using a normal distribution formula.

    Notes:
    - The function returns 0 if the actual_duration is significantly less than 60% of the minimum expected transit time (avg_duration - avg_waiting).
    - The standard deviation used in the normal distribution is calculated as avg_waiting + bias, which can be adjusted for different scenarios.

    Example:
    >>> cal_temporal_prob(30, 25, 5)
    """
    min_duration = avg_duration - avg_waiting
    if min_duration * .6 > actual_duration:
        return 0
    
    sigma = factor * avg_waiting
    if actual_duration > avg_waiting:
        sigma += bias
    
    prob = np.exp(-((actual_duration - avg_duration) ** 2) / (sigma ** 2))
    logger.debug(f"Prob({actual_duration/60:.1f} | {avg_duration/60:.1f}, "
                 f"{avg_waiting/60:.0f}, {bias/60:.0f}) = {prob*100:.1f}%")
    
    return prob

def get_time_params(traj:Trajectory, df_path:gpd.GeoDataFrame, lineid_2_waitingtime:dict):
    """
    Calculate various time parameters and the temporal probability for a given subway trajectory.

    This function computes the actual duration of a trip, the average duration including waiting times,
    and the temporal probability of the trip based on the given trajectory data.

    Parameters:
    traj : object
        A trajectory object containing raw trajectory data with timestamps.
    df_path : DataFrame
        A DataFrame representing the path of the trajectory, including information on each leg of the journey.
    lineid_2_waitingtime : dict
        A dictionary mapping line IDs to their respective average waiting times.

    Returns:
    tuple
        A tuple containing actual_duration, avg_duration, avg_waiting, and temporal_prob.

    Notes:
    - The function calculates the actual duration of the trip as the difference between the maximum and minimum timestamps in the trajectory data.
    - It computes the waiting time as the sum of durations at exchange links, minus the walking durations.
    - The first waiting time is added to the average duration.
    - The temporal probability is computed using the `cal_temporal_prob` function.
    - A waiting penalty is calculated based on the number of exchange links.
    - The function assumes that durations less than 60% of the total travel time are improbable.

    Example:
    >>> get_time_params(traj, df_path, lineid_2_waitingtime)
    """
    
    # calculate_actual_duration
    actual_duration = traj.raw_df.dt.max() - traj.raw_df.dt.min()
    
    # calculate_waiting_time
    exchange_links = df_path.query("dst_name in ['exchange', 'inner_link']")
    waiting_time = exchange_links.duration.sum() - exchange_links.walking_duration.sum()
    first_watiting_time = lineid_2_waitingtime[df_path.iloc[0].way_id]

    _sum = df_path.duration.sum()
    avg_duration = _sum + first_watiting_time
    avg_waiting = waiting_time + first_watiting_time

    waiting_penalty = 120 * (1 + exchange_links.shape[0])
    logger.debug(f"waiting_penalty: {waiting_penalty}")
    temporal_prob = cal_temporal_prob(actual_duration, avg_duration, avg_waiting, bias=waiting_penalty)
    
    return actual_duration, avg_duration, avg_waiting, temporal_prob

def trim_first_and_last_step(df_path, res, eps=.5):
    """
    Trims the first and last steps of a path based on specified criteria.

    This function modifies the path by potentially removing the first and/or last step
    if they meet certain conditions defined by `eps`. It's used to refine the path data
    by trimming unnecessary steps at the beginning and the end.

    Parameters:
    df_path : DataFrame
        The DataFrame containing the edges of the path.
    res : dict
        A dictionary containing the results and parameters of the path, including 'epath', 'status', 
        'step_0', and 'step_n'.
    eps : float, optional
        The threshold value used to determine whether the first and last steps should be trimmed. 
        Default is 0.5.

    Returns:
    DataFrame
        A DataFrame of the updated path after trimming the first and/or last steps.

    Notes:
    - The function checks 'status' in `res`; if it's not 0, it returns the original `df_path` without changes.
    - The first and last steps are trimmed based on their comparison with `eps`.
    - If the first or last step is an 'exchange' or 'inner_link', it's further adjusted.
    - The function updates `res` with the new 'epath', 'step_0', and 'step_n' based on the trimming.
    - It also processes the trimmed path data using `process_path_data` before returning.
    """
    
    df_path = df_edges.loc[res['epath']]
    if res['status'] != 0:
        return df_path
    
    # determine_trim_indices
    path_length  = len(res['epath']) 
    if res['step_0'] < eps:
        start = 0  
    else:
        start = 1
        logger.debug(f"Change start to {start}, for {res['step_0']:.3f} < {eps}")
    if res['step_n'] < eps:
        end = path_length - 2  
        logger.debug(f"Change end to {end}, for {res['step_n']:.3f} < {eps}")
    else:
        end = path_length - 1

    if df_path.iloc[start].dst_name in ['exchange',  'inner_link']:
        start += 1
    if df_path.iloc[end].dst_name in ['exchange',  'inner_link']:
        end -= 1
    
    # update `res`
    res['epath'] = res['epath'][start: end + 1]
    if start != 0:
        res['step_0'] = 0
    if end != path_length - 1:
        res['step_n'] = 1
    
    # update `path`
    df_path = df_path.iloc[start: end + 1]
    df_combined_path = process_path_data(df_path)
    
    return df_combined_path

def process_path_data(df):
    """
    Processes and aggregates path data from a DataFrame.

    This function handles a DataFrame representing a path by assigning step IDs, separating special
    cases (like 'exchange' and 'inner_link'), aggregating data by 'way_id' and 'step', and combining
    the results into a clean GeoDataFrame.

    Parameters:
    df : DataFrame
        A DataFrame representing the path data to be processed.

    Returns:
    GeoDataFrame
        A processed GeoDataFrame with aggregated path data.

    Notes:
    - The function assigns step IDs based on 'way_id' and whether the destination name is a special case.
    - Special cases where 'dst_name' is 'exchange' or 'inner_link' are separated and handled differently.
    - Data is grouped by 'way_id' and 'step', with several aggregations performed on the grouped data.
    - The function also handles missing values in 'walking_duration'.
    - The result combines aggregated data with special cases and is returned as a GeoDataFrame.
    """
    
    if df.empty:
        return df
    
    df = df.copy()
    special_cases_mask = df['dst_name'].isin(['exchange', 'inner_link'])
    
    # step id
    df.loc[:, 'order'] = range(df.shape[0])
    df['step'] = (df['way_id'] != df['way_id'].shift()).cumsum() - 1
    
    # step_id = 0
    # _len = len(df)
    # arr_steps = np.zeros(_len)
    # prev_way_id = df.iloc[0].way_id
    # for i, (_, item) in enumerate(df.iloc[1:].iterrows(), start=1):
    #     if item['way_id'] == prev_way_id and not special_cases_mask.iloc[i]:
    #         arr_steps[i] = step_id
    #         continue
    #     step_id += 1
    #     arr_steps[i] = step_id
    #     prev_way_id = item['way_id']
    # df.loc[:, 'step'] = arr_steps


    # Separate records where dst_name is 'exchange' or 'inner_link'
    special_cases = df[special_cases_mask]
    regular_cases = df[~special_cases_mask]


    # Group by eid and aggregate
    grouped = regular_cases.groupby(['way_id', 'step']).agg({
        'src': 'first',
        'dst': 'last',
        'src_name': 'first',
        'dst_name': 'last',
        'eid': lambda x: list(x),
        'dist': 'sum',
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
               .drop(columns=['order', 'step'])\
               .reset_index(drop=True)

    return gpd.GeoDataFrame(result)

def load_mapmather():
    global UTM_CRS
    df_nodes = gpd.read_file('../MapTools/exp/shezhen_subway_nodes.geojson')
    df_edges = gpd.read_file('../MapTools/exp/shezhen_subway_edges.geojson')

    df_edges = df_edges.assign(
        dist = df_edges['distance'],
        geometry = df_edges.geometry.fillna(LineString())
    )
    
    net = GeoDigraph(df_edges, df_nodes.set_index('nid'), weight='duration')
    matcher = ST_Matching(net=net, ll=False, loc_deviaction=180, prob_thres=.0)
    # FIXME 
    UTM_CRS = matcher.utm_crs
    logger.warning(f"crs: {UTM_CRS}")
    
    return matcher, net, df_edges, df_nodes

def plot_helper(traj:Trajectory, matcher: ST_Matching, res:dict, title:str=None, x_label=None):
    fig, ax = matcher.plot_result(traj.points.to_crs(4326), res)

    # plot origin traj `points` and `segments`
    traj.raw_df.to_crs(4326).plot(ax=ax, color='b', alpha=.4, marker='x', zorder=1)
    segs = traj.to_line_gdf().to_crs(4326)
    segs.plot(ax=ax, color='b', alpha=.6, linestyle=':', zorder=2)

    # plot od
    _pts = traj.points.to_crs(4326)
    _pts.iloc[1:].plot(ax=ax, color='b', facecolor='white')
    _pts.iloc[[-1]].plot(ax=ax, color='b')

    # plot od name
    net = matcher.net
    src_idx = net.get_edge(res['epath'][0], 'src')
    dst_idx = net.get_edge(res['epath'][-1], 'dst')
    od = net.get_node([src_idx, dst_idx]).to_crs(4326)
    xmin, xmax, ymin, ymax = ax.axis()
    delta_y = (ymax - ymin) / 50
    for _, p in od.iterrows():
        x = p.geometry.x
        y = p.geometry.y + delta_y
        if xmin > x or x > xmax or ymin > y or y > ymax:
            continue
        ax.text(
            x, y, p['name'], 
            transform=ax.transData,
            bbox=dict(
                facecolor='white', 
                alpha=0.4, 
                edgecolor='none', 
                boxstyle="round,pad=0.5"),
            va='bottom', 
            ha='center',
        )

    if title:
        ax.set_title(title)
    if x_label:
        ax.set_xlabel(x_label)
        
    return fig, ax

def pipeline(pts, traj_id, plot_time_dist=False, dist_eps=200, plot=True, save_img=None, title=''):
    global lineid_2_waitingtime
    
    res = {}
    # step 1: preprocess
    traj = Trajectory(pts, traj_id=traj_id, utm_crs=UTM_CRS)
    res['traj'] = traj
    traj.preprocess(
        radius=600, 
        speed_limit=0, dis_limit=None, angle_limit=60, alpha=1, strict=False, 
        tolerance=200,
        verbose=False, 
        plot=False
    )

    # step 2:map-matching
    match_res = matcher.matching(
        traj.points.to_crs(4326), 
        search_radius=600, top_k=8,
        dir_trans=False, 
        details=False, 
        plot=False, 
        simplify=False,
        debug_in_levels=False
    )
    res['match_res'] = match_res
    logger.debug(match_res)

    if len(match_res.get('epath', [])) == 0:
        logger.warning(f"No candidates/轨迹点无法映射到候选边上, status: {match_res['status']}")
        return traj, match_res, pd.DataFrame()

    # step 3: postprocess `df_path`
    df_path = traj.align_crs(df_edges.loc[match_res['epath']])
    df_path = trim_first_and_last_step(df_path, match_res)
    res['df_path'] = df_path
        
    # step 4: metric, 计算轨迹分数：时间、空间 以及 Cell
    route = merge_linestrings(df_path.geometry)
    dists = traj.distance(route)
    cell_dis_prob = (dists < dist_eps).mean()
    match_res['probs'] = {**match_res['probs'], 'cell_dis_prob': cell_dis_prob} # **dist_dict

    # step 4.2: travel time probs
    actual_duration, avg_duration, avg_waiting,  temporal_prob = get_time_params(traj, df_path, lineid_2_waitingtime)
    if plot_time_dist:
        _plot_temporal_prob_dist(actual_duration, avg_duration, avg_waiting)

    match_res['probs']['temporal_prob'] = temporal_prob
    for p in ['trans_prob', 'prob']:
        if p in match_res['probs']:
            del match_res['probs'][p]

    res = {
        'traj': traj, 
        'matching': match_res, 
        'df_path': df_path.drop(columns=['dir', 'distance']),
    }    
    
    if plot:
        start = df_path.iloc[0].src_name
        end = df_path.iloc[-1].dst_name
        trip_info = f" \n{start} -> {end}\n{actual_duration/60:.1f} min"\
                    f" / [{(avg_duration - avg_waiting)/60:.1f}, {avg_duration/60:.1f}, "\
                    f"{(avg_duration + avg_waiting) / 60:.1f}]"
        res['fig'], res['ax'] = plot_helper(traj, matcher, match_res, f"{title}, {trip_info}")
        if save_img:
            res['fig'].savefig(save_img, bbox_inches='tight', pad_inches=0.1, dpi=500)
            plt.close()
    
    return res

def exp(folder, desc=None):
    global logger
    logger = make_logger(folder, 'cell', console=True, include_timestamp=False)

    folder = Path(folder)
    img_folder = folder  / desc if desc else folder
    img_folder_0 = img_folder / 'Subway'
    img_folder_1 = img_folder / 'NoSubway'
    for f in [folder, img_folder, img_folder_0, img_folder_1]:
        f.mkdir(parents=True, exist_ok=True)

    matching_res = []
    raw_points_res = []
    points_res = []
    path_res= []

    for fn in sorted(glob.glob(f"{folder}/csv/*.csv")):
        fn_name = Path(fn).name.split('.')[0]
        traj_id = int(fn_name)

        # read
        logger.info(f"processing: {fn}")
        pts = read_csv_to_geodataframe(fn)

        try:
            result = pipeline(pts, traj_id=traj_id, plot_time_dist=False, title=Path(fn).name)
            traj = result['traj']
            
            # collect geojson
            traj.raw_df.loc[:, 'traj_id'] = int(fn_name)
            raw_points_res.append(traj.raw_df)
            if not traj.points.empty:
                points_res.append(traj.points)
            if not result['df_path'].empty:
                result['df_path'].loc[:, 'traj_id'] = int(fn_name)
                path_res.append(result['df_path'])

            # save img for debug
            lable = _judge_not_subway(result['matching']['probs'])
            if lable:
                save_fig(result['fig'], img_folder_0 / f'{fn_name}.jpg')
            else:
                save_fig(result['fig'], img_folder_1 / f'{fn_name}.jpg')
                result['ax'].title.set_backgroundcolor('orange')
            save_fig(result['fig'], img_folder / f'{fn_name}.jpg')
            plt.close()
            
            # collect probs
            result['matching']['probs'] = {"fn": fn, **result['matching']['probs']}
            matching_res.append(result['matching']['probs'])
            
        except:
            logger.error(fn)

    # save result
    if desc:
        folder = folder / desc
    probs = pd.DataFrame(matching_res)
    probs.rename(columns={'prob': 'status'}, inplace=True)
    probs.loc[:, 'status'] = ''
    probs.to_excel(folder / 'probs.xlsx', index=False)

    to_geojson(gpd.GeoDataFrame(pd.concat(raw_points_res), geometry='geometry'), folder / 'raw_points')
    to_geojson(gpd.GeoDataFrame(pd.concat(points_res), geometry='geometry'), folder / 'points')

    df_routes = gpd.GeoDataFrame(pd.concat(path_res), geometry='geometry')
    df_routes.loc[:, 'eid'] = df_routes['eid'].astype(str)
    to_geojson(df_routes, folder / 'routes')
    
    return


if __name__ == '__main__':
    matcher, net, df_edges, df_nodes = load_mapmather()
    lineid_2_waitingtime = df_edges[['way_id', 'duration', 'dst_name']]\
                                   .query(" dst_name == 'inner_link' ")\
                                   .drop_duplicates()\
                                   .set_index('way_id').to_dict()['duration']

    exp('./exp/12-08/0800', 'attempt_1')


#%%
# folder = Path('./exp/12-06/0800/csv')
# folder = Path('./exp/12-08/0800/csv')

# fns = sorted(glob.glob(f"{folder}/*.csv"))

# #%%
# traj_id = 1
# fn = fns[traj_id]
# pts = read_csv_to_geodataframe(fn)
# res = pipeline(pts, traj_id=traj_id, plot_time_dist=False, title=Path(fn).name, save_img=False)
# traj = res['traj']
# df_path = res['df_path']


# %%
