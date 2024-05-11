#%%
import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
from shapely import LineString
from geopandas import GeoDataFrame

from mapmatching import ST_Matching
from mapmatching.graph import GeoDigraph

from maptools.trajectory import Trajectory
from maptools.geo.linestring import merge_linestrings

# local debug
from maptools.utils.misc import set_chinese_font_style
from maptools.utils.logger import make_logger, logger_dataframe
from maptools.geo.serialization import read_csv_to_geodataframe, csv_to_geodf, to_geojson, load_folder_to_gdf
from maptools.utils.serialization import save_fig


set_chinese_font_style()


UTM_CRS = None
UPDATE_RADIUS = 800
SIMPLIFY_TOLERANCE = 250
ANGLE_LIMIT = 60
SPEED_LIMIT = 0
DISTANCE_LIMIT = None
DRIFT_ALPHA = 3

SEARCH_RADIUS = 500
TOK_K_CANDIDATES = 8
TRIM_EDGE_RATIO = 0.15
CELL_SERVICE_RADIUS = 200


def load_map_matcher():
    global UTM_CRS
    df_nodes = gpd.read_file('../MapTools/exp/shezhen_subway_nodes.geojson')
    df_edges = gpd.read_file('../MapTools/exp/shezhen_subway_edges.geojson')

    lineid_2_waitingtime = df_edges[['way_id', 'duration', 'dst_name']]\
                                .query(" dst_name == 'inner_link' ")\
                                .drop_duplicates()\
                                .set_index('way_id').to_dict()['duration']

    df_edges = df_edges.assign(
        dist = df_edges['distance'],
        geometry = df_edges.geometry.fillna(LineString())
    )
    
    # FIXME
    net = GeoDigraph(df_edges, df_nodes.set_index('nid'), weight='duration')
    matcher = ST_Matching(net=net, ll=False, prob_thres=.0)
    matcher.set_search_candidates_variables(top_k = TOK_K_CANDIDATES, search_radius = SEARCH_RADIUS)

    UTM_CRS = matcher.get_utm_crs()
    logger.warning(f"crs: {UTM_CRS}")
        
    return matcher, lineid_2_waitingtime

def pred_subway_trip(probs:dict, eps=.6):
    """通过卡阈值的方式判断是否地铁出行"""
    if not probs:
        return False
    
    for key, val in probs.items():
        if val < eps:
            return False
    return True

def cal_cell_dis_prob(traj:GeoDataFrame, df_path:GeoDataFrame, dist_eps:int=CELL_SERVICE_RADIUS):
    route = merge_linestrings(df_path.geometry)
    dists = traj.distance(route)
    cell_dis_prob = (dists < dist_eps).mean()
    
    return cell_dis_prob

def cal_temporal_prob(actual_duration, avg_duration, avg_waiting, factor=2, bias=60, verbose=False):
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
    if verbose:
        logger.debug(f"Prob({actual_duration/60:.1f} | {avg_duration/60:.1f}, "
                     f"{avg_waiting/60:.0f}, {bias/60:.0f}) = {prob*100:.1f}%")
    
    return prob

def get_time_values(traj:Trajectory, df_path:gpd.GeoDataFrame, lineid_2_waitingtime:dict):
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
    if df_path.empty:
        return actual_duration, np.float('inf'), np.float('inf'), 0
    
    actual_duration = traj.get_duration()
    
    # calculate_waiting_time
    exchange_links = df_path.query("dst_name in ['exchange', 'inner_link']")
    waiting_time = exchange_links.duration.sum() - exchange_links.walking_duration.sum()
    first_watiting_time = lineid_2_waitingtime.get(df_path.iloc[0].way_id, 180)

    _sum = df_path.duration.sum()
    avg_duration = _sum + first_watiting_time
    avg_waiting = waiting_time + first_watiting_time

    waiting_penalty = 120 * (1 + exchange_links.shape[0])
    temporal_prob = cal_temporal_prob(actual_duration, avg_duration, avg_waiting, bias=waiting_penalty)
    
    return actual_duration, avg_duration, avg_waiting, temporal_prob

def trim_first_and_last_step(df_path, res, eps=TRIM_EDGE_RATIO, skip_exchange_link=False, verbose=False):
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
    - It also processes the trimmed path data using `combine_subway_edges` before returning.
    """
    if res['status'] != 0:
        return df_path

    if verbose: logger.debug(f"step_0: {res['step_0']:.3f}, step_n: {res['step_n']:.3f}")
    start, end = 0, len(res['epath'])  - 1
    # strat
    if res['step_0'] < eps:
        if verbose: logger.debug(f"Change `step_0` to 0, for {res['step_0']:.3f} < {eps:.3f}")
        res['step_0'] = 0
    elif res['step_0'] > 1 - eps:
        if verbose: logger.debug(f"Change `step_0` to 0, start += 1, for {res['step_0']:.3f} < {1 - eps:.3f}")
        res['step_0'] = 0
        start += 1
    
    # end
    if res['step_n'] < eps:
        if verbose: logger.debug(f"Change `step_n` to 1, end -= 1, for {res['step_n']:.3f} < {eps:.3f}")
        res['step_n'] = 1
        end -= 1
    elif res['step_n'] > 1 - eps:
        if verbose: logger.debug(f"Change `step_n` to 1, for {res['step_n']:.3f} > {1 - eps:.3f}")
        res['step_n'] = 1

    # exchange link
    if skip_exchange_link:
        if df_path.iloc[start].dst_name in ['exchange',  'inner_link']:
            start += 1
        if df_path.iloc[end].dst_name in ['exchange',  'inner_link']:
            end -= 1
        
    # update `res`
    res['epath'] = res['epath'][start: end + 1]
    df_path = df_path.iloc[start: end + 1]
    if df_path.shape[0] > 1:
        df_path.iloc[0].duration *= res['step_0']
        df_path.iloc[-1].duration *= res['step_n']
    
    return df_path

def combine_subway_edges(df):
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
    # step id
    df.loc[:, 'order'] = range(df.shape[0])
    df['step'] = (df['way_id'] != df['way_id'].shift()).cumsum() - 1
    
    # Separate records where dst_name is 'exchange' or 'inner_link'
    special_cases_mask = df['dst_name'].isin(['exchange', 'inner_link'])
    special_cases = df[special_cases_mask]
    regular_cases = df[~special_cases_mask]

    # Group by eid and aggregate
    agg_dict = {
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
    }
    grouped = regular_cases.groupby(['way_id', 'step']).agg(agg_dict).reset_index()

    # Handle missing values in walking_duration
    grouped['walking_duration'] = grouped['walking_duration'].replace({0: np.nan})

    # Combine the grouped data with the special cases
    result = pd.concat([grouped, special_cases], ignore_index=True)\
               .sort_values(['order', 'step'])\
               .drop(columns=['order', 'step'])\
               .reset_index(drop=True)

    return gpd.GeoDataFrame(result)

def plot_matching_result(traj:Trajectory, matcher: ST_Matching, res:dict, title:str=None, x_label=None, legend=False):
    fig, ax = matcher.plot_result(traj.points.to_crs(4326), res, legend=legend)

    # plot origin traj `points` and `segments`
    traj.raw_df.to_crs(4326).plot(ax=ax, color='b', alpha=.4, marker='x', zorder=1)
    segs = traj.to_line_gdf()
    if not segs.empty:
        segs.to_crs(4326).plot(ax=ax, color='b', alpha=.6, linestyle=':', zorder=2)

    # plot od
    _pts = traj.points.to_crs(4326)
    if _pts.shape[0] > 1:
        _pts.iloc[1:].plot(ax=ax, color='b', facecolor='white')
        _pts.iloc[[-1]].plot(ax=ax, color='b')

    # plot od name
    if 'epath' in res:
        src_idx = matcher.get_edges(res['epath'][0], 'src')
        dst_idx = matcher.get_edges(res['epath'][-1], 'dst')
        od = matcher.get_nodes([src_idx, dst_idx]).to_crs(4326)
        xmin, xmax, ymin, ymax = ax.axis()
        delta_y = (ymax - ymin) / 50
        for _, p in od.iterrows():
            x = p.geometry.x
            y = p.geometry.y + delta_y
            if xmin > x or x > xmax or ymin > y or y > ymax:
                continue
            ax.text(
                x, y, p['name'], va='bottom', ha='center', transform=ax.transData,
                bbox=dict(facecolor='white', alpha=0.4, edgecolor='none', boxstyle="round,pad=0.5"),
            )

    if title:
        ax.set_title(title)
    
    if x_label:
        ax.set_xlabel(x_label)
        
    return fig, ax

def pipeline(pts, traj_id, dist_eps=CELL_SERVICE_RADIUS, plot=False, save_img=None, title='', verbose=False):
    global lineid_2_waitingtime, matcher

    # step 1: preprocess
    traj = Trajectory(pts, traj_id=traj_id, traj_id_col='traj_id', utm_crs=UTM_CRS, t='dt')
    res = {'traj': traj, 'match_res': {}}
    traj.preprocess(
        radius = UPDATE_RADIUS, 
        speed_limit = SPEED_LIMIT, 
        dis_limit = DISTANCE_LIMIT, 
        angle_limit = ANGLE_LIMIT, 
        alpha = DRIFT_ALPHA, 
        tolerance = SIMPLIFY_TOLERANCE,
        strict = False, 
        verbose = False, 
        plot = False
    )

    # step 2: map-matching
    match_res = matcher.matching(traj.points, simplify = False)
    res['match_res'] = match_res 
    if verbose: logger.debug(match_res)

    if len(match_res.get('epath', [])) == 0:
        if verbose: logger.warning(f"No candidates/轨迹点无法映射到候选边上, status: {match_res['status']}")
        if plot:
            res['fig'], res['ax'] = plot_matching_result(traj, matcher, match_res, f"{title}")  
        res['df_path'] = pd.DataFrame()

        return res

    # step 3: postprocess `df_path`
    df_path = matcher.get_edges(match_res['epath'])
    df_path = traj.align_crs(df_path)
    df_path = trim_first_and_last_step(df_path, match_res)
    df_path = combine_subway_edges(df_path)
    res['df_path'] = df_path.drop(columns=['dir', 'distance'])    
        
    # step 4: metric, 计算轨迹分数：时间、空间 以及 Cell
    cell_dis_prob = cal_cell_dis_prob(traj.raw_df, df_path, dist_eps)

    # step 4.2: travel time probs
    actual_duration, avg_duration, avg_waiting,  temporal_prob = get_time_values(traj, df_path, lineid_2_waitingtime)
    res['temporal'] = {'actual_duration': actual_duration, 'avg_duration': avg_duration, 'avg_waiting': avg_waiting}
    match_res['probs'] = {**match_res['probs'], 'cell_dis_prob': cell_dis_prob, 'temporal_prob': temporal_prob} # **dist_dict
    for p in ['trans_prob', 'prob']:
        if p in match_res['probs']:
            del match_res['probs'][p]
    
    if plot and not df_path.empty:
        start = df_path.iloc[0].src_name
        end = df_path.iloc[-1].dst_name
        try:
            ratio = f" / {match_res.get('step_0', 0):.2f}, {match_res.get('step_n', 0):.2f}"
        except:
            ratio = ''
        trip_info = f" \n{start} -> {end}{ratio}\n"\
                    f"{actual_duration/60:.1f} min"\
                    f" / [{(avg_duration - avg_waiting)/60:.1f}, {avg_duration/60:.1f}, "\
                    f"{(avg_duration + avg_waiting) / 60:.1f}]"
        res['fig'], res['ax'] = plot_matching_result(traj, matcher, match_res, f"{title}, {trip_info}")
        if save_img:
            res['fig'].savefig(save_img, bbox_inches='tight', pad_inches=0.1, dpi=500)
            plt.close()
    
    return res

def exp(trajs, out_folder=None, save_imgs=True, debug=False):
    global logger
    logger = make_logger(out_folder, 'log', console=True, include_timestamp=False)
    out_folder = Path(out_folder)
    
    def _create_folder(folder):
        img_folder = folder  / 'Imgs'
        img_folder_0 = folder / 'Subway'
        img_folder_1 = folder / 'NoSubway'
        for f in [folder, img_folder, img_folder_0, img_folder_1]:
            f.mkdir(parents=True, exist_ok=True)
            
        return img_folder, img_folder_0, img_folder_1

    def _collect_result(traj, result, raw_points_lst, points_lst, path_lst, matching_lst):
        # collect geojson
        raw_points_lst.append(traj.raw_df)
        if not traj.points.empty:
            points_lst.append(traj.points)
        if not result['df_path'].empty:
            result['df_path'].loc[:, 'traj_id'] = traj_id
            path_lst.append(result['df_path'])
        
        # collect probs
        pred = pred_subway_trip(result['match_res'].get('probs', {'prob': 0}))
        result['pred'] = pred
        info = {"traj_id": traj_id, "pred": pred, **result['match_res']['probs']}
        if 'step_0' in result['match_res']:
            info['step_0'] = result['match_res']['step_0']
        if 'step_n' in result['match_res']:
            info['step_n'] = result['match_res']['step_n']
        if 'temporal' in result:
            info.update(result['temporal'])
        logger.debug(info)
        matching_lst.append(info)

    def _save_fig(result, sep=True):
        # save img for debug
        if sep:
            if result['pred']:
                save_fig(result['fig'], sub_img_folder / f'{fn_name}.jpg')
            else:
                result['ax'].title.set_backgroundcolor('orange')
                save_fig(result['fig'], no_sub_img_folder / f'{fn_name}.jpg')
        else:
            save_fig(result['fig'], img_folder / f'{fn_name}.jpg')
        plt.close()
        pass

    def _save_result(out_folder, raw_points_lst, points_lst, path_lst, matching_lst):
        # save result
        probs = pd.DataFrame(matching_lst)
        probs.rename(columns={'prob': 'status'}, inplace=True)
        probs.loc[:, 'status'] = ''
        probs.to_csv(out_folder / 'matching_res.csv', index=False)

        to_geojson(gpd.GeoDataFrame(pd.concat(raw_points_lst), geometry='geometry'), out_folder / 'raw_points')
        to_geojson(gpd.GeoDataFrame(pd.concat(points_lst), geometry='geometry'), out_folder / 'points')

        df_routes = gpd.GeoDataFrame(pd.concat(path_lst), geometry='geometry')
        df_routes.loc[:, 'eid'] = df_routes['eid'].astype(str)
        to_geojson(df_routes, out_folder / 'routes')
         
    img_folder, sub_img_folder, no_sub_img_folder = _create_folder(out_folder)

    path_lst= []
    points_lst = []
    matching_lst = []
    raw_points_lst = []
    
    for traj_id, pts in trajs.groupby('traj_id'):
        fn_name = traj_id
        pts.reset_index(drop=True, inplace=True)

        # try:
        result = pipeline(pts, traj_id=traj_id, plot=save_imgs, title=f"traj_id: {traj_id}")
        if 'match_res' not in result:
            result['match_res'] = {'probs': {'norm_prob': 0}}
        # except:
        #     logger.error(f"process {fn} failed")

        traj = result['traj']
        traj.raw_df.loc[:, 'traj_id'] = traj_id

        _collect_result(traj, result, raw_points_lst, points_lst, path_lst, matching_lst)
        if save_imgs: 
            _save_fig(result)

    _save_result(out_folder, raw_points_lst, points_lst, path_lst, matching_lst)
    
    return


#%%
if __name__ == '__main__':
    """ load map macther """
    matcher, lineid_2_waitingtime = load_map_matcher()

    #%%
    trajs = csv_to_geodf('./data/trajs/sample.csv')
    # trajs = csv_to_geodf('./data/trajs/1249.csv')
    exp(trajs, out_folder='./debug/0423', save_imgs=True)
    
    #%%
    """ 单条轨迹测试 """
    traj_id = 0
    pts = trajs.query("traj_id == @traj_id")
    pts.reset_index(drop=True, inplace=True)
    traj_res = pipeline(pts, traj_id=traj_id, title=traj_id, save_img=None, plot=True)

    #%%
    """ 单条轨迹测试 """
    fn = Path('./exp/demo/025.csv') # 公交 / 地铁

    traj_id = int(fn.name.split('.')[0])
    pts = read_csv_to_geodataframe(fn)
    traj_res = pipeline(pts, traj_id=traj_id, title=Path(fn).name, save_img=False, plot=True)
    traj_res['match_res']


    #%%
    """ 针对某一个文件夹统一处理 """
    folder = './exp/12-08/0800/csv'
    trajs = load_folder_to_gdf(folder)
    exp(trajs, './debug/attempt_0422', save_imgs=True, debug=False)
    

# %%
