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
from shapely.ops import linemerge
from shapely import LineString, MultiLineString

from tilemap import plot_geodata
from maptools.trajectory import Trajectory
from maptools.geo.serialization import read_csv_to_geodataframe

from mapmatching import ST_Matching
from mapmatching.graph import GeoDigraph
from mapmatching.geo.io import read_csv_to_geodataframe, to_geojson

from maptools.utils.misc import set_chinese_font_style
from maptools.utils.logger import make_logger, logger_dataframe


set_chinese_font_style()
UTM_CRS = None

#%%

def _save_fig(fig, fn):
    return fig.savefig(fn, bbox_inches='tight', pad_inches=0.1, dpi=500)

def _judge_lable(probs:dict, eps=.6):
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

def cal_temporal_prob(actual_duration, avg_duration, avg_waiting, factor=5, bias=120):
    """
    根据正态分布函数计算概率。
    
    :param actual_duration: 实际通行时间
    :param avg_duration: 平均预测通行时间
    :param sigma: 标准差
    :return: 给定实际通行时间的可能性
    """
    min_duration = avg_duration - avg_waiting
    if min_duration * .6 > actual_duration:
        return 0
    
    sigma = avg_waiting + bias
    
    return np.exp(-((actual_duration - avg_duration) ** 2) / (factor * sigma ** 2))

def get_time_params(traj, df_path, lineid_2_waitingtime):
    """
    cal_time_prob
    小于列车行驶时间的总和的 95%，认为是不可能
    增加首站候车时间
    增加首站候车时间 [0, 2 * waiting], 取最靠近 1 的数值
    time_prob = expect_duration  / duration
    计算一个最短的时间，即每一个线路都没有等候，直接上车
    """
    actual_duration = traj.raw_df.dt.max() - traj.raw_df.dt.min()
    excahnge_links = df_path.query("dst_name in ['exchange', 'inner_link']")
    waiting_time = excahnge_links.duration.sum() - excahnge_links.walking_duration.sum()
    first_watiting_time = lineid_2_waitingtime[df_path.iloc[0].way_id]

    _sum = df_path.duration.sum()
    min_duration = _sum - waiting_time
    avg_duration = _sum + first_watiting_time
    avg_waiting = waiting_time + first_watiting_time

    temporal_prob = cal_temporal_prob(actual_duration, avg_duration, avg_waiting)
    
    return actual_duration, avg_duration, avg_waiting,  temporal_prob

def trim_first_and_last_step(df_path, res, eps=.2):
    # 裁剪首尾段
    df_path = df_edges.loc[res['epath']]
    if res['status'] != 0:
        return df_path
    
    start = 0 if res['step_0'] < eps else 1
    _len  = len(res['epath']) 
    end = _len - 2 if res['step_n'] < eps else _len - 1

    if df_path.iloc[start].dst_name in ['exchange',  'inner_link']:
        start += 1
    if df_path.iloc[end].dst_name in ['exchange',  'inner_link']:
        end -= 1
    
    # update `res`
    res['epath'] = res['epath'][start: end + 1]
    if start != 0:
        res['step_0'] = 0
    if end != _len - 1:
        res['step_n'] = 1
    
    # update `path`
    df_path = df_path.iloc[start: end + 1]
    df_combined_path = process_path_data(df_path)
    
    return df_combined_path

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
    traj.raw_df.to_crs(4326).plot(ax=ax, color='b', alpha=.4, marker='x', zorder=1)

    segs = traj.to_line_gdf().to_crs(4326)
    segs.plot(ax=ax, color='b', alpha=.6, linestyle=':', zorder=2)

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
    for i, p in od.iterrows():
        ax.text(
            p.geometry.x, p.geometry.y + delta_y, p['name'], transform=ax.transData,
            bbox=dict(facecolor='white', alpha=0.4, edgecolor='none', boxstyle="round,pad=0.5"),
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
        search_radius=500, top_k=8,
        dir_trans=False, 
        details=False, 
        plot=False, 
        simplify=False,
        debug_in_levels=False
    )
    res['match_res'] = match_res
    logger.debug(match_res)

    if len(match_res.get('epath', [])) == 0:
        print("status: ", match_res['status'])
        logger.warning("No candidates/轨迹点无法映射到候选边上")
        return traj, match_res, pd.DataFrame()

    # step 3: postprocess `df_path`
    df_path = traj.align_crs(df_edges.loc[match_res['epath']])
    df_path = trim_first_and_last_step(df_path, match_res, eps=0.1)
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
        fig, ax = plot_helper(traj, matcher, match_res, f"{title}, {trip_info}")
        res['fig'] = fig
        res['ax'] = ax
        if save_img:
            fig.savefig(save_img, bbox_inches='tight', pad_inches=0.1, dpi=500)
            plt.close()
    
    return res

def exp(folder):
    folder = Path(folder)
    logger = make_logger(folder, 'cell', console=True, include_timestamp=False)

    img_folder = folder / 'imgs'
    img_folder_0 = img_folder / 'Subway'
    img_folder_1 = img_folder / 'NoSubway'
    for f in [folder, img_folder_0, img_folder_1]:
        # if f.exists():
        #     os.remove(f)
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
            _res = pipeline(pts, traj_id=traj_id, plot_time_dist=False, title=Path(fn).name)
            traj = _res['traj']
            
            # collect geojson
            traj.raw_df.loc[:, 'traj_id'] = int(fn_name)
            raw_points_res.append(traj.raw_df)
            if not traj.points.empty:
                points_res.append(traj.points)
            if not _res['df_path'].empty:
                _res['df_path'].loc[:, 'traj_id'] = int(fn_name)
                path_res.append(_res['df_path'])

            # save img for debug
            lable = _judge_lable(_res['matching']['probs'])
            if lable:
                _save_fig(_res['fig'], img_folder_0 / f'{fn_name}.jpg')
            else:
                _save_fig(_res['fig'], img_folder_1 / f'{fn_name}.jpg')
            plt.close()
            
            # collect probs
            _res['matching']['probs'] = {"fn": fn, **_res['matching']['probs']}
            matching_res.append(_res['matching']['probs'])
            
        except:
            logger.error(fn)

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


    exp('./exp/12-06/0800')


#%%

folder = Path('./exp/12-06/0800/csv')
folder = Path('./exp/12-08/0800/csv')

fns = sorted(glob.glob(f"{folder}/*.csv"))

# traj_id = 0
# fn = fns[traj_id]
# pts = read_csv_to_geodataframe(fn)
# res = pipeline(pts, traj_id=traj_id, plot_time_dist=False, title=Path(fn).name, save_img=False)
# traj = res['traj']
# df_path = res['df_path']



# %%
