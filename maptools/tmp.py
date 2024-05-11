#%%
import time
import pandas as pd
import numpy as np

from loguru import logger
from provider.direction import query_transit_directions, get_subway_routes

DIRECTION_MEMO = {}
UNAVAILABEL_STATIONS = set()

CITYCODE = "0755"

#%%

def judge_ring(stations):
    return stations.iloc[0].id == stations.iloc[-1].id

def _extract_segment_info_from_routes(routes_lst:list):
    df_segs = pd.concat(routes_lst)
    df_segs['cost'] = df_segs['cost'].astype(np.int64)
    df_segs['distance'] = df_segs['distance'].astype(np.int64)

    _len = df_segs.shape[0]
    edges = []
    for i in range(1, _len):
        prev, cur = df_segs.iloc[i-1], df_segs.iloc[i]
        edges.append({
            'src': prev.arrival_stop['id'],
            'dst': cur.arrival_stop['id'],
            'src_name': prev.arrival_stop['name'], 
            'dst_name': cur.arrival_stop['name'],
            'distance': cur.distance - prev.distance,
            'cost': cur.cost - prev.cost,
        })
        
    df_edges = pd.DataFrame(edges)
    
    return df_edges

def _query_helper(src, dst, strategy=0, citycode=CITYCODE, sleep_dt=.2, routes_lst=[], steps_lst=[], walking_lst=[]):
    routes, steps, walking_steps = get_subway_routes(src, dst, strategy, citycode, memo=DIRECTION_MEMO)
    routes.query("stop_check == True", inplace=True)
    if sleep_dt: 
        time.sleep(sleep_dt)
    
    if routes.empty:
        logger.warning(f"unavailabel subway station interval: {src['name']} -> {dst['name']}")

        return [None] * 4
    
    idx = routes.index.values[0]
    step = steps.loc[[idx]]
    step = step.assign(
        src_seq = src.sequence,
        dst_seq = dst.sequence,
    )

    routes_lst.append(routes)
    steps_lst.append(step)
    walking_lst.append(walking_steps)    

def get_subway_segment_info(stations, strategy=0, citycode=CITYCODE, sleep_dt=.2, auto_save=True):
    # strategy = 0; citycode='010'; sleep_dt=.2
    is_ring = judge_ring(stations)
    if is_ring: logger.warning(f"环线：{fn}")

    logger.info(f"stations: {stations['name'].values.tolist()}")
    routes_lst = []
    steps_lst = []
    walking_lst = []
    unavailabel_stops = []

    query_params = {
        "strategy": strategy, 
        "routes_lst": routes_lst,
         "steps_lst": steps_lst,
        "walking_lst": walking_lst
    }

    length = 2
    for i in range(1, len(stations)):
        j = max(0, i - length)
        logger.info(f"{j} -> {i}: {stations.iloc[j]['name']}, {stations.iloc[i]['name']}")
        _query_helper(stations.iloc[j], stations.iloc[i], **query_params)

    if is_ring:
        _query_helper(stations.iloc[len(stations) - 2], stations.iloc[0], **query_params)
        
    _query_helper(stations.iloc[1], stations.iloc[2], **query_params)

    return steps_lst, unavailabel_stops

def cal_trip_info(od_2_trip_info, src, dst):
    return {
    "distance": od_2_trip_info[src]['distance'] - od_2_trip_info[dst]['distance'],
    'cost': od_2_trip_info[src]['cost'] - od_2_trip_info[dst]['cost'],
}

def get_waiting_time(od_2_trip_info):
     return od_2_trip_info[(1, 2)]['cost'] - cal_trip_info(od_2_trip_info, (0, 2), (0, 1))['cost']


fn = '../tests/cases/110100023099-地铁2号线外环.csv'; CITYCODE = "010"
# fn = '../tests/cases/440300024075-地铁4号线.csv'
fn = '../tests/cases/440300024051-地铁7号线.csv'; CITYCODE = "0755"

stations = pd.read_csv(fn)
stations.sequence -= 1

name_2_seq = stations[['sequence', 'name']]\
    .drop_duplicates(subset='name', keep='first').set_index('name')['sequence'].to_dict()
seq_2_name = stations[['sequence', 'name']].set_index('sequence')['name'].to_dict()

 
steps_lst, unavailabel_stops = get_subway_segment_info(stations=stations)

df_directions = pd.concat(steps_lst)[['line_id', 'src_seq', 'dst_seq', 'departure_stop', 'arrival_stop', 'mode',  'distance', 'cost']].reset_index(drop=True)
df_directions

valid_seqs = sorted(np.unique(df_directions[['src_seq', 'dst_seq']].values.flatten()))
valid_seqs # FIXME 环线

# TODO 
# UNAVAILABEL_STATIONS.add((dst['line_id'], dst['name']))

# %%
od_2_trip_info = df_directions.set_index(['src_seq', 'dst_seq'])[['distance', 'cost']].astype(int).to_dict(orient='index')
od_2_trip_info

#%%
line_waiting_time = get_waiting_time(od_2_trip_info)
logger.info(f"line_waiting_time: {line_waiting_time}")

od_2_trip_info_with_waiting = {}
tmp = od_2_trip_info[(valid_seqs[0], valid_seqs[1])]
tmp['cost'] -= line_waiting_time
od_2_trip_info_with_waiting[(valid_seqs[0], valid_seqs[1])] = tmp

for i in range(1, len(valid_seqs) - 1):
    prv = valid_seqs[i - 1]
    cur = valid_seqs[i]
    nxt = valid_seqs[i + 1]
    logger.info(f"{prv}, {cur}, {nxt}")
    
    try:
        tmp = cal_trip_info(od_2_trip_info, (prv, nxt), (prv, cur))
        tmp['cost'] -= line_waiting_time
        od_2_trip_info[(cur, nxt)] = tmp
        od_2_trip_info_with_waiting[(cur, nxt)] = od_2_trip_info[(cur, nxt)] 
    except:
        pass

res = pd.DataFrame.from_dict(od_2_trip_info_with_waiting).T
# res.reset_index(inplace=True)
res = res.assign(
    src_name = res.index.get_level_values(0).map(seq_2_name.get),
    dst_name = res.index.get_level_values(1).map(seq_2_name.get)
)
res[['src_name', 'dst_name', 'distance', 'cost']]


# %%
steps_lst = []
_query_helper(stations.iloc[9], stations.iloc[12], steps_lst=steps_lst)
steps_lst[0].iloc[0][['distance', 'cost']].astype(int).to_dict()

# %%
