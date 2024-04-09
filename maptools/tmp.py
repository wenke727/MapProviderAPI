#%%
import time
import pandas as pd
from loguru import logger
from provider.direction import query_transit_directions, get_subway_routes

DIRECTION_MEMO = {}
UNAVAILABEL_STATIONS = set()

stations = pd.read_csv('../tests/cases/地铁2号线外环_station.csv')
stations

#%%
# def get_subway_segment_info(stations, strategy=0, citycode=CITYCODE, sleep_dt=.2, auto_save=True):
strategy = 0; citycode='010'; sleep_dt=.2

logger.info(f"stations: {stations['name'].values.tolist()}")
routes_lst = []
steps_lst = []
walking_lst = []
unavailabel_stops = []

def _query_helper(src, dst):
    routes, steps, walking_steps = get_subway_routes(src, dst, strategy, citycode, memo=DIRECTION_MEMO)
    routes.query("stop_check == True", inplace=True)
    if sleep_dt: 
        time.sleep(sleep_dt)
    
    if routes.empty:
        logger.warning(f"unavailabel stop: {dst['name']}")
        unavailabel_stops.append(dst['name'])
        UNAVAILABEL_STATIONS.add((dst['line_id'], dst['name']))
        return [None] * 4
    
    idx = routes.index.values[0]
    routes_lst.append(routes)
    steps_lst.append(steps.loc[[idx]])
    walking_lst.append(walking_steps)    

# segs: 1 - n
length = 8
src = stations.iloc[0]
for i in range(1, len(stations) + 1):
    j = max(0, i - 1 - length)
    i = i % len(stations)
    _query_helper(stations.iloc[j], stations.iloc[i])

# %%
import numpy as np

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


_extract_segment_info_from_routes(steps_lst)

# %%
_query_helper(stations.iloc[0], stations.iloc[1])
steps_lst[-1]

# %%
_query_helper(stations.iloc[0], stations.iloc[2])
steps_lst[-1]

# %%
_query_helper(stations.iloc[1], stations.iloc[2])
steps_lst[-1]

# %%
