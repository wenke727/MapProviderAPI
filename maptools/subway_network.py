#%%
"""
WBS:
    x API 获取数据
    x API 数据解析
    - 遍历策略 
    - 数据处理方面
        - station -> line mapping
"""


#%%
import json
import requests
import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
from copy import deepcopy
from loguru import logger
from itertools import islice

from cfg import KEY

from utils.logger import make_logger
logger = make_logger('../cache', 'network')


def get_transit_directions(ori, dst, city1, city2, show_fields='cost,navi', multiexport=1, key=KEY):
    """
    高德地图公交路线规划 API 服务地址
    
    Ref:
        - https://lbs.amap.com/api/webservice/guide/api/newroute#t9
    """
    url = "https://restapi.amap.com/v5/direction/transit/integrated"
    params = {
        'key': key,
        'origin': ori,
        'destination': dst,
        'city1': city1,
        'city2': city2,
        'show_fields': show_fields,
        'multiexport': multiexport
    }

    response = requests.get(url, params=params)
    return response.text

def extract_steps_from_plan(plan):
    steps = []
    seg_id = 0
    for segment in plan['segments']:
        for key, val in segment.items():
            val = deepcopy(val)
            if 'bus' == key:
                for line in val['buslines']:
                    line['seg_id'] = seg_id
                    steps.append(line)

            if 'walking' == key:
                val['seg_id'] = seg_id
                val['type'] = 'walking'
                val['departure_stop'] = val['origin']
                val['arrival_stop'] = val['destination']
                del val['origin']
                del val['destination']
                steps.append(val)
            
            seg_id += 1

    return pd.DataFrame(steps)

def parse_transit_directions(response_text):
    data = json.loads(response_text)
    # logger.debug(f"Status: {data.get('status')}, Info: {data.get('info')}, Total Routes: {data.get('count')}")
    df = pd.DataFrame()
    
    if 'route' in data:
        route = data['route']
        origin = route.get('origin')
        destination = route.get('destination')

        if 'transits' in route:
            # logger.debug(route)
            lst = []
            for i, transit in enumerate(route['transits'], start=0):
                distance = transit.get('distance')
                cost = transit.get('cost')
                # logger.debug(f"Plan {i}: distance: {distance}, cost: {cost}")
                
                df = extract_steps_from_plan(transit)
                df.loc[:, 'plan'] = i
                lst.append(df)
                
            df = pd.concat(lst)

    df = df.replace('', np.nan).dropna(axis=1, how='all')
    df.loc[:, 'cost'] = df.cost.apply(lambda x: x.get('duration', np.nan))

    return df  # 返回解析后的数据

def get_subway_segment_info(demo_stations):
    # the sencod and following segments
    links = []
    directions_res = []
    
    src = demo_stations.iloc[0]
    for dst in demo_stations.iloc[1:].itertuples():
        logger.info(f"{src['name']} -> {dst.name}")
        response_text = get_transit_directions(src.location, dst.location, '0755', '0755')
        plans = parse_transit_directions(response_text)
        directions_res.append(plans.query("type == '地铁线路'").iloc[[0]])

    # TODO 查看顺序
    for i in range(len(directions_res) - 1):
        cur = directions_res[i].iloc[0]
        nxt = directions_res[i + 1].iloc[0]
        links.append({
            'src': cur.arrival_stop, 
            'dst': nxt.arrival_stop ,
            'distance': int(nxt.distance) - int(cur.distance),
            'cost': int(nxt.cost) - int(cur.cost),
        })

    # the fisrt segment
    src, dst = demo_stations.iloc[1].location, demo_stations.iloc[-1].location
    response_text = get_transit_directions(src, dst, '0755', '0755')
    plans = parse_transit_directions(response_text)
    directions_res.append(plans.query("type == '地铁线路'").iloc[[0]])

    cur = directions_res[-1].iloc[0]
    prev = directions_res[-2].iloc[0]
    links = [{
        "src": prev.departure_stop,
        "dst": cur.departure_stop,
        'distance': int(prev.distance) - int(cur.distance),
        'cost': int(prev.cost) - int(cur.cost)
    }] + links

    df_links = pd.DataFrame(links)
    nodes = np.concatenate([df_links.src.values, df_links.iloc[[-1]].dst.values])
    nodes = pd.json_normalize(nodes).set_index('id')
    
    df_links.src = df_links.src.apply(lambda x: x['id'])
    df_links.dst = df_links.dst.apply(lambda x: x['id'])

    return nodes, df_links, directions_res

def get_subway_interval():
    # TODO 获取地铁的发车时间间隔
    pass

def extract_exchange_data():
    # TODO 提取换乘的信息
    pass


class MetroNetwork:
    def __init__(self):
        self.graph = nx.Graph()

    def add_node(self, id, *args, **kwargs):
        self.graph.add_node(id, *args, **kwargs)

    def add_nodes(self, nodes:pd.DataFrame):
        for id, node in zip(nodes.index, nodes.to_dict(orient='records')):
            self.add_node(id, **node)

    def add_edge(self, src, dst, distance=0, duration=0, *args, **kwargs):
        self.graph.add_edge(src, dst, distance=distance, duration=duration, *args, **kwargs)

    def add_edges(self, edges:pd.DataFrame):
        for link in edges.itertuples():
            self.add_edge(link.src, link.dst, distance=link.distance, duration=link.cost)

    def shortest_path(self, source, target, weight='distance'):
        try:
            return nx.shortest_path(self.graph, source, target, weight=weight)
        except nx.NetworkXNoPath:
            return "No path found"

    def top_k_paths(self, source, target, k, weight='distance'):
        try:
            return list(islice(nx.shortest_simple_paths(self.graph, source, target, weight=weight), k))
        except nx.NetworkXNoPath:
            return "No path found"

    def nodes_to_dataframe(self):
        return pd.DataFrame.from_dict(dict(self.graph.nodes(data=True)), orient='index')

    def edges_to_dataframe(self):
        edges_data = self.graph.edges(data=True)
        return pd.DataFrame([{'src': u, 'dst': v, **data} for u, v, data in edges_data])


metro = MetroNetwork()

#%%
df_lines = gpd.read_file('../data/subway/wgs/shenzhen_subway_lines_wgs.geojson')
df_stations = gpd.read_file('../data/subway/wgs/shenzhen_subway_station_wgs.geojson')

#%%
nanshan = df_stations.query("name == '南山'").location
shangmeilin = df_stations.query("name == '上梅林'").location

response_text = get_transit_directions(nanshan, shangmeilin, '0755', '0755')
response_text

commute = parse_transit_directions(response_text)
commute

# %%
""" 
初步结论：
    1. 候车时间设定比较随意，2 min, 3 min
"""

for line_id in [24]:
    line = df_lines.loc[line_id]
    demo_stations = pd.json_normalize(json.loads(line.busstops))
    nodes, edges, directions_res = get_subway_segment_info(demo_stations)

    metro.add_nodes(nodes)
    metro.add_edges(edges)

#%%
metro.nodes_to_dataframe()

#%%
nodes.merge(demo_stations.rename(columns={'id': 'bid', 'location': 'station_coord'}), on='name', how='left')


# %%

commute = parse_transit_directions(response_text)
navi_0 = commute.query('plan == 0')
navi_0

# %%
subways = navi_0.iloc[::2]
walkings = navi_0.iloc[1::2]

cond = (subways['type'] == '地铁线路').all() and \
       (walkings['type'] == 'walking').all() and \
       len(subways) - 1 == len(walkings)

links = []
for i in range(walkings.shape[0]):
    links.append({
        'src': subways.iloc[i].arrival_stop['id'],
        'dst': subways.iloc[i + 1].departure_stop['id'],
        'distance': walkings.iloc[i].distance,
        'duration': walkings.iloc[i].cost,
        'type': 'walking',
        'steps': walkings.iloc[i].steps,
    })
for link in links:
    metro.add_edge(**link)

# %%
metro.edges_to_dataframe()

# %%
