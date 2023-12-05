#%%
"""
WBS:
    - API 获取数据
    - API 数据解析
    - 遍历策略 
    - 数据处理方面
        - station -> line mapping
"""

walking = {
        "destination": "114.028694,22.535616",
        "distance": "297",
        "origin": "114.025826,22.536245",
        "cost": {
            "duration": "254"
        },
        "steps": [
            {
                "instruction": "步行400米到达车公庙",
                "road": "",
                "distance": "400",
                "navi": {
                    "action": "",
                    "assistant_action": "到达车公庙",
                    "walk_type": "5"
                }
            }
        ]
    }

bus = {
        "buslines": [
            {
                "departure_stop": {
                    "name": "车公庙",
                    "id": "440300024055003",
                    "location": "114.028702,22.535615"
                },
                "arrival_stop": {
                    "name": "上梅林",
                    "id": "440300024055009",
                    "location": "114.060432,22.568440",
                    "exit": {
                        "name": "C口",
                        "location": "114.059319,22.570120"
                    }
                },
                "name": "地铁9号线(梅林线)(前湾--文锦)",
                "id": "440300024055",
                "type": "地铁线路",
                "distance": "7151",
                "cost": {
                    "duration": "947"
                },
                "bus_time_tips": "可能错过末班车",
                "bustimetag": "4",
                "start_time": "",
                "end_time": "",
                "via_num": "5",
                "via_stops": [
                    {
                        "name": "香梅",
                        "id": "440300024055004",
                        "location": "114.039625,22.545491"
                    },
                    {
                        "name": "景田",
                        "id": "440300024055005",
                        "location": "114.043343,22.553419"
                    },
                    {
                        "name": "梅景",
                        "id": "440300024055023",
                        "location": "114.037934,22.561028"
                    },
                    {
                        "name": "下梅林",
                        "id": "440300024055024",
                        "location": "114.041768,22.565672"
                    },
                    {
                        "name": "梅村",
                        "id": "440300024055025",
                        "location": "114.052423,22.568443"
                    }
                ]
            }
        ]
    }
segment = {
    "walking": walking,
    "bus": bus
}

#%%
import json
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
from copy import deepcopy
from loguru import logger

from cfg import KEY


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
    logger.debug(f"Status: {data.get('status')}, Info: {data.get('info')}, Total Routes: {data.get('count')}")
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
    res = []
    src = demo_stations.iloc[0]
    for dst in demo_stations.iloc[1:].itertuples():
        response_text = get_transit_directions(src.location, dst.location, '0755', '0755')
        plans = parse_transit_directions(response_text)
        
        res.append(plans.query("type == '地铁线路'").iloc[[0]])

    plans = pd.concat(res)
    plans.distance = plans.distance.astype(int)
    plans.cost = plans.cost.astype(int)

    links = []
    for i in range(len(plans) - 1):
        src, dst = plans.iloc[i].arrival_stop, plans.iloc[i + 1].arrival_stop 
        distance = plans.iloc[i + 1].distance - plans.iloc[i].distance
        duration = plans.iloc[i + 1].cost - plans.iloc[i].cost
        
        links.append({
            'src': src, 
            'dst': dst,
            'distance': distance,
            'cost': duration,
        })

    # the fisrt segment
    response_text = get_transit_directions(
        demo_stations.iloc[1].location, 
        demo_stations.iloc[-1].location, 
        '0755', '0755')
    plans = parse_transit_directions(response_text)
    plans = plans.query("type == '地铁线路'")
    cur = plans.iloc[0]
    prev = res[-1].iloc[0]
    res.append(plans.query("type == '地铁线路'").iloc[[0]])

    links = [{
        "src": prev.departure_stop,
        "dst": cur.departure_stop,
        'distance': int(prev.distance) - int(cur.distance),
        'cost': int(prev.cost) - int(cur.cost)
    }] + links

    df_links = pd.DataFrame(links)

    return df_links

def get_subway_interval():
    # TODO 获取地铁的发车时间间隔
    pass

def extract_exchange_data():
    # TODO 提取换乘的信息
    pass

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
line_id = 22
line = df_lines.loc[line_id]
demo_stations = pd.json_normalize(json.loads(line.busstops))
demo_stations

#%%
#! iter items
df_links = get_subway_segment_info(demo_stations)
df_links

# TODO df_lniks 里的节点信息拆分开来 和 demo_stations 合并

# %%
import networkx as nx
from itertools import islice

class MetroNetwork:
    def __init__(self):
        self.graph = nx.Graph()

    def add_node(self, id, *args, **kwargs):
        self.graph.add_node(id, *args, **kwargs)

    def add_edge(self, src, dst, distance=0, duration=0, *args, **kwargs):
        self.graph.add_edge(src, dst, distance=distance, duration=duration, *args, **kwargs)

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

nodes = np.concatenate([df_links.src.values, df_links.iloc[[-1]].dst.values])
for node in nodes:
    metro.add_node(**node)

for link in df_links.itertuples():
    metro.add_edge(link.src['id'], link.dst['id'], distance=link.distance, duration=link.cost)

metro.graph

# print(metro.shortest_path("A", "C", weight='duration'))  # 根据持续时间查找最短路径
# print(metro.top_k_paths("A", "C", 2, weight='distance'))  # 查找两点间距离最短的前两条路径
#%%
nodes_df = pd.DataFrame.from_dict(dict(metro.graph.nodes(data=True)), orient='index')
nodes_df

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
