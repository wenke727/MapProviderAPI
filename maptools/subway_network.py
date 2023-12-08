#%%
"""
WBS:
    x API 获取数据
    x API 数据解析
    - 遍历策略 
    - 数据处理方面
        x station -> line mapping

初步结论：
    1. 候车时间设定比较随意，2 min, 3 min
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


def query_dataframe(df, attr, val=None, contains=None):
    if val is None and contains is None:
        return df
    if contains is None:
        return df.query(f"{attr} == @val ")
    if val is None:
        return df.query(f"{attr}.str.contains('{contains}')", engine='python')

    return df

def get_transit_directions(ori, dst, city1, city2, strategy=0, show_fields='cost,navi', multiexport=1, key=KEY):
    """
    高德地图公交路线规划 API 服务地址

    strategy: 
        0：推荐模式，综合权重，同高德APP默认
        1：最经济模式，票价最低
        2：最少换乘模式，换乘次数少
        3：最少步行模式，尽可能减少步行距离
        4：最舒适模式，尽可能乘坐空调车
        5：不乘地铁模式，不乘坐地铁路线
        6：地铁图模式，起终点都是地铁站（地铁图模式下originpoi及destinationpoi为必填项）
        7：地铁优先模式，步行距离不超过4KM
        8：时间短模式，方案花费总时间最少
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
        'strategy': strategy,
        'show_fields': show_fields,
        'multiexport': multiexport
    }

    response = requests.get(url, params=params)
    return response.text

def parse_transit_directions(response_text, route_type='地铁线路'):
    df = pd.DataFrame()
    data = json.loads(response_text)

    def _extract_steps_from_plan(plan):
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
                    val['type'] = '步行'
                    val['departure_stop'] = val['origin']
                    val['arrival_stop'] = val['destination']
                    del val['origin']
                    del val['destination']
                    steps.append(val)
                
                seg_id += 1

        return pd.DataFrame(steps)

    if 'route' in data:
        route = data['route']
        origin = route.get('origin')
        destination = route.get('destination')

        if 'transits' in route:
            lst = []
            for i, transit in enumerate(route['transits'], start=0):
                df = _extract_steps_from_plan(transit)
                if route_type not in df['type'].unique():
                    continue
                df.loc[:, 'route'] = i
                lst.append(df)
            
            if lst: df = pd.concat(lst)

    df = df.replace('', np.nan).dropna(axis=1, how='all')
    df.rename(columns={'id': 'bid'}, inplace=True)
    df.loc[:, 'cost'] = df.cost.apply(lambda x: x.get('duration', np.nan))

    return df  # 返回解析后的数据

def get_subway_segment_info(line_stations, strategy=2):
    # the sencod and following segments
    links = []
    directions_res = []
    
    src = line_stations.iloc[0]
    for dst in line_stations.iloc[1:].itertuples():
        logger.info(f"{src['name']} -> {dst.name}")
        response_text = get_transit_directions(src.location, dst.location, '0755', '0755', strategy)
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
    src, dst = line_stations.iloc[1].location, line_stations.iloc[-1].location
    response_text = get_transit_directions(src, dst, '0755', '0755', strategy)
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
    nodes = pd.json_normalize(nodes).rename(columns={'id': 'nid'})
    nodes = nodes.merge(
        line_stations.rename(columns={'id': 'bvid', 'location': 'station_coord'}),
        on='name', how='left')
    
    df_links.src = df_links.src.apply(lambda x: x['id'])
    df_links.dst = df_links.dst.apply(lambda x: x['id'])

    return nodes.set_index('nid'), df_links, directions_res

def get_subway_interval():
    # TODO 获取地铁的发车时间间隔
    pass

def extract_exchange_data():
    # TODO 提取换乘的信息
    pass

def query_transit_by_pinyin(src, dst, df_stations, strategy=0):
    """ 通过中文的`起点站名`和`终点站名`查询线路 """
    nanshan = df_stations.query(f"name == '{src}'").location
    shangmeilin = df_stations.query(f"name == '{dst}'").location

    response_text = get_transit_directions(nanshan, shangmeilin, '0755', '0755', strategy)
    commute = parse_transit_directions(response_text)
    return commute

def extract_stations_from_lines(df_lines):
    stations = df_lines.busstops.apply(eval).explode()
    df_stations = pd.json_normalize(stations)
    df_stations.index = stations.index

    keep_attrs = ['id', 'name']
    reanme_dict = {'name': 'line_name', 'id': 'line_id'}

    df_stations = df_stations.merge(
        df_lines[keep_attrs].rename(columns=reanme_dict), 
        left_index=True, right_index=True, how='left'
    )

    return df_stations.reset_index(drop=True)


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


""" iniatilization """
df_lines = gpd.read_file('/Users/wenke/Library/CloudStorage/OneDrive-Personal/3_Codes/MapTools/data/subway/wgs/shenzhen_subway_lines_wgs.geojson')
df_stations = extract_stations_from_lines(df_lines)
# df_stations = gpd.read_file('../data/subway/wgs/shenzhen_subway_station_wgs.geojson')

# query_dataframe(df_stations, 'name', '南山')

metro = MetroNetwork()


#%%
#! Bad Case 
routes = query_transit_by_pinyin('左炮台东', '福永', df_stations, strategy=2)
routes

# %%
""" 获取某一条线路的信息 """
# metro = MetroNetwork()

# 0: 1, 22: 11, 24: 12
for line_id in [0]:
    line = df_lines.loc[line_id]
    line_stations = query_dataframe(df_stations, 'line_id', line['id'])
    
    nodes, edges, directions_res = get_subway_segment_info(line_stations)
    metro.add_nodes(nodes)
    metro.add_edges(edges)

metro.edges_to_dataframe()
metro.nodes_to_dataframe()


# %%
routes = query_transit_by_pinyin('南山', '上梅林', df_stations)
routes

#%%
#! 增加换乘的记录
route_0 = query_dataframe(routes, 'route', 0)
subways = route_0.iloc[::2]
walkings = route_0.iloc[1::2]

cond = (subways['type'] == '地铁线路').all() and \
       (walkings['type'] == '步行').all() and \
       len(subways) - 1 == len(walkings)

links = []
for i in range(walkings.shape[0]):
    links.append({
        'src': subways.iloc[i].arrival_stop['id'],
        'dst': subways.iloc[i + 1].departure_stop['id'],
        'distance': walkings.iloc[i].distance,
        'duration': walkings.iloc[i].cost,
        'type': '步行',
        'steps': walkings.iloc[i].steps,
    })
for link in links:
    metro.add_edge(**link)


# %%
