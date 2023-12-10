#%%
"""
WBS:
    x API 获取数据
    x API 数据解析
    - 遍历策略 
    - 数据处理方面
        x station -> line mapping

初步结论: 
    1. 候车时间设定比较随意, 2 min, 3 min
"""


#%%
import time
import json
import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
from copy import deepcopy
from loguru import logger
from itertools import islice

from cfg import KEY
from provider.direction import query_transit_directions
from utils.logger import make_logger
from utils.dataframe import query_dataframe
from utils.serialization import load_checkpoint, save_checkpoint
logger = make_logger('../cache', 'network', include_timestamp=False)

ROUTE_COLUMNS = ['route', 'seg_id', 'type', 'name', 'departure_stop', 'arrival_stop',  'distance', 'cost']


""" 弃用 """
def __route_formatter():
    idx_0 = 1 if route.iloc[0].type == '步行' else 0
    idx_n = -1 if route.iloc[-1].type == '步行' else None

    filter_route = route.iloc[idx_0: idx_n]

    instructions = []
    for seg in route.itertuples():
        if seg.type == '步行':
            instructions.append(f"步行{seg.distance}m({seg.cost}s)")
        if seg.type == "地铁线路":
            line = seg.name.split('(')[0]
            src, dst = seg.departure_stop['name'], seg.arrival_stop['name']
            instructions.append(f"\n{line}({src}--{dst}): {int(seg.cost)/60:.1f}m, ")

    logger.debug("".join(instructions))

""" 辅助函数 """
def _split_line_name(series):
    """
    Splits a pandas Series containing metro line data into three parts: Line Name, Alias, and Direction.

    Parameters:
    - series (pd.Series): A pandas Series containing strings in the format 'Line Name(Alias)(Direction)' or 'Line Name(Direction)'.

    Returns:
    - pd.DataFrame: A DataFrame with columns 'Line Name', 'Alias', 'Direction'.
    """
    split_data = series.str.extract(r'(.*?)(?:\((.*?)\))?\((.*?)\)')
    split_data.columns = ['line_name', 'alias', 'direction']

    return split_data

def _load_subway_lines(line_fn):
    df_lines = gpd.read_file(line_fn).rename(columns={'id': 'line_id'})
    names = _split_line_name(df_lines.name)
    df_lines = pd.concat([names, df_lines], axis=1)
    
    return df_lines

def _extract_stations_from_lines(df_lines):
    stations = df_lines.busstops.apply(eval).explode()
    df_stations = pd.json_normalize(stations)
    df_stations.index = stations.index

    keep_attrs = ['line_id', 'line_name']

    df_stations = df_stations.merge(
        df_lines[keep_attrs], 
        left_index=True, right_index=True, how='left'
    )

    return df_stations.reset_index(drop=True)

""" 解析函数 """
def filter_dataframe_columns(df, cols=ROUTE_COLUMNS):
    cols = [i for i in cols if i in list(df)]
    
    return df[cols]

def parse_transit_directions(data, route_type='地铁线路'):
    df = pd.DataFrame()

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
                if route_type is not None and route_type not in df['type'].unique():
                    continue
                df.loc[:, 'route'] = i
                lst.append(df)
            
            if lst: df = pd.concat(lst)

    df = df.replace('', np.nan).dropna(axis=1, how='all')
    df.rename(columns={'id': 'bid'}, inplace=True)
    df.loc[:, 'cost'] = df.cost.apply(lambda x: x.get('duration', np.nan))

    return df  # 返回解析后的数据

def get_subway_segment_info(line_stations, strategy=2):
    links = []
    directions_res = []
    
    # 第一个站点作为起始点
    src = line_stations.iloc[0]
    for i, dst in enumerate(line_stations.iloc[1:].itertuples(), start=1):
        logger.info(f"{src['name']} -> {dst.name}")
        response_data = query_transit_directions(src.location, dst.location, '0755', '0755', KEY, strategy)
        plans = parse_transit_directions(response_data)
        routes = plans.query("type == '地铁线路'")
        if routes.empty:
            info = f"Failed: query_transit_directions('{src.location}', '{dst.location}', '0755', '0755', {strategy})"
            logger.warning(info)
            continue

        directions_res.append(routes.iloc[[0]])
        if i > 1:
            cur = directions_res[-1].iloc[0]
            prev = directions_res[-2].iloc[0]
            link = {
                'src': prev.arrival_stop,
                'dst': cur.arrival_stop,
                'distance': int(cur.distance) - int(prev.distance),
                'cost': int(cur.cost) - int(prev.cost),
            }
            links.append(link)
        time.sleep(1)

    # the fisrt segment
    src, dst = line_stations.iloc[1].location, line_stations.iloc[-1].location
    response_data = query_transit_directions(src, dst, '0755', '0755', KEY, strategy)
    plans = parse_transit_directions(response_data)
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

def query_transit_by_pinyin(src, dst, df_stations, strategy=0, route_type=None):
    """ 通过中文的`起点站名`和`终点站名`查询线路 """
    start = df_stations.query(f"name == '{src}'")
    end = df_stations.query(f"name == '{dst}'")

    response_text = query_transit_directions(start.location, end.location, '0755', '0755', KEY, strategy)
    commute = parse_transit_directions(response_text, route_type)
    
    return commute


class MetroNetwork:
    def __init__(self, line_fn, ckpt=None, refresh=True):
        self.ckpt = ckpt
        if ckpt is not None:
            try:
                load_checkpoint(ckpt, self)
                return
            except:
                logger.warning(f"load {ckpt} failed!")
            
            
        self.graph = nx.DiGraph()
        self.visited_lines = set([])
        
        # lines
        self.df_lines = _load_subway_lines(line_fn)
        
        self.df_stations = _extract_stations_from_lines(self.df_lines)

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

    def add_line(self, line_id, strategy=2):
        """ 获取某一条线路的信息 """
        if line_id in self.visited_lines:
            return
        
        stations = query_dataframe(self.df_stations, 'line_id', line_id)
        nodes, edges, directions_res = get_subway_segment_info(stations, strategy=strategy)
        self.add_nodes(nodes)
        self.add_edges(edges)
        self.visited_lines.add(line_id)
    
        return nodes, edges, directions_res

    def get_node(self, nid, attr=None):
        if nid in self.graph.nodes:
            item = self.graph.nodes[nid]
            if attr in item:
                return item.get(attr, None)
            return item
        
        return None

    @property
    def exchange_stations(self):
        tmp = self.df_stations.groupby('name')\
                .agg({'id': 'count', 'line_id': list})\
                .query("id > 2")\
                .reset_index().rename(columns={'id': 'num'})

        return tmp

    def get_adj_nodes(self, nid, type='same_line'):
        assert type in ['same_line', 'exchange', None]
        
        def _get_adj(func):
            res = []
            for i in iter(func(nid)):
                if type is None or type == '':
                    res.append(i)
                    continue
                
                line_id = self.get_node(nid, 'line_id')
                if type == 'same_line' and self.get_node(i, 'line_id') == line_id:
                    res.append(i)
                if type == 'exchange' and  self.get_node(i, 'line_id') != line_id:
                    res.append(i)
                    
            return res

        nodes = {
            'cur': nid,
            'prev': _get_adj(self.graph.predecessors),
            'next': _get_adj(self.graph.successors),
        }
        
        return nodes

    def save_ckpt(self, ckpt=None):
        if ckpt is not None:
            return save_checkpoint(self, ckpt)
        if self.ckpt is not None:
            return save_checkpoint(self, self.ckpt)

        return False
    
    def get_subway_interval(self, ):
        # TODO 获取地铁的发车时间间隔
        pass

""" 待开发 """
def check_shortest_path():
    src = "440300024057010" # 11号线，南山
    dst = "440300024057013" # 11号线，福田

    routes = metro.top_k_paths(src, dst, 3, weight='cost')
    nodes.loc[routes[0]]


if __name__ == "__main__":
    line_fn = '../data/subway/wgs/shenzhen_subway_lines_wgs.geojson'
    ckpt = '../data/subway/shenzhen_network.ckpt'
    # ckpt = None
    metro = MetroNetwork(line_fn=line_fn, ckpt=ckpt)

    self = metro
    G = metro.graph
    df_lines = metro.df_lines
    df_stations = metro.df_stations

    # metro.add_line('440300024064')
    # metro.add_line('440300024063')
    # metro.add_line('440300024077')
    # metro.add_line('440300024076')
    metro.add_line('440300024061') # 地铁3号线(龙岗线)(福保--双龙) 
    metro.add_line('440300024075') # 地铁4号线(龙华线)(福田口岸--牛湖)
    metro.add_line('440300024056') # 地铁11号线(机场线)(岗厦北--碧头)
    metro.add_line('440300024057') # 地铁11号线(机场线)(碧头--岗厦北)
    metro.add_line('900000094862') # 地铁12号线(南宝线)(海上田园东--左炮台东)

    line_id = '900000094863' # 地铁12号线(南宝线)(左炮台东--海上田园东)
    metro.add_line(line_id)
    
    # nodes, edges
    nodes = metro.nodes_to_dataframe()
    edges = metro.edges_to_dataframe()



#%%

def get_exchange_link_info(nodes, station_name):
    #! 如何获取一个换乘站的所有连接
    nidxs = np.sort(query_dataframe(nodes, 'name', station_name).index)
    logger.debug(f"{station_name}: {nidxs}")

    lst = []
    for nid in nidxs:
        lst.append(metro.get_adj_nodes(nid))
    points = pd.DataFrame(lst)
    points.loc[:, 'prev'] = points['prev'].apply(lambda x: x[0] if isinstance(x, list) else x)
    points.loc[:, 'next'] = points['next'].apply(lambda x: x[0] if isinstance(x, list) else x)
    points.loc[:, 'line_name'] = points['cur'].apply(lambda x: nodes.loc[x].line_name)

    for i, src in enumerate(points.itertuples()):
        for j, dst in enumerate(points.itertuples()):
            if i == j or src.line_name == dst.line_name:
                continue

            start = nodes.loc[src.prev]
            end = nodes.loc[dst.next]
            logger.debug(f"{start['name']} -> {end['name']}")

station_name = '南山'
nodes = metro.nodes_to_dataframe()
get_exchange_link_info(nodes, station_name)


#%%
def get_exchange_info():
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

#%%
"""
#! 重新梳理流程
# TODO
    x 第一段的步行和最后一段的步行可以忽略
    x formatter
    - 校核起点和终点是否一致；
"""

# src, dst = '后海', '南光'
src, dst = '南山', '上梅林'
# src, dst = '左炮台东', '福永'

strategy = 0
route_type = "地铁线路"
start = df_stations.query(f"name == '{src}'")
end = df_stations.query(f"name == '{dst}'")

response_data = query_transit_directions(start.location, end.location, '0755', '0755', KEY, strategy)

#%%

def parse_transit_directions(data, route_type='地铁线路'):
    def _extract_steps_from_plan(route, route_id):
        steps = []
        for seg_id, segment in enumerate(route['segments']):
            connector = 'walking_0'
            step = {"seg_id": seg_id, 'sub_seg': ",".join(segment.keys())}
            for key, val in segment.items():
                val = deepcopy(val)
                if key == 'bus':
                    connector = 'walking_1'
                    if len(val['buslines']) != 1:
                        # FIXME 针对公交场景，存在2条或以上的公交车共线的情况，但就地铁而言，可能不存在此情况
                        logger.warning(f"Check route {route_id} the buslines length:\n{val}")
                    step.update(val['buslines'][0])

                if key == 'walking':
                    step[connector] = val
                    step[connector+"_info"] = {
                        "cost": int(val['cost']['duration']), 
                        "distance": int(val['distance'])
                    }
                    
            steps.append(step)                    

        # 删除首尾步行的部分
        steps = pd.DataFrame(steps)
        if steps.iloc[0].type != steps.iloc[0].type:
            steps = steps.iloc[1:]
        if steps.iloc[-1].type != steps.iloc[-1].type:
            steps = steps.iloc[:-1]
        
        return steps

    df = pd.DataFrame()
    transits = data.get('route', {}).get("transits")
    if not transits:
        logger.warning("No tranists records!")
        return df
    
    lst = []
    for i, transit in enumerate(transits, start=0):
        df = _extract_steps_from_plan(transit, i)
        if route_type is not None and route_type not in df['type'].unique():
            continue
        df.loc[:, 'route'] = i
        lst.append(df)
    
    if lst: df = pd.concat(lst).reset_index(drop=True)

    df = df.replace('', np.nan).dropna(axis=1, how='all')
    df.rename(columns={'id': 'bid'}, inplace=True)
    df.loc[:, 'cost'] = df.cost.apply(lambda x: x.get('duration', np.nan) if isinstance(x, dict) else x)

    return df

routes = parse_transit_directions(response_data, route_type)
routes = filter_dataframe_columns(routes, ROUTE_COLUMNS + ['walking_0_info', 'walking_1_info', 'sub_seg'])
routes.departure_stop = routes.departure_stop.apply(lambda x: x['name'])
routes.arrival_stop = routes.arrival_stop.apply(lambda x: x['name'])
routes

# %%
str_routes = []
for route_id in routes.route.unique():
    route = routes.query(f"route == {route_id}")
    str_routes.append(f"Route {route_id}:\n{route}")

pre_states = ""
logger.debug(pre_states + "\n" + "\n\n".join(str_routes))



# %%
