#%%
import time
import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
from loguru import logger
from itertools import islice

from cfg import KEY, DATA_FOLDER, ROUTE_COLUMNS
from provider.direction import get_subway_routes
from utils.logger import make_logger
from utils.dataframe import query_dataframe
from utils.serialization import load_checkpoint, save_checkpoint

DIRECTION_MEMO = {}
DIRECTION_MEMO_FN = DATA_FOLDER / "direction_memo.pkl"
DIRECTION_MEMO = load_checkpoint(DIRECTION_MEMO_FN)

logger = make_logger(DATA_FOLDER, 'network', include_timestamp=False)

#%%

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
    if "status" in list(df_lines):
        df_lines.status = df_lines.status.astype(int)
        unavailabel_line = df_lines.query("status != 1").name.unique().tolist()
        logger.warning(f"Unavailabel lines: {unavailabel_line}")
        df_lines.query('status == 1', inplace=True)

    names = _split_line_name(df_lines.name)
    df_lines = pd.concat([names, df_lines], axis=1)
    
    return df_lines

def _extract_stations_from_lines(df_lines, keep_attrs = ['line_id', 'line_name']):
    stations = df_lines.busstops.apply(eval).explode()
    df_stations = pd.json_normalize(stations)
    df_stations.index = stations.index

    df_stations = df_stations.merge(
        df_lines[keep_attrs], 
        left_index=True, right_index=True, how='left'
    )

    return df_stations.reset_index(drop=True)

def _get_exchange_station_names(df_stations):
    """ get the name list of exhange stations. """
    tmp = df_stations.groupby('name')\
            .agg({'id': 'count', 'line_id': list, 'location': np.unique})\
            .query("id > 2")\
            .reset_index().rename(columns={'id': 'num'})
    tmp.loc[:, 'link'] = tmp.location.apply(lambda x: len(x) > 1)
    
    return tmp.sort_values(['num', 'link'], ascending=False)


""" 解析函数 """
def filter_dataframe_columns(df, cols=ROUTE_COLUMNS):
    cols = [i for i in cols if i in list(df)]
    
    return df[cols]

def query_transit_by_pinyin(src, dst, df_stations, strategy=0, route_type=None):
    """ 通过中文的`起点站名`和`终点站名`查询线路 """
    start = df_stations.query(f"name == '{src}'")
    end = df_stations.query(f"name == '{dst}'")

    response_text = query_transit_directions(start.location, end.location, '0755', '0755', KEY, strategy, memo=DIRECTION_MEMO)
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

    # FIXME
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

    def get_exchange_stations(self):
        return _get_exchange_station_names(self.df_stations)

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


""" 待开发 & 开发 """
def check_shortest_path():
    src = "440300024057010" # 11号线，南山
    dst = "440300024057013" # 11号线，福田

    routes = metro.top_k_paths(src, dst, 3, weight='cost')
    nodes.loc[routes[0]]

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

def print_routes(routes):
    _routes = filter_dataframe_columns(routes, ROUTE_COLUMNS + ['walking_0_info', 'walking_1_info', 'mode'])
    str_routes = []
    for route_id in routes.route.unique():
        route = _routes.query(f"route == {route_id}").copy()
        route.departure_stop = route.departure_stop.apply(lambda x: x['name'])
        route.arrival_stop = route.arrival_stop.apply(lambda x: x['name'])
        # route.drop(columns=['route'], inplace=True)
        str_routes.append(f"Route {route_id}:\n{route}")

    pre_states = ""
    logger.debug(pre_states + "\n" + "\n\n".join(str_routes))

def _extract_segment_info_from_routes(routes_lst:list):
    df_segs = pd.concat(routes_lst)
    df_segs['cost'] = df_segs['cost'].astype(int)
    df_segs['distance'] = df_segs['distance'].astype(int)

    _len = df_segs.shape[0]
    edges = []
    for i in range(1, _len):
        prev, cur = df_segs.iloc[i-1], df_segs.iloc[i]
        if i < _len - 1:
            edges.append({
                'src': prev.arrival_stop['id'],
                'dst': cur.arrival_stop['id'],
                'src_name': prev.arrival_stop['name'], 
                'dst_name': cur.arrival_stop['name'],
                'distance': cur.distance - prev.distance,
                'cost': cur.cost - prev.cost,
            })
        else:
            edges = [{
                'src': prev.departure_stop['id'],
                'dst': cur.departure_stop['id'],
                'src_name': prev.departure_stop['name'], 
                'dst_name': cur.departure_stop['name'],
                'distance': - cur.distance + prev.distance,  
                'cost': - cur.cost + prev.cost,
            }] + edges
        
    df_edges = pd.DataFrame(edges)
    
    return df_edges

def get_subway_segment_info(stations, strategy=0, citycode='0755', memo={}):
    if stations.empty or 'name' not in stations.columns:
        logger.error("Invalid stations data")
        return pd.DataFrame()

    steps_lst = []
    walking_lst = []
    # 1 - n
    for i, (src, dst) in enumerate(
        zip(stations.iloc[[0] * (len(stations) - 1)].itertuples(), 
            stations.iloc[1:].itertuples())):
        routes, steps, walking_steps = get_subway_routes(src, dst, strategy, citycode, memo=memo)
        # if routes.empty:
            
        # FIXME 判断 direct 的情况，决定当前节点的状态
        steps_lst.append(steps.iloc[[0]])
        walking_lst.append(walking_steps)
        time.sleep(.1)

    # first seg
    src, dst = stations.iloc[1], stations.iloc[-1]
    routes, steps, walking_steps = get_subway_routes(src, dst, strategy, citycode, memo=memo)
    walking_lst.append(walking_steps)
    steps_lst.append(steps.iloc[[0]])

    df_links = _extract_segment_info_from_routes(steps_lst)
    
    seg_first = df_links[['src', 'src_name']].values
    seg_last = df_links.iloc[[-1]][['dst', 'dst_name']].values
    df_nodes = pd.DataFrame(np.concatenate([seg_first, seg_last], axis=0), columns=['nid', 'name'])
    assert (df_nodes.name.values == stations.name.values).all(), "check the sequence of station"
    df_nodes = df_nodes.merge(stations.rename(columns={'id': 'bvid'}), on='name').set_index('nid')
    
    logger.info(f"stations: {stations['name'].tolist()}")
    
    return df_links, df_nodes


if __name__ == "__main__":
    line_fn = DATA_FOLDER / 'wgs/shenzhen_subway_lines_wgs.geojson'
    ckpt = DATA_FOLDER / 'shenzhen_network.ckpt'
    ckpt = None
    metro = MetroNetwork(line_fn=line_fn, ckpt=ckpt)

    self = metro
    G = metro.graph
    df_lines = metro.df_lines
    df_stations = metro.df_stations

    
    # nodes, edges
    nodes = metro.nodes_to_dataframe()
    edges = metro.edges_to_dataframe()

#%%
lines_iter = iter(df_lines.iterrows())

while True:
    i, line = next(lines_iter)
    if line.line_name == "地铁2号线":
        break

# for i, line in df_lines.iterrows():
# for i, line in df_lines.iterrows():
#     if line.line_name == "地铁11号线":
#         break

#%%
#! 地铁2号线, 地铁7号线(赤尾 出现两次, 因为 `福邻` 地铁站暂未开通)
# 存在环线：地铁5号线, 
i, line = next(lines_iter)

logger.info(f"{line['name']}")
line_id = line['line_id']
line_name = line['line_name']

stations = query_dataframe(self.df_stations, 'line_id', line_id)
_edges, _nodes = get_subway_segment_info(stations, strategy=2, memo=DIRECTION_MEMO)
# metro.add_line(line_id)
# metro.save_ckpt()
save_checkpoint(DIRECTION_MEMO, DIRECTION_MEMO_FN)
assert (_edges.cost > 0).all()
_edges

# %%
station_name = "南山"
# station_name = "上梅林"
df_stations.query("name == @station_name").iloc[0].to_dict()

# %%