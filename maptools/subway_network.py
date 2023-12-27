#%%
import time
import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
from loguru import logger

from cfg import KEY, DATA_FOLDER, ROUTE_COLUMNS, EXP_FOLDER
from geo.network import Network
from provider.direction import get_subway_routes
from utils.logger import make_logger
from utils.dataframe import query_dataframe
from utils.dataframe import filter_dataframe_columns
from utils.serialization import load_checkpoint, save_checkpoint

DIRECTION_MEMO = {}
DIRECTION_MEMO_FN = DATA_FOLDER / "direction_memo.pkl"
DIRECTION_MEMO = load_checkpoint(DIRECTION_MEMO_FN)

UNAVAILABEL_STATIONS = set([])

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


class MetroNetwork(Network):
    def __init__(self, line_fn, ckpt=None, refresh=True):
        super().__init__(ckpt=ckpt)  # 调用基类的构造函数
        self.visited_lines = set([])

        # lines
        self.df_lines = _load_subway_lines(line_fn)
        self.df_stations = _extract_stations_from_lines(self.df_lines)

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

    def get_subway_interval(self, ):
        # TODO 获取地铁的发车时间间隔
        pass

    def add_all_lines(self, lines_iter):
        #! 地铁2号线, 地铁7号线(赤尾 出现两次, 因为 `福邻` 地铁站暂未开通)
        # 存在环线：地铁5号线, 
        df_routes_lst = []
        df_steps_lst = []
        df_edges_lst = []
        df_nodes_lst = []

        def _add_line_name(df, name):
            atts = list(df)
            df = df.assign(name=name)
            
            return df[['name']+atts]

        for i, line in (lines_iter):
            # i, line = next(lines_iter)
            logger.info(f"{line['name']}")
            line_id = line['line_id']
            line_name = line['line_name']

            stations = query_dataframe(self.df_stations, 'line_id', line_id)
            df_routes, df_steps, _edges, _nodes = get_subway_segment_info(stations, strategy=2, memo=DIRECTION_MEMO, sleep_dt=0)
            # save_checkpoint(DIRECTION_MEMO, DIRECTION_MEMO_FN)
            assert (_edges.cost > 0).all()
            
            df_routes_lst.append(_add_line_name(df_routes, line['name']))
            df_edges_lst.append(_add_line_name(_edges, line['name']))
            df_nodes_lst.append(_nodes)
            # df_steps_lst.append(add_line_name(df_steps.reset_index(drop=True), line['name']))
            (df_steps.reset_index()).to_excel(EXP_FOLDER / f"{line['name']}_steps.xlsx")

        df_routes = pd.concat(df_routes_lst)
        df_edges = pd.concat(df_edges_lst)
        df_nodes = pd.concat(df_nodes_lst)

        df_routes.to_excel(EXP_FOLDER / f"routes.xlsx")
        df_edges.to_excel(EXP_FOLDER / f"edges.xlsx")
            

        self.add_nodes(df_nodes)
        self.add_edges(df_edges)


""" 待开发 & 开发 """
def check_shortest_path():
    src = "440300024057010" # 11号线，南山
    dst = "440300024057013" # 11号线，福田

    routes = metro.top_k_paths(src, dst, 3, weight='cost')
    nodes.loc[routes[0]]

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

def get_subway_segment_info(stations, strategy=0, citycode='0755', memo={}, sleep_dt=.2):
    global UNAVAILABEL_STATIONS
    if stations.empty or 'name' not in stations.columns:
        logger.error("Invalid stations data")
        return pd.DataFrame()

    routes_lst = []
    steps_lst = []
    walking_lst = []
    unavailabel_stops = []
    def __helper(src, dst):
        routes, steps, walking_steps = get_subway_routes(src, dst, strategy, citycode, memo=memo)
        routes.query("stop_check == True", inplace=True)
        if sleep_dt: time.sleep(sleep_dt)
        
        if routes.empty:
            # TODO 增加线路的信息，仅仅靠名字是无法判断的
            logger.warning(f"unavailabel stop: {dst['name']}")
            unavailabel_stops.append(dst['name'])
            return
        
        idx = routes.index.values[0]
        routes_lst.append(routes)
        steps_lst.append(steps.loc[[idx]])
        walking_lst.append(walking_steps)    

    # 1 - n
    src = stations.iloc[0]
    for i in range(1, len(stations)):
        dst = stations.iloc[i]
        __helper(src, dst)

    # first seg
    __helper(stations.iloc[1], stations.iloc[-1])

    # stations
    if len(unavailabel_stops):
        stations.query("name not in @unavailabel_stops", inplace=True)

    df_links = _extract_segment_info_from_routes(steps_lst)
    seg_first = df_links[['src', 'src_name']].values
    seg_last = df_links.iloc[[-1]][['dst', 'dst_name']].values
    df_nodes = pd.DataFrame(np.concatenate([seg_first, seg_last], axis=0), columns=['nid', 'name'])
    if not (df_nodes.name.values == stations.name.values).all():
        logger.warning("check the sequence of station")
    df_nodes = df_nodes.merge(stations.rename(columns={'id': 'bvid'}), on='name').set_index('nid')
    
    df_routes = pd.concat(routes_lst)
    df_steps = pd.concat(steps_lst)
    logger.debug(f"stations: {stations['name'].tolist()}")
    
    return df_routes, df_steps, df_links, df_nodes


if __name__ == "__main__":
    line_fn = DATA_FOLDER / 'wgs/shenzhen_subway_lines_wgs.geojson'
    ckpt = DATA_FOLDER / 'shenzhen_network.ckpt'
    ckpt = None
    metro = MetroNetwork(line_fn=line_fn, ckpt=ckpt)

    self = metro
    G = metro.graph
    df_lines = metro.df_lines
    df_stations = metro.df_stations

    
    lines_iter = iter(df_lines.iterrows())
    metro.add_all_lines(lines_iter)

    # nodes, edges
    nodes = metro.nodes_to_dataframe()
    edges = metro.edges_to_dataframe()


# %%
def get_exchange_link_info(nodes, station_name):
    #! 如何获取一个换乘站的所有连接
    station_nodes= query_dataframe(nodes, 'name', station_name)
    nidxs = np.sort(station_nodes.index)
    logger.debug(f"{station_name}: {nidxs}")

    lst = []
    for nid in nidxs:
        lst.append(metro.get_adj_nodes(nid))
    points = pd.DataFrame(lst)
    points.loc[:, 'prev'] = points['prev'].apply(lambda x: x[0] if isinstance(x, list) else x)
    points.loc[:, 'next'] = points['next'].apply(lambda x: x[0] if isinstance(x, list) else x)
    points.loc[:, 'line_name'] = points['cur'].apply(lambda x: nodes.loc[x].line_name)

    ods = []
    connectors_lst = []
    for i, src in enumerate(points.itertuples()):
        for j, dst in enumerate(points.itertuples()):
            if i == j or src.line_name == dst.line_name:
                continue

            start = nodes.loc[src.prev]
            end = nodes.loc[dst.next]
            ods.append([start, end])
            _, _, connectors = get_subway_routes(start, end, strategy=0, citycode='0755', memo=DIRECTION_MEMO)
            if connectors.empty:
                continue
            connectors.query('src_id in @nidxs and dst_id in @nidxs', inplace=True)
            connectors_lst.append(connectors)
            _len = len(connectors)
            level='debug'
            if _len == 0:
                level = 'warning'
            elif _len > 1:
                level = 'info'
            getattr(logger, level)(f"{station_name} 连接线: {start['name']} -> {end['name']} : {_len}")
            time.sleep(.1)
            
    return pd.DataFrame(ods, columns=['src', 'dst']), pd.concat(connectors_lst)

station_name = "上梅林"
_df, df_connectors = get_exchange_link_info(nodes, station_name)
df_connectors
# routes, steps, walkings = get_subway_routes(_df.iloc[0].src, _df.iloc[0].dst, strategy=0, citycode='0755', memo=DIRECTION_MEMO)
# walkings

# %%
df_connectors = df_connectors.drop_duplicates(['src_id', 'dst_id']).reset_index(drop=True)

df_connectors.src_id.value_counts()
# %%
