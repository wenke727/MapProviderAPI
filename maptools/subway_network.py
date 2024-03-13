#%%
import time
import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
from loguru import logger
import shapely
from shapely.geometry import MultiPoint, LineString
from collections import defaultdict

from cfg import KEY, DATA_FOLDER, ROUTE_COLUMNS, EXP_FOLDER
from geo.network import Network
from geo.coords_utils import str_to_point
from provider.direction import get_subway_routes
from utils.logger import make_logger
from utils.dataframe import query_dataframe
from utils.dataframe import filter_dataframe_columns
from utils.serialization import load_checkpoint, save_checkpoint, to_geojson

DIRECTION_MEMO = {}
DIRECTION_MEMO_FN = DATA_FOLDER / "direction_memo.pkl"
DIRECTION_MEMO = load_checkpoint(DIRECTION_MEMO_FN)

CITYCODE = '0755'
UNAVAILABEL_STATIONS = set([])

# 默认换乘距离，即：起点和终点经纬度一致的时候
DEFAULT_TRANSFER_DISTANCE = 100

logger = make_logger(DATA_FOLDER, 'network', include_timestamp=False)

#%%

""" 辅助函数 """
def _split_subway_line_name(series):
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
        df_lines.status = df_lines.status.astype(np.int64)
        unavailabel_line = df_lines.query("status != 1").name.unique().tolist()
        logger.warning(f"Unavailabel lines: {unavailabel_line}")
        df_lines.query('status == 1', inplace=True)

    names = _split_subway_line_name(df_lines.name)
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

def split_linestring_by_stations(line_id, df_lines, df_stations):
    """split subway linestring into segments."""
    line = df_lines.query(f"line_id == '{line_id}'")
    pts = df_stations.query(f"line_id == '{line_id}'")
    station_names = pts['name'].values
    pts = gpd.GeoDataFrame(pts, geometry=pts.location.apply(lambda x: str_to_point(x, 'wgs')), crs=4326)

    geoms = shapely.ops.split(
        line.iloc[0]['geometry'],
        shapely.MultiPoint(pts.geometry.values)
    )
    geoms = list(geoms.geoms)
    assert len(geoms) == len(pts) - 1
    
    segs = gpd.GeoDataFrame(
        {'line_id': line_id, 'src_name': station_names[:-1], 'dst_name': station_names[1:]}, 
        geometry=geoms, crs=4326)
    
    return segs

def _extract_segment_info_from_routes(routes_lst:list):
    df_segs = pd.concat(routes_lst)
    df_segs['cost'] = df_segs['cost'].astype(np.int64)
    df_segs['distance'] = df_segs['distance'].astype(np.int64)

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

def get_subway_segment_info(stations, strategy=0, citycode=CITYCODE, sleep_dt=.2, auto_save=True):
    if stations.empty or 'name' not in stations.columns:
        logger.error("Invalid stations data")
        return pd.DataFrame()

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
    src = stations.iloc[0]
    for i in range(1, len(stations)):
        # BUG 环线
        _query_helper(src, stations.iloc[i])
    
    # segs: 0
    _query_helper(stations.iloc[1], stations.iloc[-1])
    if auto_save:
        save_checkpoint(DIRECTION_MEMO, DIRECTION_MEMO_FN)

    # stations
    if len(unavailabel_stops):
        stations.query("name not in @unavailabel_stops", inplace=True)

    df_segs = _extract_segment_info_from_routes(steps_lst)
    seg_first = df_segs[['src', 'src_name']].values
    seg_last = df_segs.iloc[[-1]][['dst', 'dst_name']].values
    df_nodes = pd.DataFrame(np.concatenate([seg_first, seg_last], axis=0), columns=['nid', 'name'])
    assert (df_nodes.name.values == stations.name.values).all(), logger.warning("check the sequence of station")
    df_nodes = df_nodes.merge(stations.rename(columns={'id': 'bvid'}), on='name').set_index('nid')
    
    df_routes = pd.concat(routes_lst)
    df_steps = pd.concat(steps_lst)
    
    return df_routes, df_steps, df_segs, df_nodes

def get_exchange_link_info(nodes, station_name):
    # 获取一个换乘站的所有连接
    def logger_helper(start, end, connectors):
        if connectors.empty:
            logger.warning(f"{start['name']}({start['line_name']}) -> {end['name']}({start['line_name']}): empty")
            return
            
        connectors_lst.append(connectors)
        _len = len(connectors)
        level='debug'
        if _len == 0:
            level = 'warning'
        elif _len > 1:
            level = 'info'
        if level == "debug":
            return
        getattr(logger, level)(f"{station_name} 连接线: {start['name']} -> {end['name']} : {_len}")
        logger.debug(f"\n{connectors}")        

    def get_nid(x):
        if isinstance(x, list):
            if len(x) == 0:
                return None
            return x[0]
        else:
            return x
        
    station_nodes = query_dataframe(nodes, 'name', station_name)
    nidxs = np.sort(station_nodes.index)
    logger.debug(f"{station_name}: {nidxs}")

    points = pd.DataFrame([metro.get_adj_nodes(nid) for nid in nidxs])
    points.loc[:, 'prev'] = points['prev'].apply(get_nid)
    points.loc[:, 'next'] = points['next'].apply(get_nid)
    points.loc[:, 'line_name'] = points['cur'].apply(lambda x: nodes.loc[x].line_name)

    connectors_lst = []
    coord_pair_2_ods = defaultdict(set)
    for i, src in enumerate(points.itertuples()):
        for j, dst in enumerate(points.itertuples()):
            if i == j or src.line_name == dst.line_name:
                continue
            # if nodes.loc[src.cur].location == nodes.loc[dst.cur].location:
            #     continue
            
            start, end = nodes.loc[src.cur], nodes.loc[dst.cur]
            coord_pair_2_ods[(start.location, end.location)].add((src.cur, dst.cur, src, dst))
            #TODO 可以测试数值是否一致

    logger.info(f"coord_pair_2_ods: {coord_pair_2_ods.keys()}")
    for key, values in coord_pair_2_ods.items():
        src_loc, dst_loc = key
        for item in values:
            src_id, dst_id, src, dst = item
            nidxs = [src_id, dst_id]
            if src.prev is None or dst.next is None:
                continue
            
            start, end = nodes.loc[src.prev], nodes.loc[dst.next]
            _, _, connectors = get_subway_routes(
                start, end, strategy=0, citycode=CITYCODE, memo=DIRECTION_MEMO)
            
            if not connectors.empty:
                connectors.query('src in @nidxs and dst in @nidxs', inplace=True)
                attrs = ['src', 'dst']
                for att in ['cost', 'distance']:
                    if att in connectors.columns:
                        attrs.append(att)
                connectors.drop_duplicates(attrs, keep='first', inplace=True)
                if connectors.shape[0] > 1:
                    logger.info(f"\n{connectors}")
                # assert connectors.shape[0] == 1, "check the connetors."
            if not connectors.empty:
                logger_helper(start, end, connectors)
                for src_id, dst_id, _, _ in values:
                    _connectors = connectors.copy()
                    _connectors.src = src_id
                    _connectors.dst = dst_id
                    connectors_lst.append(_connectors)
                break
        else:
            logger.warning(f"({src_loc}, {dst_loc}) don't have connectings")    
    connecters = pd.concat(connectors_lst)\
                   .drop_duplicates(['src', 'dst'])\
                   .reset_index(drop=True) if len(connectors_lst) else pd.DataFrame()
    
    return connecters

def get_edges(df_lines, df_stations, edges, nodes, keys=['src', 'dst']):
    name_2_idx = nodes.reset_index().set_index(['line_id', 'name'])['index']
    available_srarion_mask = df_stations.apply(lambda x: (x['line_id'], x['name']) not in UNAVAILABEL_STATIONS, axis=1)
    
    seg_geoms = []
    for line_id in df_lines.line_id.values:
        segs = split_linestring_by_stations(line_id, df_lines, df_stations[available_srarion_mask])
        segs.loc[:, 'src'] = segs.apply(lambda x: name_2_idx.get((x.line_id, x.src_name)), axis=1)
        segs.loc[:, 'dst'] = segs.apply(lambda x: name_2_idx.get((x.line_id, x.dst_name)), axis=1)
        segs = gpd.GeoDataFrame(segs).set_geometry('geometry')
        seg_geoms.append(segs)
        
    seg_geoms = gpd.GeoDataFrame(pd.concat(seg_geoms), crs=4326)

    # add `distance` and `duration`
    seg_geoms.sort_values(keys, inplace=True)
    edges.sort_values(keys, inplace=True)
    logger.info(seg_geoms.keys())
    assert np.allclose(seg_geoms[keys].astype(np.int64), edges[keys].astype(np.int64))
    seg_geoms = seg_geoms.merge(edges, on=keys)

    seg_geoms.set_geometry('geometry', inplace=True)
    seg_geoms.set_index(keys, inplace=True)

    return seg_geoms

def get_exchange_links(metro):
    # 获取换乘站点的连接信息
    exchanges = metro.get_exchange_stations()
    special_exchanges = set(['深圳北站', '福民', '福永', '机场东', '少年宫', '市民中心'])
    nodes = metro.nodes_to_dataframe()
    
    df_connectors_lst = []
    for station_name in exchanges.name.values:
        df_connectors = get_exchange_link_info(nodes, station_name)
        df_connectors_lst.append(df_connectors)

    df_connectors = pd.concat(df_connectors_lst)
    df_connectors = df_connectors.fillna(0).rename(
        columns={'station_name': 'src_name', 'cost': 'walking_duration'})
    df_connectors.drop(columns=['route', 'src_loc', 'dst_loc', 'same_loc'], inplace=True)

    df_connectors = df_connectors.assign(
        dst_name = 'exchange',
        line_id = df_connectors.dst.apply(lambda x: x[:-3]),
        geometry = LineString()
    )

    # add `waiting time` for transfers
    df_connectors.loc[:, 'duration'] = df_connectors.apply(
        lambda x: x.walking_duration + metro.lineid_2_waiting_time[x.line_id], axis=1)

    df_connectors.loc[df_connectors['distance'] <= 0, 'distance'] = DEFAULT_TRANSFER_DISTANCE

    return gpd.GeoDataFrame(df_connectors, crs=4326)

def get_station_inner_links(nodes, lineid_2_waiting_time):
    inner_station_links = nodes.groupby(['line_name', 'name'])\
                            .agg({'nid': list}).reset_index()\
                            .rename(columns={'name': 'src_name'})

    # 检查所有 'nid' 条目的长度是否为 2
    assert all(len(nid) == 2 for nid in inner_station_links['nid'])

    # 将 'nid' 拆分为 'src' 和 'dst' 两列
    inner_station_links[['src', 'dst']] = pd.DataFrame(inner_station_links['nid'].tolist(), index=inner_station_links.index)
    inner_station_links.drop(columns=['line_name', 'nid'], inplace=True)

    # add opposite
    _revert = inner_station_links.copy()
    _revert.src, _revert.dst = _revert.dst, _revert.src
    inner_station_links = pd.concat([inner_station_links, _revert])

    # line_id
    inner_station_links = inner_station_links.assign(
        line_id = inner_station_links.dst.apply(lambda x: x[:-3]),
    )
    
    inner_station_links = inner_station_links.assign(
        dst_name = 'inner_link',
        distance = DEFAULT_TRANSFER_DISTANCE,
        duration = inner_station_links.line_id.apply(lambda x: lineid_2_waiting_time[x]),
        geometry = LineString() # FIXME 可能存在不一样的坐标
    )    

    return gpd.GeoDataFrame(inner_station_links, crs=4326)


class MetroNetwork(Network):
    def __init__(self, line_fn, ckpt=None, refresh=True):
        super().__init__(ckpt=ckpt)  # 调用基类的构造函数
        self.visited_lines = set([])

        # lines
        self.df_lines = _load_subway_lines(line_fn)
        self.df_stations = _extract_stations_from_lines(self.df_lines)
        self.lineid_2_waiting_time = {}

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

    def add_line(self, line:pd.Series, sleep_dt=0.2, debug=False):
        def _add_line_name(df, name):
            atts = list(df)
            df = df.assign(name=name)
            
            return df[['name']+atts]
        
        def judge_ring(stations):
            return stations.iloc[0].id == stations.iloc[-1].id
        
        logger.info(f"{line['name']}")
        line_id = line['line_id']
        line_name = line['line_name']

        stations = query_dataframe(self.df_stations, 'line_id', line_id)
        is_ring = judge_ring(stations)
        if not is_ring:
            df_routes, df_steps, _edges, _nodes = get_subway_segment_info(stations, strategy=2, sleep_dt=sleep_dt)
        else:
            # TODO
            raise NotImplementedError
        
        # 获取发车时间间隔
        time_gap = np.unique(df_steps.cost.astype(np.int64).values[:len(_edges)] - _edges.cost.astype(np.int64).values.cumsum())
        assert len(time_gap) == 1, "可能存在大小区间"
        self.lineid_2_waiting_time[line_id] = time_gap[0]
        
        assert (_edges.cost > 0).all()
        
        df_routes = _add_line_name(df_routes, line['name'])
        _edges = _add_line_name(_edges, line['name'])
        if debug:
            df_steps.reset_index().to_excel(EXP_FOLDER / f"{line['name']}_steps.xlsx")
            
        return df_routes, _edges, _nodes
            
    def add_all_lines(self, sleep_dt=0.2, debug=False):
        #! 地铁2号线, 地铁7号线(赤尾 出现两次, 因为 `福邻` 地铁站暂未开通)
        # 存在环线：地铁5号线, 识别尚未开通的车站
        df_routes_lst = []
        df_edges_lst = []
        df_nodes_lst = []


        for i, line in self.df_lines.iterrows():
            df_routes, _edges, _nodes = self.add_line(line, sleep_dt)
            df_routes_lst.append(df_routes)
            df_edges_lst.append(_edges)
            df_nodes_lst.append(_nodes)
            
        df_routes = pd.concat(df_routes_lst)
        df_edges = pd.concat(df_edges_lst)
        df_nodes = pd.concat(df_nodes_lst)

        if debug:
            df_routes.to_excel(EXP_FOLDER / f"routes.xlsx")
            df_edges.to_excel(EXP_FOLDER / f"edges.xlsx")
                
        self.add_nodes(df_nodes)
        self.add_edges(df_edges, length='distance', duration='cost')
        
        self.nodes = self.nodes_to_dataframe()
        self.edges = self.edges_to_dataframe()
        
        # 轨道边
        seg_geoms = get_edges(self.df_lines, self.df_stations, self.edges, self.nodes, keys=['src', 'dst'])
        # 换乘边
        df_connectors = get_exchange_links(self)
        df_inner_connetors = get_station_inner_links(self.nodes, self.lineid_2_waiting_time)

        self.edges_and_links = pd.concat(
            [seg_geoms, df_connectors.set_index(['src', 'dst']), df_inner_connetors.set_index(['src', 'dst'])], 
            axis=0)
        # self.edges_and_links = gpd.GeoDataFrame(self.edges_and_links, crs=4326)
        
        return self.edges_and_links

    def adapt_graph_to_GeoDigraph(self):
        df_nodes = self.nodes_to_dataframe()
        df_nodes.index = df_nodes.index.astype(np.int64)
        df_nodes = df_nodes.assign(
            nid = df_nodes.index,
            geometry = df_nodes.location.apply(lambda x:str_to_point(x, 'wgs'))
        )
        df_nodes.drop(columns=['location'], inplace=True)
        df_nodes = df_nodes[['name', 'nid', 'bvid', 'line_name', 'line_id', 'sequence', 'geometry']]
        df_nodes = gpd.GeoDataFrame(df_nodes, crs=4326)

        df_edges = self.edges_and_links.copy()
        df_edges.rename(columns={'line_id': 'way_id'}, inplace=True) # , 'cost': 'walking', 'duration': 'cost'
        df_edges.reset_index(inplace=True)
        df_edges[['src', 'dst']] = df_edges[['src', 'dst']].astype(np.int64)
        df_edges = df_edges.reset_index().rename(columns={'index': 'eid'})
        df_edges = df_edges.assign(
            dir = 0,
            speed = df_edges['distance'] / df_edges['duration']
        )
        df_edges = gpd.GeoDataFrame(df_edges, crs=4326)
        
        return df_nodes, df_edges


#%%
if __name__ == "__main__":
    city = "beijing"
    line_fn = DATA_FOLDER / f'wgs/{city}_subway_lines_wgs.geojson'
    ckpt = DATA_FOLDER / f'{city}_network.ckpt'
    # ckpt = None

    metro = MetroNetwork(line_fn=line_fn, ckpt=ckpt)
    line_ring = metro.df_lines.loc[6]
    metro.add_line(line_ring)
    
    # metro.add_all_lines(sleep_dt=0)
    save_checkpoint(DIRECTION_MEMO, DIRECTION_MEMO_FN)
    
    self = metro
    G = metro.graph
    df_lines = metro.df_lines
    df_stations = metro.df_stations
    
    # nodes, edges
    nodes = metro.nodes_to_dataframe()
    edges = metro.edges_to_dataframe()

    # ? 福民 还有缺失
    # df_connectors = get_exchange_link_info(nodes, '福民')

    # graph 适配    
    df_nodes, df_edges = metro.adapt_graph_to_GeoDigraph()
    

    # save
    to_geojson(df_edges, "../exp/shezhen_subway_edges")
    to_geojson(df_nodes, "../exp/shezhen_subway_nodes")

    # %%
    src = '440300024060032' # 福保（终点）
    dst = '440300024060030'
    dst = '440300024061032' # 福保（起点）
    # dst = '440300024061002' # 益田

    # metro.shortest_path(src, dst)
    routes = metro.top_k_paths(src, dst, 3, weight='cost')
    nodes.loc[routes[0]]


    # %%
    src = np.int64(src)
    df_edges.query("src == @src or dst == @src")

# %%
