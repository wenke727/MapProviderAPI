#%%
station_name = '南山' # 车公庙 福田
get_exchange_link_info(nodes, station_name)


#%%
" query -> parse -> filter -> result "
"""
#! 重新梳理流程
# TODO
    x 第一段的步行和最后一段的步行可以忽略
    x formatter
    - 校核起点和终点是否一致；
"""

src, dst = '后海', '南光'
# src, dst = '南山', '上梅林'
# src, dst = '左炮台东', '福永'

strategy = 0
mode = "地铁线路"
start = df_stations.query(f"name == '{src}'").iloc[0]
end = df_stations.query(f"name == '{dst}'").iloc[0]

routes, walking_steps = get_routes(start, end, strategy)

walking_steps

# %%

line_id = '440300024057' # 地铁11号线(机场线)(岗厦北--碧头)
line_id = '440300024063' # 地铁1号线(罗宝线)(机场东--罗湖)
# line_id = '440300024064' # 地铁1号线(罗宝线)(罗湖--机场东)
line_id = '440300024055' # 地铁9号线(梅林线)(前湾--文锦)
line_id = '440300024054' # 地铁9号线(梅林线)(文锦--前湾)
stations = query_dataframe(self.df_stations, 'line_id', line_id)

# nodes, edges, directions_res = get_subway_segment_info(stations, strategy=strategy)
# self.add_nodes(nodes)
# self.add_edges(edges)

_edges, _nodes = get_subway_segment_info(stations)
#%%
_edges

# %%
_nodes
 
# %%
metro.add_nodes(_nodes)
metro.nodes_to_dataframe()

# %%
metro.add_edges(_edges)
metro.edges_to_dataframe()