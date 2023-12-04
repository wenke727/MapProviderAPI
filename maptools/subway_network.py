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
from loguru import logger

from cfg import KEY

#%%
df_lines = gpd.read_file('../data/subway/shenzhen_subway_lines_wgs.goojson')
df_stations = gpd.read_file('../data/subway/shenzhen_subway_station_wgs.goojson')

#%%
nanshan = df_stations.query("name == '南山'").location
shangmeilin = df_stations.query("name == '上梅林'").location

#%%

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


response_text = get_transit_directions(nanshan, shangmeilin, '0755', '0755')
response_text

# %%

def extract_steps_from_plan(plan):
    steps = []
    seg_id = 0
    for segment in plan['segments']:
        for key, val in segment.items():
            if 'bus' == key:
                for line in val['buslines']:
                    # bus_info = {
                    #     'type': 'Bus',
                    #     'name': line['name'],
                    #     'departure_stop': line['departure_stop']['name'],
                    #     'arrival_stop': line['arrival_stop']['name'],
                    #     'distance': line['distance'],
                    #     'duration': line['cost']['duration']
                    # }
                    line['seg_id'] = seg_id
                    steps.append(line)

            # if 'walking' == key:
            #     walking_info = {
            #         'type': 'Walking',
            #         'origin': val['origin'],
            #         'destination': val['destination'],
            #         'distance': val['distance'],
            #         'duration': val['cost']['duration']
            #     }
            #     val['seg_id'] = seg_id
            #     steps.append(val)
            
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
            logger.debug(route)
            lst = []
            for i, transit in enumerate(route['transits'], start=0):
                distance = transit.get('distance')
                cost = transit.get('cost')
                logger.debug(f"Plan {i}: distance: {distance}, cost: {cost}")
                
                df = extract_steps_from_plan(transit)
                df.loc[:, 'plan'] = i
                lst.append(df)
                
            df = pd.concat(lst)

    df = df.replace('', np.nan).dropna(axis=1, how='all')
    df.loc[:, 'cost'] = df.cost.apply(
        lambda x: x.get('duration', np.nan))

    return df  # 返回解析后的数据

route = parse_transit_directions(response_text)
route

# %%
""" 
初步结论：
1. 候车时间设定比较随意，2 min
"""
line_id = 23
line = df_lines.loc[line_id]
demo_stations = pd.json_normalize(json.loads(line.busstops))
# demo_stations.loc[:, 'sid'] = demo_stations.sequence.apply(
#     lambda x: f"{line.id}{int(x):03d}")
demo_stations

# %%
response_text = get_transit_directions(
    demo_stations.loc[1].location, 
    demo_stations.loc[2].location, 
    '0755', '0755')
route1 = parse_transit_directions(response_text)
route1.iloc[[0]]

# %%
response_text = get_transit_directions(
    demo_stations.loc[2].location, 
    demo_stations.loc[3].location, 
    '0755', '0755')

route2 = parse_transit_directions(response_text)
route2.iloc[[0]]

# %%
response_text = get_transit_directions(
    demo_stations.loc[1].location, 
    demo_stations.loc[3].location, 
    '0755', '0755')

route3 = parse_transit_directions(response_text)
route3.iloc[[0]]

# %%
response_text = get_transit_directions(
    demo_stations.loc[1].location, 
    demo_stations.loc[5].location, 
    '0755', '0755')

route3 = parse_transit_directions(response_text)
route3.iloc[[0]]
# %%
