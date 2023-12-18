#%%
import json
import requests
import numpy as np
import pandas as pd
from copy import deepcopy
from loguru import logger

ROUTE_COLUMNS = ['route', 'seg_id', 'type', 'name', 'line_id', 'departure_stop', 'arrival_stop',  'distance', 'cost']

def __filter_dataframe_columns(df, cols=ROUTE_COLUMNS):
    cols = [i for i in cols if i in list(df)]
    
    return df[cols]

def query_transit_directions(src, dst, city1, city2, key, strategy=0, show_fields='cost,navi', multiexport=1, memo={}):
    """
    高德地图公交路线规划 API 服务地址

    strategy: 
        0: 推荐模式, 综合权重, 同高德APP默认
        1: 最经济模式, 票价最低
        2: 最少换乘模式, 换乘次数少
        3: 最少步行模式, 尽可能减少步行距离
        4: 最舒适模式, 尽可能乘坐空调车
        5: 不乘地铁模式, 不乘坐地铁路线
        6: 地铁图模式, 起终点都是地铁站（地铁图模式下 originpoi 及 destinationpoi 为必填项）
        7: 地铁优先模式, 步行距离不超过4KM
        8: 时间短模式, 方案花费总时间最少
    Ref:
        - https://lbs.amap.com/api/webservice/guide/api/newroute#t9
    """
    if (src, dst, strategy) in memo:
        return memo[(src, dst, strategy)]
    
    url = "https://restapi.amap.com/v5/direction/transit/integrated"
    params = {
        'key': key,
        'origin': src,
        'destination': dst,
        'city1': city1,
        'city2': city2,
        'strategy': strategy,
        'show_fields': show_fields,
        'multiexport': multiexport
    }
    logger.debug(f"{url}?{'&'.join([f'{k}={v}' for k, v in params.items()])}")

    response = requests.get(url, params=params)
    response = json.loads(response.text)
    memo[(src, dst, strategy)] = response

    return response


def parse_transit_directions(data, mode='地铁线路', verbose=False):
    def _extract_steps_from_plan(route, route_id):
        steps = []
        for seg_id, segment in enumerate(route['segments']):
            connector = 'walking_0'
            step = {"seg_id": seg_id, 'mode': ",".join(segment.keys())}
            for key, val in segment.items():
                val = deepcopy(val)
                # bus
                if key == 'bus':
                    connector = 'walking_1'
                    if len(val['buslines']) != 1:
                         # 针对公交场景，存在2条或以上的公交车共线的情况，但就地铁而言不存在此情况
                        modes = np.unique([item['type'] for item in val['buslines']])
                        if len(modes) > 1:
                            logger.warning(f"Check route {route_id} the buslines length, types: {list(modes)}")
                    line = val['buslines'][0]
                    step.update(line)
                # walking
                elif key == 'walking':
                    step[connector] = val
                    step[connector+"_info"] = {
                        "cost": int(val['cost']['duration']), 
                        "distance": int(val['distance'])
                    }
                # taxi
                elif key == 'taxi':
                    step.update(val)
            steps.append(step)                    

        # 删除首尾步行的部分
        steps = pd.DataFrame(steps)
        if steps.iloc[0].type != steps.iloc[0].type:
            steps = steps.iloc[1:]
        if steps.iloc[-1].type != steps.iloc[-1].type:
            steps = steps.iloc[:-1]
        
        return steps

    routes = pd.DataFrame()
    transits = data.get('route', {}).get("transits")
    
    # `transits` is None
    if not transits:
        logger.warning("No tranists records!")
        return routes
    
    lst = []
    for i, transit in enumerate(transits, start=0):
        routes = _extract_steps_from_plan(transit, i)
        if mode is not None and mode not in routes['type'].unique():
            continue
        routes.loc[:, 'route'] = i
        lst.append(routes)
    
    if lst: routes = pd.concat(lst).reset_index(drop=True)

    routes = routes.replace('', np.nan).dropna(axis=1, how='all')
    routes.rename(columns={'id': 'line_id'}, inplace=True)
    routes.loc[:, 'cost'] = routes.cost.apply(
        lambda x: x.get('duration', np.nan) if isinstance(x, dict) else x)

    # TODO 
    if verbose and False:
        _routes = __filter_dataframe_columns(routes, ROUTE_COLUMNS + ['walking_0_info', 'walking_1_info', 'mode'])
        str_routes = []
        for route_id in routes.route.unique():
            route = _routes.query(f"route == {route_id}").copy()
            route.departure_stop = route.departure_stop.apply(lambda x: x['name'])
            route.arrival_stop = route.arrival_stop.apply(lambda x: x['name'])
            # route.drop(columns=['route'], inplace=True)
            str_routes.append(f"Route {route_id}:\n{route}")

        pre_states = ""
        logger.debug(pre_states + "\n" + "\n\n".join(str_routes))

    return routes


if __name__ == "__main__":
    src = pd.Series({
        'id': 'BV10244676',
        'location': '113.923483,22.524037',
        'name': '南山',
        'sequence': '14',
        'line_id': '440300024057',
        'line_name': '地铁11号线',
    })
    dst = pd.Series(
        {'id': 'BV10243815',
        'location': '114.059415,22.570459',
        'name': '上梅林',
        'sequence': '7',
        'line_id': '440300024075',
        'line_name': '地铁4号线'
    })
    citycode = '0755'
    
    data = query_transit_directions(src.location, dst.location, citycode, citycode, '25db7e8486211a33a4fcf5a80c22eaf0')

# %%
df_routes = parse_transit_directions(data)
df_routes

#%%

__filter_dataframe_columns(df_routes)

# %%
