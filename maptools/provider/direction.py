import json
import requests
from loguru import logger


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

if __name__ == "__main__":
    # query_transit_directions()
    pass