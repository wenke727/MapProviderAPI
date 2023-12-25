import numpy as np
import pandas as pd
import geopandas as gpd

from geo.distance import cal_pointwise_distance_geoseries

def traj_clean_drift(data, col=['VehicleNum', 'Time', 'Lng', 'Lat'],
                     method='twoside',
                     speedlimit=80,
                     dislimit=1000,
                     anglelimit=30):
    '''
    Delete the drift in the trajectory data. The drift is defined as the data with a speed greater than the speed limit or the distance between the current point and the next point is greater than the distance limit or the angle between the current point, the previous point, and the next point is smaller than the angle limit. The speed limit is 80km/h by default, and the distance limit is 1000m by default. The method of cleaning drift data is divided into two methods: ‘oneside’ and ‘twoside’. The ‘oneside’ method is to consider the speed of the current point and the next point, and the ‘twoside’ method is to consider the speed of the current point, the previous point, and the next point.

    Parameters
    -------
    data : DataFrame
        Data
    col : List
        Column names, in the order of [‘VehicleNum’, ‘Time’, ‘Lng’, ‘Lat’]
    method : string
        Method of cleaning drift data, including ‘oneside’ and ‘twoside’
    speedlimit : number
        Speed limitation
    dislimit : number
        Distance limit
    anglelimit : number
        Angle limit

    Returns
    -------
    data1 : DataFrame
        Cleaned data
    '''
    [VehicleNum, Time, Lng, Lat] = col
    data1 = data.copy()
    data1 = data1.drop_duplicates(subset=[VehicleNum, Time])
    data1[Time+'_dt'] = pd.to_datetime(data1[Time])
    data1 = data1.sort_values(by=[VehicleNum, Time])

    #计算前后点距离、时间差、速度
    for i in [VehicleNum, Lng, Lat, Time+'_dt']:
        data1[i+'_pre'] = data1[i].shift()
        data1[i+'_next'] = data1[i].shift(-1)

    data1['dis_pre'] = getdistance(
        data1[Lng],
        data1[Lat],
        data1[Lng+'_pre'],
        data1[Lat+'_pre'])
    data1['dis_next'] = getdistance(
        data1[Lng],
        data1[Lat],
        data1[Lng+'_next'],
        data1[Lat+'_next'])
    data1['dis_prenext'] = getdistance(
        data1[Lng+'_pre'],
        data1[Lat+'_pre'],
        data1[Lng+'_next'],
        data1[Lat+'_next'])
    
    #计算前后点时间差
    data1['timegap_pre'] = data1[Time+'_dt'] - data1[Time+'_dt_pre']
    data1['timegap_next'] = data1[Time+'_dt_next'] - data1[Time+'_dt']
    data1['timegap_prenext'] = data1[Time+'_dt_next'] - data1[Time+'_dt_pre']

    #计算前后点速度
    data1['speed_pre'] = data1['dis_pre'] /  data1['timegap_pre'].dt.total_seconds()*3.6
    data1['speed_next'] = data1['dis_next'] / data1['timegap_next'].dt.total_seconds()*3.6
    data1['speed_prenext'] = data1['dis_prenext'] / data1['timegap_prenext'].dt.total_seconds()*3.6

    #余弦定理计算夹角
    angle_cos = (data1['dis_pre']**2+data1['dis_next']**2-data1['dis_prenext']**2)/(2*data1['dis_pre']*data1['dis_next'])
    angle_cos = np.maximum(np.minimum(angle_cos, 1), -1)
    data1['angle'] = np.degrees(np.arccos(angle_cos))

    #以速度限制删除异常点
    if speedlimit:
        if method == 'oneside':
            data1 = data1[
                -((data1[VehicleNum+'_pre'] == data1[VehicleNum]) &
                  (data1['speed_pre'] > speedlimit))]
        elif method == 'twoside':
            data1 = data1[
                -((data1[VehicleNum+'_pre'] == data1[VehicleNum]) &
                  (data1[VehicleNum+'_next'] == data1[VehicleNum]) &
                    (data1['speed_pre'] > speedlimit) &
                    (data1['speed_next'] > speedlimit) &
                    (data1['speed_prenext'] < speedlimit))]
    #以距离限制删除异常点
    if dislimit:
        if method == 'oneside':
            data1 = data1[
                -((data1[VehicleNum+'_pre'] == data1[VehicleNum]) &
                  (data1['dis_pre'] > dislimit))]
        elif method == 'twoside':
            data1 = data1[
                -((data1[VehicleNum+'_pre'] == data1[VehicleNum]) &
                  (data1[VehicleNum+'_next'] == data1[VehicleNum]) &
                    (data1['dis_pre'] > dislimit) &
                    (data1['dis_next'] > dislimit) &
                    (data1['dis_prenext'] < dislimit))]
    #以角度限制删除异常点
    if anglelimit:
        data1 = data1[
            -((data1[VehicleNum+'_pre'] == data1[VehicleNum]) &
              (data1[VehicleNum+'_next'] == data1[VehicleNum]) &
                (data1['angle'] < anglelimit))]
    data1 = data1[data.columns]
    return data1

