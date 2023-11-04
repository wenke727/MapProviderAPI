import shapefile as shp
# import data_process.settings as settings
import CoordTransform_utils
# settings.PORT_POLY_FILE


def load_poly_from_shp(shp_file='e:/Yantian Port/GIS_data/YantianPort_wgs.shp', in_cood='wgs', out_coord='gcj'):
    sf = shp.Reader(shp_file)
    polys = sf.shapes()
    # for there is only one polyline in the shapfile
    points = polys[0].points
    points = [[i[0], i[1]] for i in points]

    if out_coord == 'gcj':
        points = [CoordTransform_utils.wgs84_to_gcj02(x[0], x[1]) for x in points]
        points = [CoordTransform_utils.wgs84_to_gcj02(x[0], x[1]) for x in points]
    return points


def isPointinPolygon(point, rangelist):  # [[0,0],[1,1],[0,1],[0,0]] [1,0.8]
    # 判断是否在外包矩形内，如果不在，直接返回false
    lnglist = []
    latlist = []
    for i in range(len(rangelist)-1):
        lnglist.append(rangelist[i][0])
        latlist.append(rangelist[i][1])
    # print(lnglist, latlist)
    maxlng = max(lnglist)
    minlng = min(lnglist)
    maxlat = max(latlist)
    minlat = min(latlist)
    # print(maxlng, minlng, maxlat, minlat)
    if (point[0] > maxlng or point[0] < minlng or
            point[1] > maxlat or point[1] < minlat):
        return False

    count = 0
    point1 = rangelist[0]
    for i in range(1, len(rangelist)):
        point2 = rangelist[i]
        # 点与多边形顶点重合
        if (point[0] == point1[0] and point[1] == point1[1]) or (point[0] == point2[0] and point[1] == point2[1]):
            # print("在顶点上")
            return False
        # 判断线段两端点是否在射线两侧 不在肯定不相交 射线（-∞，lat）（lng,lat）
        if (point1[1] < point[1] and point2[1] >= point[1]) or (point1[1] >= point[1] and point2[1] < point[1]):
            # 求线段与射线交点 再和lat比较
            point12lng = point2[0] - (point2[1] - point[1]) * \
                (point2[0] - point1[0])/(point2[1] - point1[1])
            # print(point12lng)
            # 点在多边形边上
            if (point12lng == point[0]):
                # print("点在多边形边上")
                return False
            if (point12lng < point[0]):
                count += 1
        point1 = point2
    # print(count)
    if count % 2 == 0:
        return False
    else:
        return True
 
def isTrajectoryWithinPoly( pois, poly, pencentage_threshold = 0.6 ):
    '''
    input: 
        pois = [ [x1,y1], ... , [xn,yn] ]
        poly = [[x1,y1],[x2,y2],……,[xn,yn],[x1,y1]],[[w1,t1],……[wk,tk]] 
    通过 点在区域内的数量/总数，判断是否在港口内部行动
    '''
    num_poi_within_poly = 0
    for poi in pois:
        if isPointinPolygon( poi, poly):
            num_poi_within_poly +=1
    return True if num_poi_within_poly/len(pois) > pencentage_threshold else False


if __name__ == '__main__':
    area = load_poly_from_shp()
    isPointinPolygon([114.27905833333334, 22.574086666666666], area)
    pass
