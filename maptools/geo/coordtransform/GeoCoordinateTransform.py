import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import transform
from coordtransform import gcj_to_wgs
from coordTransform_py import wgs84_to_gcj02 as wgs_to_gcj

def convert_point(point, convert_func):
    return Point(convert_func(point.x, point.y))

def convert_line(line, convert_func):
    return LineString([convert_point(pt, convert_func) for pt in line.coords])

def convert_polygon(polygon, convert_func):
    exterior = convert_line(polygon.exterior, convert_func)
    interiors = [convert_line(interior, convert_func) for interior in polygon.interiors]
    return Polygon(exterior, interiors)

def transform_geometry(geometry, convert_func):
    if isinstance(geometry, Point):
        return convert_point(geometry, convert_func)
    elif isinstance(geometry, LineString):
        return convert_line(geometry, convert_func)
    elif isinstance(geometry, Polygon):
        return convert_polygon(geometry, convert_func)
    else:
        raise ValueError("不支持的几何类型")

def coordinate_converter(source_crs, target_crs):
    """
    返回一个转换函数，该函数根据源坐标系和目标坐标系进行转换。
    您需要根据您的具体转换逻辑来实现这个函数。
    """
    # 示例：假设您有 gcj_to_wgs 和 wgs_to_gcj 等函数
    if source_crs == 'GCJ-02' and target_crs == 'WGS-84':
        return gcj_to_wgs
    elif source_crs == 'WGS-84' and target_crs == 'GCJ-02':
        return wgs_to_gcj
    # 添加更多的条件以支持不同的转换
    else:
        raise ValueError("不支持的坐标系转换")

def convert_geodf_coordinates(geodf, source_crs, target_crs):
    convert_func = coordinate_converter(source_crs, target_crs)
    geodf['geometry'] = geodf['geometry'].apply(lambda geom: transform_geometry(geom, convert_func))
    return geodf
