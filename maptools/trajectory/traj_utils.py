from shapely import LineString


def point_gdf_to_linestring(df, geom_col_name):
    """
    Convert GeoDataFrame of Points to shapely LineString
    """
    if len(df) > 1:
        return LineString(df[geom_col_name].tolist())
    else:
        raise RuntimeError("DataFrame needs at least two points to make line!")
