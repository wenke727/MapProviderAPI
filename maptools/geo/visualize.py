import re
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

try:
    from tilemap import providers
    from tilemap import plot_geodata as plotGeodata
    Normal = providers.Amap.Normal
    Vector = providers.Amap.Vector
    Satellite = providers.Amap.Satellite
    FLAG = True
except:
    Satellite, Normal, Vector = [None] * 3
    FLAG = False
    

plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统可用“微软雅黑”或“SimHei”
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

def plot_geodata(gdf, provider=Satellite, *args, **kwargs):
    if isinstance(provider, int):
        provider = {
            0: Satellite,
            1: Vector,
            2: Normal,
        }[provider]
    if FLAG:
        return plotGeodata(gdf, provider=provider, *args, **kwargs)
    
    return None, gdf.plot()

def plot_buffered_zones(gdf: gpd.GeoDataFrame, buffer_sizes: list, title: str, label=None):
    """
    Creates a plot of buffered entrances for a given GeoDataFrame.

    Parameters
    ----------
    df_entrances : gpd.GeoDataFrame
        A GeoDataFrame containing the entrances to plot.
    buffer_sizes : list
        A list of integers representing buffer sizes in meters.
    title : str
        The title to be used for the plot.

    Returns
    -------
    plt.Figure
        The matplotlib Figure object with the plotted data.
    """
    buffer_sizes.sort(reverse=True)

    ori_crs = gdf.crs
    utm_crs = gdf.estimate_utm_crs()

    df = gdf.copy()
    df.to_crs(utm_crs, inplace=True)

    # Create buffers and store them as GeoDataFrames
    geoms = []
    for size in buffer_sizes:
        geoms.append({"geometry": df.buffer(size).unary_union, "buffer_size": size})
    buffers = gpd.GeoDataFrame(geoms, crs=utm_crs).to_crs(ori_crs)

    # Create a plot
    params = {'facecolor': 'none', 'linestyle': '--'} 
    fig, ax = plot_geodata(buffers, **params, tile_alpha=.8, alpha=0, legend=True) # provider=1, zoom=19
    buffers.plot(ax=ax, alpha=.6, categorical=True, legend=True, **params)
    gdf.plot(color='r', ax=ax)

    # Extract and annotate entrance names
    if label:
        for x, item in gdf.iterrows():
            ax.text(item.geometry.x, item.geometry.y, f" {item[label]}", transform=ax.transData) 

    ax.set_title(title)

    return fig