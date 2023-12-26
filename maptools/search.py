import re
import geopandas as gpd
import matplotlib.pyplot as plt
from utils.vis import plot_geodata
from provider.search import search_API
from cfg import KEY


def plot_buffered_zones(gdf: gpd.GeoDataFrame, buffer_sizes: list, title: str) -> plt.Figure:
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
    pattern = r"地铁站(\w*)口"
    for x, item in gdf.iterrows():
        match = re.search(pattern, item['name'])
        name = match.group(1) if match else item['name']
        ax.text(item.geometry.x, item.geometry.y, f" {name}", transform=ax.transData) 

    ax.set_title(title)

    return fig

if __name__ == "__main__":
    station_name = "下梅林地铁站" # 景田, 上梅林，车公庙
    show_fields = ['children', 'business', 'indoor']
    df_entrances = search_API(station_name, 150501, "0755", page_size=10, ll_sys='wgs', show_fields=show_fields, key=KEY)
    
    # FIXME 定位原则
    max_idx = df_entrances.parent.value_counts().index[0]
    plot_buffered_zones(df_entrances.query("parent == @max_idx"), [50, 100, 200], station_name);

