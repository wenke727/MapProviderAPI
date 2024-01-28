import time
import hashlib
import pickle
import os
import geopandas as gpd
import matplotlib.pyplot as plt


def to_geojson(gdf:gpd.GeoDataFrame, filename:str):
    if not str(filename).endswith("geojson"):
        filename = f"{filename}.geojson"

    convert_cols = []
    for col in gdf.columns:
        if gdf[col].apply(lambda x: isinstance(x, list)).any():
            gdf[col] = gdf[col].apply(str)
            convert_cols.append(col)
    
    return gdf.to_file(filename, driver="GeoJSON")

def load_checkpoint(ckpt_file_name, obj=None):
    _dict = {}
    if obj is not None and hasattr(obj, "__dict__"):
        _dict = obj.__dict__

    with open(ckpt_file_name, 'rb') as f:
        dict_ = pickle.load(f)
    _dict.update(dict_)
    
    return _dict
    
def save_checkpoint(obj, ckpt_file_name, ignore_att=[]):
    def _save(tmp):
        with open(ckpt_file_name, 'wb') as f:
            pickle.dump({ k: v for k, v in tmp.items() if k not in ignore_att}, f)
    
    if isinstance(obj, dict):
         _save(obj)
         return True
     
    try:                
        _save(obj.__dict__)
        return True
    except:
        return False
    
def save_fig(fig, fn, bbox_inches='tight', dpi=500, pad_inches=0.1, *args, **kwargs):
    """
    Save a matplotlib figure to a file.

    Parameters:
    fig : matplotlib.figure.Figure
        The figure object to be saved.
    fn : str
        The filename or path to save the figure to.
    bbox_inches : str or `~matplotlib.transforms.Bbox`, optional
        The bounding box in inches. Only the given portion of the figure is saved. 
        If 'tight', try to figure out the tight bbox of the figure.
    dpi : int, optional
        The resolution of the figure in dots-per-inch.
    pad_inches : float, optional
        Amount of padding around the figure when bbox_inches is 'tight'.

    Other Parameters:
    *args, **kwargs : 
        Additional arguments and keyword arguments to be passed to `fig.savefig()`.

    Returns:
    None

    Example:
    >>> fig = plt.figure()
    >>> save_fig(fig, 'my_plot.png')
    """
    
    return fig.savefig(
        fn, bbox_inches=bbox_inches, 
        pad_inches=pad_inches, dpi=dpi, *args, **kwargs
    )
    
