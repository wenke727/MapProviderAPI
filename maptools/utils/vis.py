import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统可用“微软雅黑”或“SimHei”
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

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
    

def plot_geodata(gdf, provider=Satellite, *args, **kwargs):
    if isinstance(provider, int):
        provider = {
            0: Satellite,
            1: Vector,
            2: Normal,
        }[provider]
    if FLAG:
        return plotGeodata(gdf, provider=provider, *args, **kwargs)
    
    return gdf.plot()
