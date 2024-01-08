#%%
import sys
sys.path.append('./maptools')

import re
from provider.search import search_API
from geo.visualize import plot_buffered_zones
from cfg import KEY


def extract_entrance_num(item, pattern = r"地铁站(\w*)口"):
    match = re.search(pattern, item)
    name = match.group(1) if match else item
    
    return name


if __name__ == "__main__":
    station_name = "下梅林地铁站" # 景田, 上梅林，车公庙
    show_fields = ['children', 'business', 'indoor']
    df_entrances = search_API(station_name, 150501, "0755", show_fields=show_fields, key=KEY)
    
    # FIXME 定位原则
    max_idx = df_entrances.parent.value_counts().index[0]
    df_entrances = df_entrances.assign(label=df_entrances['name'].apply(extract_entrance_num))
    # plot_buffered_zones(df_entrances.query("parent == @max_idx"), [50, 100, 200], station_name);
    plot_buffered_zones(df_entrances.query("parent == @max_idx"), [10], station_name, label='label');


# %%
