# %%
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import Point
from geo.distance import cal_pointwise_distance_geoseries as cal_distance

records = gpd.read_file("../data/cells/traj_00011.geojson")
records.geometry = records.geometry.fillna(Point())
records.lac = records.lac.fillna(-1).astype(int)
records.duration = records.duration.fillna(0)
records.loc[:, 'rid'] = 1

records.loc[80:90]


# %%
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt

class CellGraphBuilder:
    def __init__(self, geo_df=None):
        self.graph = nx.DiGraph()
        self.coordinates = {}  # Dictionary to store coordinates
        if geo_df is not None:
            self.update_graph(geo_df)

    def _extract_cell_data(self, geo_df):
        return [(row['lac'], row['cid']) for _, row in geo_df.iterrows()]

    def update_graph(self, geo_df):
        """
        Updates the graph with new GeoDataFrame data.
        :param geo_df: New GeoDataFrame to update the graph.
        """
        cell_data = self._extract_cell_data(geo_df)

        for i in range(len(cell_data) - 1):
            source = cell_data[i]
            target = cell_data[i + 1]

            if self.graph.has_edge(source, target):
                self.graph[source][target]['weight'] += 1
            else:
                self.graph.add_edge(source, target, weight=1)

    def get_graph(self):
        return self.graph

    def visualize_graph(self):
        pos = nx.spring_layout(self.graph)

        nx.draw_networkx_nodes(self.graph, pos, alpha=.1)
        nx.draw_networkx_edges(self.graph, pos, width=[data['weight'] for _, _, data in self.graph.edges(data=True)])
        # nx.draw_networkx_labels(self.graph, pos, font_size=10, font_family="sans-serif")

        plt.axis('off')
        plt.show()

# Example usage:
builder = CellGraphBuilder()
builder.update_graph(records)
builder.visualize_graph()


# %%
records.plot()

# %%
edges = pd.DataFrame(builder.graph.edges(data=True), columns=['src', 'dst', 'attrs'])
edges = edges.assign(weight = edges['attrs'].apply(lambda x: x['weight'])) 
edges.sort_values('weight')


# %%
