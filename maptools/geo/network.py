import pandas as pd
import networkx as nx
from itertools import islice
from loguru import logger

from utils.serialization import save_checkpoint, load_checkpoint


class Network:
    def __init__(self, ckpt=None):
        self.ckpt = ckpt
        self.graph = nx.DiGraph()

    def _load_ckpt(self, ckpt):
        if ckpt is not None:
            try:
                load_checkpoint(ckpt, self)
                return True
            except:
                logger.warning(f"load {ckpt} failed!")
        
        return False

    def add_node(self, id, *args, **kwargs):
        self.graph.add_node(id, *args, **kwargs)

    def add_nodes(self, nodes:pd.DataFrame):
        for id, node in zip(nodes.index, nodes.to_dict(orient='records')):
            self.add_node(id, **node)

    def add_edge(self, src, dst, length=0, duration=0, *args, **kwargs):
        self.graph.add_edge(src, dst, distance=length, duration=duration, *args, **kwargs)

    def add_edges(self, edges:pd.DataFrame, length='length', duration='duration'):
        for _, link in edges.iterrows():
            self.add_edge(link.src, link.dst, length=link[length], duration=link[duration])

    def shortest_path(self, source, target, weight='distance'):
        try:
            return nx.shortest_path(self.graph, source, target, weight=weight)
        except nx.NetworkXNoPath:
            return "No path found"

    def top_k_paths(self, source, target, k, weight='distance'):
        try:
            return list(islice(nx.shortest_simple_paths(self.graph, source, target, weight=weight), k))
        except nx.NetworkXNoPath:
            return "No path found"

    def nodes_to_dataframe(self):
        df = pd.DataFrame.from_dict(dict(self.graph.nodes(data=True)), orient='index')
        df.loc[:, 'nid'] = df.index
        
        return df

    def edges_to_dataframe(self):
        edges_data = self.graph.edges(data=True)
        return pd.DataFrame([{'src': u, 'dst': v, **data} for u, v, data in edges_data])

    def get_node(self, nid, attr=None):
        if nid in self.graph.nodes:
            item = self.graph.nodes[nid]
            if attr in item:
                return item.get(attr, None)
            return item
        
        return None

    def save_ckpt(self, ckpt=None):
        if ckpt is not None:
            return save_checkpoint(self, ckpt)
        if self.ckpt is not None:
            return save_checkpoint(self, self.ckpt)

        return False
