import collections
NODES = "nodes"
EDGES = "edges"
RECEIVERS = "receivers"
SENDERS = "senders"
N_NODE = "n_node"
N_EDGE = "n_edge"
GRAPH_MAPPING = "graph_mapping"
STATION_NAMES = "station_names"

GRAPH_FEATURE_FIELDS = (NODES, EDGES)
GRAPH_INDEX_FIELDS = (RECEIVERS, SENDERS)
GRAPH_DATA_FIELDS = (NODES, EDGES, RECEIVERS, SENDERS, N_NODE, N_EDGE, GRAPH_MAPPING, STATION_NAMES)
class GraphsTuple(
    collections.namedtuple("GraphsTuple",
                           GRAPH_DATA_FIELDS)):
    def __init__(self, *args, **kwargs):
        del args, kwargs
        # The fields of a `namedtuple` are filled in the `__new__` method.
        # `__init__` does not accept parameters.
        super(GraphsTuple, self).__init__()

    def replace(self, **kwargs):
        output = self._replace(**kwargs)
        return output

    def map(self, field_fn, fields=GRAPH_FEATURE_FIELDS):
        return self.replace(**{k: field_fn(getattr(self, k)) for k in fields})
