import torch
from layers.graphs import GraphsTuple
import numpy as np


import torch
from layers.graphs import GraphsTuple


def data_dicts_to_graphs_tuple(graph_dict, device):
    # This function now handles a single dictionary, not a list
    n_node = torch.tensor(graph_dict['n_node'], dtype=torch.int64, device=device)
    n_edge = torch.tensor(graph_dict['n_edge'], dtype=torch.int64, device=device)
    nodes = torch.tensor(graph_dict['nodes'], dtype=torch.float32, device=device)
    edges = torch.tensor(graph_dict['edges'], dtype=torch.float32, device=device)
    senders = torch.tensor(graph_dict['senders'], dtype=torch.int64, device=device)
    receivers = torch.tensor(graph_dict['receivers'], dtype=torch.int64, device=device)

    # 'graph_mapping' is not used in the new setup, but kept for compatibility
    graph_mapping = graph_dict.get('graph_mapping', np.zeros(graph_dict['nodes'].shape[0], dtype=np.int64))
    graph_mapping = torch.tensor(graph_mapping, dtype=torch.int64, device=device)

    station_names = graph_dict['station_names']

    return GraphsTuple(
        nodes=nodes,
        edges=edges,
        senders=senders,
        receivers=receivers,
        n_node=n_node,
        n_edge=n_edge,
        graph_mapping=graph_mapping,
        station_names=station_names
    )


# In development. Should work fine, but might contain some bugs.
def split_torch_graph(graph, target_gpus):
    target_gpus = ['cuda:' + str(gpu) for gpu in target_gpus]
    bs = graph.n_node.shape[0]

    sub_bs = np.array_split(np.arange(bs), len(target_gpus))

    sum_node_prev = 0
    sum_edge_prev = 0
    graph_list = []
    for gpu_i, sub_i in zip(target_gpus, sub_bs):
        sub_i = torch.from_numpy(sub_i).to(graph.nodes.device).long()
        graph_i = graph
        end_node = torch.sum(graph.n_node[sub_i]) + sum_node_prev
        end_edge = torch.sum(graph.n_edge[sub_i]) + sum_edge_prev

        graph_i = graph_i.replace(
            nodes=graph.nodes[sum_node_prev:end_node].to(gpu_i),
            edges=graph.edges[sum_edge_prev:end_edge].to(gpu_i),
            senders=(graph.senders[sum_edge_prev:end_edge] - sum_node_prev).to(gpu_i),
            receivers=(graph.receivers[sum_edge_prev:end_edge] - sum_node_prev).to(gpu_i),
            n_node=graph.n_node[sub_i].to(gpu_i),
            n_edge=graph.n_edge[sub_i].to(gpu_i),
            station_names=graph.station_names[sum_node_prev:end_node],
        )
        sum_node_prev = end_node
        sum_edge_prev = end_edge
        graph_list.append(graph_i)

    return graph_list, sub_bs, target_gpus
