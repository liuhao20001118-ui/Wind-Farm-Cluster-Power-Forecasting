

import torch
import torch.nn as nn
from torch_scatter import scatter_add


class GCNLayer(nn.Module):

    def __init__(self, in_features, out_features, use_bias=True):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=use_bias)

    def forward(self, graph):
        sender_nodes = graph.nodes[graph.senders]

        aggregated_nodes = scatter_add(sender_nodes, graph.receivers, dim=0, dim_size=graph.nodes.shape[0])

        node_degrees = scatter_add(torch.ones_like(graph.senders, dtype=graph.nodes.dtype, device=graph.nodes.device),
                                   graph.receivers, dim=0, dim_size=graph.nodes.shape[0]).unsqueeze(1).clamp(min=1)
        aggregated_nodes = aggregated_nodes / node_degrees

        return self.linear(aggregated_nodes)