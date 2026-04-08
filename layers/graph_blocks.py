

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.graphs import GraphsTuple


def broadcast_receiver_nodes_to_edges(graph: GraphsTuple):
    return graph.nodes.index_select(index=graph.receivers.long(), dim=0)
def broadcast_sender_nodes_to_edges(graph: GraphsTuple):
    return graph.nodes.index_select(index=graph.senders.long(), dim=0)
def unsorted_sum_agg(data, segment_ids):
    num_edges = torch.unique(segment_ids).shape[0]
    output = torch.zeros((num_edges, *data.shape[1:])).to(data.device)
    if len(data.shape) == 3:
        output = output.scatter_add(0, segment_ids.unsqueeze(-1).unsqueeze(-1).expand(data.shape), data)
    else:
        output.scatter_add(0, segment_ids.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(data.shape), data)
    return output
def unsorted_mean_agg(data, segment_ids):
    unique_ids, counts = torch.unique(segment_ids, return_counts=True)
    num_edges = len(unique_ids)
    output = torch.zeros((num_edges, *data.shape[1:])).to(data.device)
    if len(data.shape) == 3:
        output = output.scatter_add(0, segment_ids.unsqueeze(-1).unsqueeze(-1).expand(data.shape), data)
        counts = counts.view(-1, 1, 1).expand_as(output)  # 形状从 [num_edges] 扩展到 [num_edges, 36, 128]
        mask = counts > 0  # 形状为 [num_edges, 36, 128]
        output[mask] = output[mask] / counts[mask]
    elif len(data.shape) == 2:
        output = output.scatter_add(0, segment_ids.unsqueeze(-1).expand(data.shape), data)
        counts = counts.view(-1, 1).expand_as(output)
        mask = counts > 0
        output[mask] = output[mask] / counts[mask]
    else:
        raise NotImplementedError
    return output


def unsorted_softmax(data, segment_ids):
    data = torch.exp(data)
    num_edges = torch.unique(segment_ids).shape[0]
    denom = torch.zeros((num_edges, *data.shape[1:])).to(data.device)
    denom = denom.scatter_add(0, segment_ids.unsqueeze(-1).unsqueeze(-1).expand(data.shape), data)
    denom = denom.index_select(index=segment_ids.long(), dim=0)
    return data / denom


class EdgeBlock(nn.Module):
    def __init__(self, update_fn, d_model, use_edges=True, use_receiver_nodes=True, use_sender_nodes=True,
                 num_node_series=1, num_edge_series=1):
        super(EdgeBlock, self).__init__()
        self._use_edges = use_edges
        self._use_receiver_nodes = use_receiver_nodes
        self._use_sender_nodes = use_sender_nodes
        d_in = int(d_model * (
                    use_edges * num_edge_series + use_receiver_nodes * num_node_series + use_sender_nodes * num_node_series))
        self.project = nn.Linear(d_in, d_model, bias=False)
        self.update_fn = update_fn
        self.num_node_series = num_node_series
        self.num_edge_series = num_edge_series
        # 自适应邻接矩阵参数
        self.edge_weight = nn.Parameter(torch.ones(1, d_model))
        # 风向影响权重
        self.wind_dir_weight = nn.Linear(1, 1)

    def forward(self, graph: GraphsTuple, **kwargs):
        # 收集边特征
        edges_to_collect = []

        if self._use_edges:
            edges_to_collect.append(graph.edges)
        if self._use_receiver_nodes:
            edges_to_collect.append(broadcast_receiver_nodes_to_edges(graph))
        if self._use_sender_nodes:
            edges_to_collect.append(broadcast_sender_nodes_to_edges(graph))

        # 拼接并通过线性层处理
        collected_edges = torch.cat(edges_to_collect, dim=-1)
        collected_edges = self.project(collected_edges)

        # 获取发送和接收节点的特征
        sender_feats = broadcast_sender_nodes_to_edges(graph)
        receiver_feats = broadcast_receiver_nodes_to_edges(graph)

        # 归一化特征
        sender_feats = F.normalize(sender_feats, p=2, dim=-1)
        receiver_feats = F.normalize(receiver_feats, p=2, dim=-1)

        # 计算注意力分数
        attention_scores = F.relu((sender_feats * receiver_feats).sum(dim=-1, keepdim=True)) + 1e-8

        # 调试：检查 NaN
        if torch.isnan(attention_scores).any():
            print("NaN detected in attention_scores")

        attention_weights = F.softmax(attention_scores, dim=-1)

        wind_dir_diff = graph.edges[..., -1:]
        wind_dir_factor = torch.sigmoid(self.wind_dir_weight(wind_dir_diff))

        updated_edges = collected_edges * attention_weights * wind_dir_factor

        updated_edges, attn_edges = self.update_fn(updated_edges, **kwargs)

        graph = graph.replace(edges=updated_edges)
        return graph, attn_edges


class NodeBlock(nn.Module):
    def __init__(self, update_fn, d_model, use_received_edges=True, use_sent_edges=False, use_nodes=True,
                 edges_agg=unsorted_mean_agg, num_node_series=1, num_edge_series=1):
        super(NodeBlock, self).__init__()
        self._use_received_edges = use_received_edges
        self._use_sent_edges = use_sent_edges
        self._use_nodes = use_nodes
        d_in = int(d_model * (use_received_edges + use_sent_edges + use_nodes))
        self.project = nn.Linear(d_in, d_model, bias=False)
        self.update_fn = update_fn
        self._received_edges_aggregator = _EdgesToNodesAggregator(edges_agg, use_sent_edges=False)
        self._sent_edges_aggregator = _EdgesToNodesAggregator(edges_agg, use_sent_edges=True)
        self.num_node_series = num_node_series
        self.num_edge_series = num_edge_series
        # 时间卷积
        self.time_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)

    def forward(self, graph, **kwargs):
        nodes_to_collect = []
        if self._use_received_edges:
            nodes_to_collect += self._received_edges_aggregator(graph)
        if self._use_sent_edges:
            nodes_to_collect += self._sent_edges_aggregator(graph)
        if self._use_nodes:
            nodes_to_collect.append(graph.nodes)

        collected_nodes = torch.cat(nodes_to_collect, dim=-1)
        collected_nodes = self.project(collected_nodes)

        # 时间卷积
        collected_nodes = collected_nodes.permute(0, 2, 1)  # [N, D, S]
        collected_nodes = self.time_conv(collected_nodes)  # [N, D, S]
        collected_nodes = collected_nodes.permute(0, 2, 1)  # [N, S, D]
        updated_nodes, attn_nodes = self.update_fn(collected_nodes, **kwargs)
        graph = graph.replace(nodes=updated_nodes)
        return graph, attn_nodes

class _EdgesToNodesAggregator(nn.Module):
    def __init__(self, reducer, use_sent_edges=False):
        super(_EdgesToNodesAggregator, self).__init__()
        self._reducer = reducer
        self._use_sent_edges = use_sent_edges

    def forward(self, graph):
        indices = graph.senders if self._use_sent_edges else graph.receivers
        return [self._reducer(graph.edges, indices)]