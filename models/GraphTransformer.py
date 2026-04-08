import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer, Decoder, DecoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.graph_modules import GCNLayer
from layers.Embed import GraphDataEmbedding
from layers.graph_blocks import EdgeBlock, NodeBlock

class AdaptiveGraphWeighting(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_weights = nn.Parameter(torch.randn(4)) 
        
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, graph):
        similarity_components = graph.edges[:, :4] 
        
        raw_features = graph.edges[:, 4:]
        

        normalized_weights = F.softmax(self.raw_weights, dim=0)

        weighted_sum = (similarity_components * normalized_weights).sum(dim=-1, keepdim=True)
        coupling_strength = F.relu(weighted_sum + self.bias)

        gated_raw_features = raw_features * coupling_strength

        new_edges = torch.cat([coupling_strength, gated_raw_features], dim=-1)
        
        return graph.replace(edges=new_edges), normalized_weights

class GatingNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super(GatingNetwork, self).__init__()
        self.fc1 = nn.Linear(d_model * 2, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, internal_state, env_context):
        x = torch.cat([internal_state, env_context], dim=-1)
        x = F.relu(self.fc1(x))
        gate = torch.sigmoid(self.fc2(x))
        return gate


class WindGraphProcessor(nn.Module):
    def __init__(self, configs):
        super().__init__()
        encoder_layers = [
            EncoderLayer(
                AttentionLayer(FullAttention(False, attention_dropout=configs.dropout), configs.d_model,
                               configs.n_heads),
                configs.d_model, configs.d_ff, dropout=configs.dropout, activation=configs.activation
            ) for _ in range(configs.e_layers)
        ]
        self.encoder = Encoder(encoder_layers)
        self.node_block = NodeBlock(update_fn=self.encoder, d_model=configs.d_model)
        self.edge_block = EdgeBlock(update_fn=self.encoder, d_model=configs.d_model)

    def forward(self, graph, **kwargs):
        graph, _ = self.edge_block(graph, **kwargs)
        graph, attn = self.node_block(graph, **kwargs)
        return graph, attn


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len
        self.output_attention = configs.output_attention
        self.c_out = configs.c_out
        self.d_model = configs.d_model

        self.adaptive_weighting = AdaptiveGraphWeighting()

        self.wind_dyn_enc_embedding = GraphDataEmbedding(
            c_in=configs.enc_in, d_model=configs.d_model, freq=configs.freq, dropout=configs.dropout, 
            edge_feats=4 
        )

        self.wind_stat_enc_embedding = GraphDataEmbedding(
            c_in=configs.enc_in, d_model=configs.d_model, freq=configs.freq, dropout=configs.dropout, 
            edge_feats=7 
        )
        
        self.era5_enc_embedding = GraphDataEmbedding(
            c_in=configs.era5_in, d_model=configs.d_model, freq=configs.freq, dropout=configs.dropout, edge_feats=3)

        self.wind_dec_embedding = GraphDataEmbedding(
            c_in=configs.dec_in, d_model=configs.d_model, freq=configs.freq, dropout=configs.dropout,
            edge_feats=7 
        )

        self.wind_dynamic_processor = WindGraphProcessor(configs)
        self.wind_static_processor = WindGraphProcessor(configs)
        self.wind_fusion = nn.Linear(self.d_model * 2, self.d_model)

        self.era5_time_encoder = Encoder(
            [EncoderLayer(AttentionLayer(FullAttention(False), configs.d_model, configs.n_heads), configs.d_model,
                          configs.d_ff)],
            norm_layer=nn.LayerNorm(configs.d_model)
        )
        self.era5_spatial_gnn = nn.Sequential(
            GCNLayer(configs.d_model, configs.d_model), nn.ReLU(),
            GCNLayer(configs.d_model, configs.d_model)
        )

        self.gating_network = GatingNetwork(configs.d_model, configs.d_ff)
        self.context_projection = nn.Linear(configs.d_model, configs.d_model)

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(FullAttention(True, attention_dropout=configs.dropout), configs.d_model,
                                   configs.n_heads),  # Self-Attention
                    AttentionLayer(FullAttention(False, attention_dropout=configs.dropout), configs.d_model,
                                   configs.n_heads),  # Cross-Attention
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)  # Using e_layers for decoder layers as well
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, self.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, center_indices=None, **kwargs):
        if isinstance(x_enc, dict) and 'era5' in x_enc:  # ERA5 Case
            return self._forward_era5(x_enc, x_mark_enc, x_dec, x_mark_dec, center_indices)
        else:  # WindGraph Only Case
            return self._forward_wind_only(x_enc, x_mark_enc, x_dec, x_mark_dec)

    def _prepare_encoder_output(self, x_enc, x_mark_enc, center_indices=None):
        wind_dyn_graph = x_enc['wind_dynamic']
        
        wind_dyn_graph, current_weights = self.adaptive_weighting(wind_dyn_graph)
        
        wind_dyn_emb = self.wind_dyn_enc_embedding(wind_dyn_graph, x_mark_enc)
        
        wind_stat_emb = self.wind_stat_enc_embedding(x_enc['wind_static'], x_mark_enc)

        processed_dyn_graph, _ = self.wind_dynamic_processor(wind_dyn_emb)
        processed_stat_graph, _ = self.wind_static_processor(wind_stat_emb)

        internal_state_nodes = self.wind_fusion(
            torch.cat([processed_dyn_graph.nodes, processed_stat_graph.nodes], dim=-1)
        )

        if center_indices is not None:
            era5_graph_emb = self.era5_enc_embedding(x_enc['era5'], x_mark_enc)
            era5_nodes_time_encoded, _ = self.era5_time_encoder(era5_graph_emb.nodes)
            era5_nodes_repr = era5_nodes_time_encoded[:, -1, :]  # Use last time step representation

            spatial_era5_graph = x_enc['era5'].replace(nodes=era5_nodes_repr)
            era5_nodes_processed = spatial_era5_graph.nodes
            for layer in self.era5_spatial_gnn:
                if isinstance(layer, GCNLayer):
                    era5_nodes_processed = layer(
                        spatial_era5_graph.replace(nodes=era5_nodes_processed)) + era5_nodes_processed
                else:
                    era5_nodes_processed = layer(era5_nodes_processed)

            era5_node_counts = x_enc['era5'].n_node
            node_offsets = torch.cat([torch.tensor([0], device=era5_node_counts.device, dtype=torch.long),
                                      torch.cumsum(era5_node_counts, 0)[:-1]])
            abs_center_indices = center_indices.to(node_offsets.device) + node_offsets
            env_context = era5_nodes_processed[abs_center_indices]

            wind_node_counts = processed_dyn_graph.n_node
            broadcasted_context = torch.repeat_interleave(env_context, wind_node_counts, dim=0)

            projected_context = self.context_projection(broadcasted_context)
            projected_context = projected_context.unsqueeze(1).expand(-1, internal_state_nodes.shape[1], -1)

            gate = self.gating_network(internal_state_nodes, projected_context)
            encoder_output = (1 - gate) * internal_state_nodes + gate * projected_context
        else:  # No ERA5 data
            encoder_output = internal_state_nodes

        return encoder_output, processed_dyn_graph 

    def _forward_era5(self, x_enc, x_mark_enc, x_dec, x_mark_dec, center_indices):
        enc_out, graph_struct = self._prepare_encoder_output(x_enc, x_mark_enc, center_indices)

        dec_emb = self.wind_dec_embedding(x_dec, x_mark_dec)
        dec_out, _ = self.decoder(dec_emb.nodes, enc_out)

        return graph_struct.replace(nodes=dec_out)

    def _forward_wind_only(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc_dict = {'wind_dynamic': x_enc[0], 'wind_static': x_enc[1]}
        enc_out, graph_struct = self._prepare_encoder_output(x_enc_dict, x_mark_enc, center_indices=None)

        dec_emb = self.wind_dec_embedding(x_dec, x_mark_dec)
        dec_out, _ = self.decoder(dec_emb.nodes, enc_out)

        return graph_struct.replace(nodes=dec_out)