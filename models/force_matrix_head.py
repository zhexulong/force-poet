"""
Force Matrix Head for Pairwise Force Prediction

This module implements a specialized head for predicting pairwise forces between objects,
inspired by the DenseGraphTransformerHead from relation_grasp.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops as ei


class MLP(nn.Module):
    """Multi-layer perceptron"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class PairwiseConcatLayer(nn.Module):
    """Layer for creating pairwise features by concatenating node features"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x, y):
        """
        Args:
            x: [B, N, C] - First set of node features
            y: [B, N, C] - Second set of node features
        Returns:
            [B, N, N, 2*C] - Pairwise concatenated features
        """
        d1, d2 = x.shape[-2], y.shape[-2]
        grid_x, grid_y = torch.meshgrid(
            torch.arange(d1, device=x.device), 
            torch.arange(d2, device=y.device), 
            indexing='ij'
        )
        res = torch.concat([
            torch.index_select(x, dim=-2, index=grid_x.flatten()), 
            torch.index_select(y, dim=-2, index=grid_y.flatten())
        ], dim=-1)
        res = ei.rearrange(res, '... (L1 L2) C -> ... L1 L2 C', L1=d1, L2=d2)
        return res


class MultiHeadAttentionLayer(nn.Module):
    """Multi-head attention layer for node and edge features"""
    
    def __init__(self, in_dim, out_dim, num_heads, use_bias=False):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.proj_e = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.sqrt_dim = (out_dim ** 0.5)
    
    def forward(self, h, e):
        """
        Args:
            h: [B, N, C] - Node features
            e: [B, N, N, C] - Edge features
        Returns:
            h_out: [B, N, C] - Updated node features
            e_out: [B, N, N, C] - Updated edge features
        """
        b, q1, c = h.shape
        b, q2, q3, c = e.shape
        assert q1 == q2 == q3
        
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)
        proj_e = self.proj_e(e)
        
        Q_h = ei.rearrange(Q_h, 'B L (H C) -> B H L C', H=self.num_heads, C=self.out_dim)
        K_h = ei.rearrange(K_h, 'B L (H C) -> B H L C', H=self.num_heads, C=self.out_dim)
        V_h = ei.rearrange(V_h, 'B L (H C) -> B H L C', H=self.num_heads, C=self.out_dim)
        proj_e = ei.rearrange(proj_e, 'B L1 L2 (H C) -> B H L1 L2 C', H=self.num_heads, C=self.out_dim)
        
        # Compute attention scores
        score = ei.repeat(Q_h, 'B H L C -> B H L Lr C', Lr=q1) * ei.repeat(K_h, 'B H L C -> B H Lr L C', Lr=q2) / self.sqrt_dim
        score = score * proj_e
        e_out = score
        
        score = ei.reduce(score, 'B H L1 L2 C -> B H L1 L2', 'sum').clamp(-5, 5)
        score = torch.nn.functional.softmax(score, dim=2)
        h_out = score @ V_h
        
        h_out = ei.rearrange(h_out, 'B H L C -> B L (H C)')
        e_out = ei.rearrange(e_out, 'B H L1 L2 C -> B L1 L2 (H C)') 
        
        return h_out, e_out


class GraphTransformerLayerDense(nn.Module):
    """Dense Graph Transformer Layer for simultaneous node and edge updates"""
    
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, layer_norm=True, 
                 batch_norm=False, residual=True, use_bias=False):
        super().__init__()
        
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm     
        self.batch_norm = batch_norm
        
        self.attention = MultiHeadAttentionLayer(in_dim, out_dim//num_heads, num_heads, use_bias)
        
        self.O_h = nn.Linear(out_dim, out_dim)
        self.O_e = nn.Linear(out_dim, out_dim)
        
        # Normalization layers
        if layer_norm:
            self.norm1_h = nn.LayerNorm(out_dim)
            self.norm1_e = nn.LayerNorm(out_dim)
            self.norm2_h = nn.LayerNorm(out_dim)
            self.norm2_e = nn.LayerNorm(out_dim)
        else:
            self.norm1_h = nn.Identity()
            self.norm1_e = nn.Identity()
            self.norm2_h = nn.Identity()
            self.norm2_e = nn.Identity()
            
        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_h_layer2 = nn.Linear(out_dim*2, out_dim)
        
        # FFN for e
        self.FFN_e_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_e_layer2 = nn.Linear(out_dim*2, out_dim)
        
    def forward(self, h, e):
        """
        Args:
            h: [B, N, C] - Node features
            e: [B, N, N, C] - Edge features
        Returns:
            h: [B, N, C] - Updated node features
            e: [B, N, N, C] - Updated edge features
        """
        h_in1 = h  # for first residual connection
        e_in1 = e  # for first residual connection
        
        # Multi-head attention out
        h_attn_out, e_attn_out = self.attention(h, e)
        
        h = h_attn_out
        e = e_attn_out
        
        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        h = self.O_h(h)
        e = self.O_e(e)

        if self.residual:
            h = h_in1 + h  # residual connection
            e = e_in1 + e  # residual connection

        h = self.norm1_h(h)
        e = self.norm1_e(e)
        
        h_in2 = h  # for second residual connection
        e_in2 = e  # for second residual connection

        # FFN for h
        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        # FFN for e
        e = self.FFN_e_layer1(e)
        e = F.relu(e)
        e = F.dropout(e, self.dropout, training=self.training)
        e = self.FFN_e_layer2(e)

        if self.residual:
            h = h_in2 + h  # residual connection       
            e = e_in2 + e  # residual connection           

        h = self.norm2_h(h)
        e = self.norm2_e(e)
        
        return h, e


class ForceMatrixHead(nn.Module):
    """
    Specialized head for predicting pairwise force matrices between objects.
    Inspired by DenseGraphTransformerHead from relation_grasp.
    """
    
    def __init__(self, in_dim=256, hidden_dim=256, num_heads=8, num_layers=2, 
                 edge_features='sum', dropout=0.1, use_bias=False, mass_embed_dim=None):
        super().__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.edge_features = 'concat' # edge_features
        self.dropout = dropout
        self.mass_embed_dim = mass_embed_dim
        
        # Edge feature computation
        if edge_features == 'concat':
            out_proj_edge = hidden_dim // 2
            self.pairwise_layer = PairwiseConcatLayer()
        else:
            out_proj_edge = hidden_dim 

        self.proj_e1 = nn.Linear(in_dim, out_proj_edge, bias=use_bias)
        self.proj_e2 = nn.Linear(in_dim, out_proj_edge, bias=use_bias)
        
        proj_node_in_dim = in_dim + (mass_embed_dim if mass_embed_dim is not None else 0)
        self.proj_node_input = nn.Linear(proj_node_in_dim, hidden_dim, bias=use_bias)
        
        # Graph transformer layers
        self.graph_transformer_layers = nn.ModuleList([
            GraphTransformerLayerDense(
                in_dim=hidden_dim, 
                out_dim=hidden_dim, 
                num_heads=num_heads, 
                dropout=dropout,
                layer_norm=True, 
                batch_norm=False,
                use_bias=use_bias
            )
            for _ in range(num_layers)
        ])
        
        # Force prediction head
        self.force_predictor = MLP(
            input_dim=hidden_dim, 
            hidden_dim=hidden_dim//2, 
            output_dim=3,  # XYZ force components
            num_layers=3
        )
        
    def _compute_edge_features(self, features):
        """
        Compute pairwise edge features from node features
        
        Args:
            features: [B, N, C] - Node features
        Returns:
            e: [B, N, N, C] - Edge features
        """
        B, N, C = features.shape
        
        e1, e2 = self.proj_e1(features), self.proj_e2(features)
        
        if self.edge_features == 'concat':
            e = self.pairwise_layer(e1, e2)
        elif self.edge_features == 'sum':
            e = e1[:, None, :, :] + e2[:, :, None, :]
        elif self.edge_features == 'diff': 
            e = e1[:, None, :, :] - e2[:, :, None, :]
        elif self.edge_features == 'mul':
            e = e1[:, None, :, :] * e2[:, :, None, :]
        else:
            raise NotImplementedError(f'{self.edge_features} aggregation not implemented')
        
        return e
    
    def forward(self, hs, mass_embeds=None):
        """
        Forward pass for force matrix prediction
        
        Args:
            hs: [B, N, C] - Object features from PoET decoder
            mass_embeds: [B, N, D_mass] - Optional mass embeddings for each node
        Returns:
            pred_force_matrix: [B, N, N, 3] - Predicted pairwise force matrix
        """
        B, N, C = hs.shape
        
        # Compute initial edge features
        e = self._compute_edge_features(hs)
        
        # Concatenate mass embeddings to node features if provided
        node_features = hs
        if mass_embeds is not None and self.mass_embed_dim is not None:
            node_features = torch.cat([hs, mass_embeds], dim=-1)

        # Project node features to hidden dimension
        h = self.proj_node_input(node_features)
        
        # Apply graph transformer layers
        for layer in self.graph_transformer_layers:
            h, e = layer(h, e)
            
        # Predict forces from final edge features
        pred_force_matrix = self.force_predictor(e)
        
        return pred_force_matrix
