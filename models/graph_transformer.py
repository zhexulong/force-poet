# ------------------------------------------------------------------------
# Graph Transformer for Force Prediction in PoET
# Based on relation_grasp implementation
# ------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MultiHeadAttentionLayer(nn.Module):
    """Multi-head attention layer for graph transformer"""
    
    def __init__(self, in_dim: int, out_dim: int, num_heads: int, use_bias: bool = False):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        
        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.proj_e = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.sqrt_dim = np.sqrt(out_dim)
    
    def forward(self, h: torch.Tensor, e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: Node features [B, num_objects, C]
            e: Edge features [B, num_objects, num_objects, C]
        Returns:
            h_out: Updated node features [B, num_objects, C]
            e_out: Updated edge features [B, num_objects, num_objects, C]
        """
        b, q1, c = h.shape
        b, q2, q3, c_e = e.shape
        assert q1 == q2 == q3, f"Dimension mismatch: {q1}, {q2}, {q3}"
        
        Q_h = self.Q(h)  # [B, num_objects, out_dim * num_heads]
        K_h = self.K(h)
        V_h = self.V(h)
        proj_e = self.proj_e(e)  # [B, num_objects, num_objects, out_dim * num_heads]
        
        # Reshape for multi-head attention
        Q_h = Q_h.view(b, q1, self.num_heads, self.out_dim).transpose(1, 2)  # [B, num_heads, num_objects, out_dim]
        K_h = K_h.view(b, q1, self.num_heads, self.out_dim).transpose(1, 2)
        V_h = V_h.view(b, q1, self.num_heads, self.out_dim).transpose(1, 2)
        proj_e = proj_e.view(b, q2, q3, self.num_heads, self.out_dim).permute(0, 3, 1, 2, 4)  # [B, num_heads, q2, q3, out_dim]
        
        # Compute attention scores
        # Q_h: [B, H, L, C], K_h: [B, H, L, C] -> [B, H, L, L, C]
        Q_expanded = Q_h.unsqueeze(3).expand(-1, -1, -1, q1, -1)  # [B, H, L, L, C]
        K_expanded = K_h.unsqueeze(2).expand(-1, -1, q1, -1, -1)  # [B, H, L, L, C]
        
        score = Q_expanded * K_expanded / self.sqrt_dim  # [B, H, L, L, C]
        score = score * proj_e  # Element-wise multiplication with edge features
        
        e_out = score  # Updated edge features
        
        # Aggregate to get attention weights
        score_weights = score.sum(dim=-1).clamp(-5, 5)  # [B, H, L, L]
        score_weights = torch.softmax(score_weights, dim=-1)
        
        # Apply attention to values
        h_out = torch.matmul(score_weights, V_h)  # [B, H, L, C]
        
        # Reshape back
        h_out = h_out.transpose(1, 2).contiguous().view(b, q1, -1)  # [B, L, H*C]
        e_out = e_out.permute(0, 2, 3, 1, 4).contiguous().view(b, q2, q3, -1)  # [B, L, L, H*C]
        
        return h_out, e_out


class GraphTransformerLayer(nn.Module):
    """Single layer of Graph Transformer"""
    
    def __init__(self, in_dim: int, out_dim: int, num_heads: int, num_nodes: int = 100, 
                 dropout: float = 0.0, layer_norm: bool = True, batch_norm: bool = False, 
                 residual: bool = True, use_bias: bool = False):
        super().__init__()
        
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.num_nodes = num_nodes
        
        # Multi-head attention
        self.attention = MultiHeadAttentionLayer(in_dim, out_dim // num_heads, num_heads, use_bias)
        
        # Output projections
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
        
        # Feed-forward networks
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)
        
        self.FFN_e_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_e_layer2 = nn.Linear(out_dim * 2, out_dim)
    
    def forward(self, h: torch.Tensor, e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: Node features [B, num_objects, C]
            e: Edge features [B, num_objects, num_objects, C]
        Returns:
            h_out: Updated node features
            e_out: Updated edge features
        """
        h_in1 = h  # For first residual connection
        e_in1 = e  # For first residual connection
        
        # Multi-head attention
        h_attn_out, e_attn_out = self.attention(h, e)
        
        h = h_attn_out
        e = e_attn_out
        
        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)
        
        h = self.O_h(h)
        e = self.O_e(e)
        
        if self.residual:
            h = h_in1 + h  # Residual connection
            e = e_in1 + e  # Residual connection
        
        h = self.norm1_h(h)
        e = self.norm1_e(e)
        
        h_in2 = h  # For second residual connection
        e_in2 = e  # For second residual connection
        
        # Feed-forward for nodes
        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)
        
        # Feed-forward for edges
        e = self.FFN_e_layer1(e)
        e = F.relu(e)
        e = F.dropout(e, self.dropout, training=self.training)
        e = self.FFN_e_layer2(e)
        
        if self.residual:
            h = h_in2 + h  # Residual connection
            e = e_in2 + e  # Residual connection
        
        h = self.norm2_h(h)
        e = self.norm2_e(e)
        
        return h, e


class GraphTransformer(nn.Module):
    """Graph Transformer for force prediction in PoET"""
    
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2, num_heads: int = 8, 
                 max_objects: int = 100, dropout: float = 0.1, use_edge_features: bool = True):
        super().__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_objects = max_objects
        self.use_edge_features = use_edge_features
        
        # Input projection for node features
        self.node_input_proj = nn.Linear(in_dim, hidden_dim)
        
        # Input projection for edge features (if used)
        if use_edge_features:
            # Edge features could be distance, relative position, etc.
            self.edge_input_proj = nn.Linear(3, hidden_dim)  # Assuming 3D relative position as edge feature
        else:
            self.edge_input_proj = nn.Linear(1, hidden_dim)  # Simple scalar edge feature
        
        # Graph transformer layers
        self.layers = nn.ModuleList([
            GraphTransformerLayer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                num_heads=num_heads,
                num_nodes=max_objects,
                dropout=dropout,
                layer_norm=True,
                residual=True
            ) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def create_adjacency_matrix(self, batch_size: int, num_objects: torch.Tensor, 
                               forces: Optional[list] = None) -> torch.Tensor:
        """
        Create adjacency matrix based on force interactions
        Args:
            batch_size: Batch size
            num_objects: Number of objects per batch [B]
            forces: List of force dictionaries per batch
        Returns:
            adjacency: [B, max_objects, max_objects]
        """
        device = num_objects.device
        adjacency = torch.zeros(batch_size, self.max_objects, self.max_objects, device=device)
        
        for b in range(batch_size):
            n_obj = num_objects[b].item()
            if forces is not None and forces[b]:
                # Create adjacency based on force information
                force_dict = forces[b]
                for i in range(n_obj):
                    for j in range(n_obj):
                        if i != j:
                            # Check if there's force interaction between objects i and j
                            force_key = f"{i}_{j}"
                            if force_key in force_dict:
                                adjacency[b, i, j] = 1.0
            else:
                # Fully connected graph if no force information
                adjacency[b, :n_obj, :n_obj] = 1.0
                adjacency[b].fill_diagonal_(0)  # No self-loops
        
        return adjacency
    
    def create_edge_features(self, node_positions: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Create edge features based on relative positions
        Args:
            node_positions: [B, max_objects, 3] - object positions
            adjacency: [B, max_objects, max_objects] - adjacency matrix
        Returns:
            edge_features: [B, max_objects, max_objects, edge_dim]
        """
        batch_size, max_objects, _ = node_positions.shape
        
        # Compute relative positions
        pos_i = node_positions.unsqueeze(2).expand(-1, -1, max_objects, -1)  # [B, max_objects, max_objects, 3]
        pos_j = node_positions.unsqueeze(1).expand(-1, max_objects, -1, -1)  # [B, max_objects, max_objects, 3]
        
        relative_pos = pos_j - pos_i  # [B, max_objects, max_objects, 3]
        
        if self.use_edge_features:
            edge_features = relative_pos
        else:
            # Use distance as edge feature
            distance = torch.norm(relative_pos, dim=-1, keepdim=True)  # [B, max_objects, max_objects, 1]
            edge_features = distance
        
        # Mask edges based on adjacency
        adjacency_expanded = adjacency.unsqueeze(-1)  # [B, max_objects, max_objects, 1]
        edge_features = edge_features * adjacency_expanded
        
        return edge_features
    
    def forward(self, node_features: torch.Tensor, node_positions: torch.Tensor, 
                num_objects: torch.Tensor, forces: Optional[list] = None) -> torch.Tensor:
        """
        Forward pass of Graph Transformer
        Args:
            node_features: [B, max_objects, in_dim] - features from PoET decoder
            node_positions: [B, max_objects, 3] - object positions for edge features
            num_objects: [B] - number of valid objects per batch
            forces: List of force dictionaries for creating adjacency matrix
        Returns:
            output_features: [B, max_objects, hidden_dim] - updated node features
        """
        batch_size, max_objects, _ = node_features.shape
        
        # Project input node features
        h = self.node_input_proj(node_features)  # [B, max_objects, hidden_dim]
        
        # Create adjacency matrix
        adjacency = self.create_adjacency_matrix(batch_size, num_objects, forces)
        
        # Create edge features
        edge_features = self.create_edge_features(node_positions, adjacency)
        e = self.edge_input_proj(edge_features)  # [B, max_objects, max_objects, hidden_dim]
        
        # Apply graph transformer layers
        for layer in self.layers:
            h, e = layer(h, e)
        
        # Output projection
        output_features = self.output_proj(h)
        
        return output_features


def build_graph_transformer(args):
    """Build graph transformer from args"""
    return GraphTransformer(
        in_dim=args.hidden_dim,
        hidden_dim=args.graph_hidden_dim if hasattr(args, 'graph_hidden_dim') else args.hidden_dim,
        num_layers=args.graph_num_layers if hasattr(args, 'graph_num_layers') else 2,
        num_heads=args.graph_num_heads if hasattr(args, 'graph_num_heads') else 8,
        max_objects=args.num_queries,
        dropout=args.dropout if hasattr(args, 'dropout') else 0.1,
        use_edge_features=args.use_edge_features if hasattr(args, 'use_edge_features') else True
    )
