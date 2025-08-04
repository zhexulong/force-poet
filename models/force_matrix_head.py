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
    Enhanced with geometric feature injection for better 3D spatial reasoning.
    Inspired by DenseGraphTransformerHead from relation_grasp.
    """
    
    def __init__(self, in_dim=256, hidden_dim=256, num_heads=8, num_layers=2, 
                 edge_features='concat', dropout=0.1, use_bias=False, mass_embed_dim=None,
                 use_geometric_features=True):
        super().__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.edge_features = edge_features
        self.dropout = dropout
        self.mass_embed_dim = mass_embed_dim
        self.use_geometric_features = use_geometric_features
        
        # Semantic edge feature computation
        if edge_features == 'concat':
            proj_edge_dim = hidden_dim // 2  # Dimension for each projection
            semantic_edge_dim = proj_edge_dim * 2  # After concatenation
            self.pairwise_layer = PairwiseConcatLayer()
        else:
            proj_edge_dim = hidden_dim
            semantic_edge_dim = hidden_dim 

        self.proj_e1 = nn.Linear(in_dim, proj_edge_dim, bias=use_bias)
        self.proj_e2 = nn.Linear(in_dim, proj_edge_dim, bias=use_bias)
        
        # Geometric feature embedder (Project, then Fuse)
        if self.use_geometric_features:
            # Geometric features: [relative_pos(3) + relative_pos(3)] = 6 dimensions
            geometric_feature_dim = hidden_dim // 8
            self.geometric_embedder = MLP(
                input_dim=6,  # relative_pos(3) + relative_pos(3)
                hidden_dim=geometric_feature_dim,
                output_dim=geometric_feature_dim,
                num_layers=2
            )
        
        # Mass feature embedder
        if mass_embed_dim is not None:
            mass_feature_dim = hidden_dim // 8
            self.mass_pairwise_embedder = MLP(
                input_dim=mass_embed_dim * 2,  # Pairwise concatenated mass features
                hidden_dim=mass_feature_dim,
                output_dim=mass_feature_dim,
                num_layers=2
            )
        
        # Feature fusion layer
        if self.use_geometric_features and mass_embed_dim is not None:
            # semantic + geometric + mass features
            fusion_input_dim = semantic_edge_dim + (hidden_dim // 8) + (hidden_dim // 8)
        elif self.use_geometric_features:
            # semantic + geometric features
            fusion_input_dim = semantic_edge_dim + (hidden_dim // 8)
        elif mass_embed_dim is not None:
            # semantic + mass features
            fusion_input_dim = semantic_edge_dim + (hidden_dim // 8)
        else:
            # semantic features only
            fusion_input_dim = semantic_edge_dim
            
        self.fusion_layer = nn.Linear(fusion_input_dim, hidden_dim, bias=use_bias)
        
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
        
        # Contact classification head
        self.contact_predictor = MLP(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim//2,
            output_dim=1,  # Contact probability logit
            num_layers=3
        )
        
    def _compute_semantic_edge_features(self, features):
        """
        Compute pairwise semantic edge features from node features
        
        Args:
            features: [B, N, C] - Node features
        Returns:
            e: [B, N, N, C] - Semantic edge features
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
    
    def _compute_geometric_edge_features(self, pred_translations):
        """
        Compute explicit geometric features from predicted translations
        
        Args:
            pred_translations: [B, N, 3] - Predicted 3D positions
        Returns:
            geometric_features: [B, N, N, 6] - Pairwise geometric features
        """
        B, N, _ = pred_translations.shape
        
        # Calculate pairwise relative positions
        pos_i = pred_translations.unsqueeze(2)  # [B, N, 1, 3]
        pos_j = pred_translations.unsqueeze(1)  # [B, 1, N, 3]
        relative_pos_ij = pos_j - pos_i         # [B, N, N, 3] - from i to j
        relative_pos_ji = pos_i - pos_j         # [B, N, N, 3] - from j to i
        
        # Combine both relative positions
        geometric_features = torch.cat([relative_pos_ij, relative_pos_ji], dim=-1)  # [B, N, N, 6]
        
        return geometric_features
    
    def forward(self, hs, mass_embeds=None, pred_translations=None, pred_rotations=None):
        """
        Forward pass for force matrix and contact prediction with geometric feature injection
        
        Args:
            hs: [B, N, C] - Object features from PoET decoder
            mass_embeds: [B, N, D_mass] - Optional mass embeddings for each node
            pred_translations: [B, N, 3] - Optional predicted 3D translations for geometric features
            pred_rotations: [B, N, 6] - Optional predicted rotations (currently unused)
        Returns:
            pred_force_matrix: [B, N, N, 3] - Predicted pairwise force matrix
            pred_contact_matrix: [B, N, N, 1] - Predicted pairwise contact probabilities
        """
        B, N, C = hs.shape
        
        # --- START: MODIFICATION - Independent Feature Processing & Fusion ---
        
        # 1. Process Semantic Features
        semantic_edge_embedding = self._compute_semantic_edge_features(hs)  # [B, N, N, semantic_edge_dim]
        
        feature_embeddings = [semantic_edge_embedding]
        
        # 2. Process Geometric Features
        if self.use_geometric_features and pred_translations is not None:
            # Calculate pairwise relative position and distance
            raw_geometric_features = self._compute_geometric_edge_features(pred_translations)  # [B, N, N, 6]
            
            # Reshape for MLP processing: [B, N, N, 6] -> [B*N*N, 6]
            B, N, N_dim, feature_dim = raw_geometric_features.shape
            raw_geometric_features_flat = raw_geometric_features.view(-1, feature_dim)
            
            # Embed the raw geometric features into a well-behaved vector
            geometric_edge_embedding_flat = self.geometric_embedder(raw_geometric_features_flat)  # [B*N*N, hidden_dim//8]
            
            # Reshape back: [B*N*N, hidden_dim//8] -> [B, N, N, hidden_dim//8]
            geometric_edge_embedding = geometric_edge_embedding_flat.view(B, N, N_dim, -1)
            feature_embeddings.append(geometric_edge_embedding)
        
        # 3. Process Mass Features
        if mass_embeds is not None and self.mass_embed_dim is not None:
            # Create pairwise mass embeddings
            pairwise_mass_embeds = self.pairwise_layer(mass_embeds, mass_embeds)  # [B, N, N, mass_embed_dim*2]
            
            # Reshape for MLP processing: [B, N, N, mass_embed_dim*2] -> [B*N*N, mass_embed_dim*2]
            B, N, N_dim, mass_feature_dim = pairwise_mass_embeds.shape
            pairwise_mass_embeds_flat = pairwise_mass_embeds.view(-1, mass_feature_dim)
            
            # Embed the mass features
            mass_edge_embedding_flat = self.mass_pairwise_embedder(pairwise_mass_embeds_flat)  # [B*N*N, hidden_dim//8]
            
            # Reshape back: [B*N*N, hidden_dim//8] -> [B, N, N, hidden_dim//8]
            mass_edge_embedding = mass_edge_embedding_flat.view(B, N, N_dim, -1)
            feature_embeddings.append(mass_edge_embedding)
        
        # 4. Fuse all feature embeddings
        combined_features = torch.cat(feature_embeddings, dim=-1)
        
        # Project the fused features into the Graph Transformer's expected dimension
        e = self.fusion_layer(combined_features)  # This is the final initial edge feature 'e'
        
        # --- END: MODIFICATION ---
        
        # The rest of the forward pass now uses the rich, fused edge features 'e'
        # The node features 'h' can be projected as before
        node_features = hs
        if mass_embeds is not None and self.mass_embed_dim is not None:
            node_features = torch.cat([hs, mass_embeds], dim=-1)
        h = self.proj_node_input(node_features)
        
        # Apply graph transformer layers
        for layer in self.graph_transformer_layers:
            h, e = layer(h, e)
            
        # Predict forces and contacts from final edge features
        pred_force_matrix = self.force_predictor(e)
        pred_contact_matrix = self.contact_predictor(e)
        
        return pred_force_matrix, pred_contact_matrix
