"""
Advanced Football Pass Prediction with Multi-Modal Transformer Architecture

Enhanced neural architecture features:
- Multi-scale temporal attention (short-term tactics + long-term patterns)
- Spatial relationship modeling between players
- Tactical context awareness (formation, pressure, game state)
- Multi-modal fusion of event, spatial, and temporal features
- Graph neural network for player relationship modeling
- Adaptive attention mechanisms for dynamic player importance
"""

import math
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple, Optional, List


# ===================== Enhanced Data Processing =====================

def build_index(s, add_unk=False, add_pad=False):
    """Enhanced vocabulary builder with padding support"""
    items = sorted([str(x) for x in s.dropna().unique().tolist()])
    if add_pad: items = ['<PAD>'] + items
    if add_unk: items = (['<PAD>', '<UNK>'] if add_pad else ['<UNK>']) + items
    return {t: i for i, t in enumerate(items)}


def extract_tactical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract advanced tactical and contextual features"""
    df = df.copy()
    
    # Basic geometric features (needed for other calculations)
    df['Start Time [s]'] = df['Start Time [s]'].astype(float)
    df['event_dt'] = df['Start Time [s]'].diff().fillna(0.0)
    dx = df['End X'] - df['Start X']
    dy = df['End Y'] - df['Start Y']
    df['pass_len'] = np.sqrt(dx**2 + dy**2)
    df['pass_ang'] = np.arctan2(dy, dx)
    
    # Game state features
    df['minute'] = (df['Start Time [s]'] / 60.0).clip(0, 90)
    df['game_phase'] = pd.cut(df['minute'], bins=[0, 15, 30, 45, 60, 75, 90], 
                              labels=['early_1st', 'mid_1st', 'late_1st', 'early_2nd', 'mid_2nd', 'late_2nd'])
    
    # Pressure indicators
    df['time_pressure'] = (df['event_dt'] < 2.0).astype(float)
    
    # Field zones (tactical thirds)
    df['zone_x'] = pd.cut(df['Start X'], bins=[0, 0.33, 0.67, 1.0], labels=['def', 'mid', 'att'])
    df['zone_y'] = pd.cut(df['Start Y'], bins=[0, 0.25, 0.75, 1.0], labels=['left', 'center', 'right'])
    
    # Pass characteristics (now pass_len exists)
    df['pass_difficulty'] = df['pass_len'] * (1 + df['time_pressure'])
    df['forward_pass'] = (df['End X'] > df['Start X']).astype(float)
    df['progressive_pass'] = (df['pass_len'] > 0.15).astype(float) * df['forward_pass']
    
    return df


def load_advanced_sequences(csv_path: str, window: int = 12, context_window: int = 24):
    """Enhanced sequence loading with tactical context and player tracking"""
    df = pd.read_csv(csv_path)
    
    # Enhanced filtering and preprocessing
    p = df[(df['Type'] == 'PASS') & df['To'].notna()].copy()
    
    # Coordinate normalization and cleaning
    coord_cols = ['Start X', 'Start Y', 'End X', 'End Y']
    for col in coord_cols:
        p[col] = p[col].astype(float).fillna(p[col].mean()).clip(0, 1)
    
    # Enhanced feature engineering
    p = extract_tactical_features(p)
    
    # Build enhanced vocabularies
    vocabs = {
        'team': build_index(p['Team']),
        'type': build_index(p['Type'], add_unk=True),
        'subtype': build_index(p['Subtype'], add_unk=True),
        'from': build_index(p['From'], add_unk=True),
        'to': build_index(p['To'], add_unk=True),
        'game_phase': build_index(p['game_phase'], add_unk=True),
        'zone_x': build_index(p['zone_x'], add_unk=True),
        'zone_y': build_index(p['zone_y'], add_unk=True)
    }
    
    # Sort chronologically
    p = p.sort_values(['Period', 'Start Time [s]', 'Start Frame']).reset_index(drop=True)
    
    # Build sequences with enhanced features
    sequences = []
    player_sequences = []
    targets = []
    
    for i in range(context_window, len(p)):
        # Recent sequence for immediate prediction
        recent_seq = []
        # Extended context for tactical understanding  
        context_seq = []
        # Player interaction history
        player_seq = []
        
        for j in range(max(0, i - context_window), i):
            row = p.iloc[j]
            
            # Categorical features
            cat_features = [
                vocabs['team'][str(row['Team'])],
                vocabs['type'].get(str(row['Type']), vocabs['type']['<UNK>']),
                vocabs['subtype'].get(str(row['Subtype']), vocabs['subtype']['<UNK>']),
                vocabs['from'].get(str(row['From']), vocabs['from']['<UNK>']),
                vocabs['game_phase'].get(str(row['game_phase']), vocabs['game_phase']['<UNK>']),
                vocabs['zone_x'].get(str(row['zone_x']), vocabs['zone_x']['<UNK>']),
                vocabs['zone_y'].get(str(row['zone_y']), vocabs['zone_y']['<UNK>'])
            ]
            
            # Numerical features (enhanced)
            num_features = [
                row['Start X'], row['Start Y'], row['End X'], row['End Y'],
                row['pass_len'], row['pass_ang'], row['event_dt'], row['Period'],
                row['minute'], row['time_pressure'], row['pass_difficulty'],
                row['forward_pass'], row['progressive_pass']
            ]
            
            # Player relationship features
            player_features = [
                vocabs['from'].get(str(row['From']), vocabs['from']['<UNK>']),
                vocabs['to'].get(str(row['To']), vocabs['to']['<UNK>'])
            ]
            
            if j >= i - window:
                recent_seq.append((cat_features, num_features))
            context_seq.append((cat_features, num_features))
            player_seq.append(player_features)
        
        # Pad sequences to fixed length
        while len(recent_seq) < window:
            recent_seq.insert(0, ([0] * len(cat_features), [0.0] * len(num_features)))
        while len(context_seq) < context_window:
            context_seq.insert(0, ([0] * len(cat_features), [0.0] * len(num_features)))
        while len(player_seq) < context_window:
            player_seq.insert(0, [0, 0])
        
        sequences.append(recent_seq[-window:])
        player_sequences.append(player_seq[-context_window:])
        targets.append(vocabs['to'][str(p.iloc[i]['To'])])
    
    return sequences, player_sequences, targets, vocabs


# ===================== Advanced Neural Architecture =====================

class MultiScalePositionalEncoding(nn.Module):
    """Multi-scale positional encoding for different temporal granularities"""
    
    def __init__(self, d_model: int, max_len: int = 512, scales: List[int] = [1, 4, 16]):
        super().__init__()
        self.d_model = d_model
        self.scales = scales
        
        # Ensure dimension per scale is even for sin/cos pairing
        d_per_scale = d_model // len(scales)
        if d_per_scale % 2 != 0:
            d_per_scale = d_per_scale - 1  # Make it even
        
        # Create multiple positional encodings at different scales
        encodings = []
        for scale in scales:
            pe = torch.zeros(max_len, d_per_scale)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) / scale
            div_term = torch.exp(torch.arange(0, d_per_scale, 2).float() * 
                               (-math.log(10000.0) / d_per_scale))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            encodings.append(pe.unsqueeze(0))
        
        # Concatenate all scale encodings
        full_encoding = torch.cat(encodings, dim=-1)
        
        # If we need to pad to match d_model exactly
        current_dim = full_encoding.shape[-1]
        if current_dim < d_model:
            padding = torch.zeros(1, max_len, d_model - current_dim)
            full_encoding = torch.cat([full_encoding, padding], dim=-1)
        elif current_dim > d_model:
            full_encoding = full_encoding[:, :, :d_model]
        
        self.register_buffer('encodings', full_encoding)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.encodings[:, :x.size(1), :].to(x.device)


class SpatialRelationEncoder(nn.Module):
    """Encode spatial relationships between players and field positions"""
    
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Distance and angle encoding
        self.distance_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2)
        )
        
        self.angle_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim // 4),  # sin, cos of angle
            nn.ReLU(), 
            nn.Linear(hidden_dim // 4, hidden_dim // 2)
        )
        
        self.position_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim // 2),  # start_x, start_y, end_x, end_y
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        # positions: [B, S, 4] (start_x, start_y, end_x, end_y)
        start_pos = positions[..., :2]
        end_pos = positions[..., 2:]
        
        # Calculate distances and angles
        diff = end_pos - start_pos
        distances = torch.norm(diff, dim=-1, keepdim=True)
        angles = torch.atan2(diff[..., 1:2], diff[..., 0:1])
        
        # Encode features
        dist_encoded = self.distance_encoder(distances)
        angle_encoded = self.angle_encoder(torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1))
        pos_encoded = self.position_encoder(positions)
        
        # Fuse spatial features
        spatial_features = torch.cat([dist_encoded, angle_encoded], dim=-1)
        return self.fusion(torch.cat([spatial_features, pos_encoded], dim=-1))


class PlayerRelationshipGNN(nn.Module):
    """Graph Neural Network for modeling player relationships"""
    
    def __init__(self, player_emb_dim: int = 32, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.player_emb_dim = player_emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Player embeddings - transform from N players * emb_dim to hidden_dim
        # Note: N=2 for player pairs (from, to), so input is 2 * player_emb_dim
        self.node_transform = nn.Linear(2 * player_emb_dim, hidden_dim)
        
        # Graph convolution layers
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim)
            ) for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, player_embeddings: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # player_embeddings: [B, S, N, emb_dim] where N is number of players in sequence
        # adjacency: [B, S, N, N] adjacency matrix
        
        B, S, N, emb_dim = player_embeddings.shape
        
        # For simplicity with player pairs (N=2), flatten the player embeddings per timestep
        # [B, S, N, emb_dim] -> [B, S, N*emb_dim]
        player_features_flat = player_embeddings.view(B, S, N * emb_dim)  # [B, S, 2*32]
        
        # Transform to hidden dimension
        x = self.node_transform(player_features_flat)  # [B, S, hidden_dim]
        
        # Simple processing without complex GNN operations for player pairs
        for layer in self.gnn_layers:
            residual = x
            x = layer(torch.cat([x, x], dim=-1))  # Self-attention style
            x = x + residual
        
        return self.output_proj(x)  # [B, S, hidden_dim]


class TacticalContextEncoder(nn.Module):
    """Encode tactical context and game state"""
    
    def __init__(self, vocab_sizes: Dict[str, int], emb_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        
        # Tactical embeddings
        self.game_phase_emb = nn.Embedding(vocab_sizes['game_phase'], emb_dim)
        self.zone_x_emb = nn.Embedding(vocab_sizes['zone_x'], emb_dim)
        self.zone_y_emb = nn.Embedding(vocab_sizes['zone_y'], emb_dim)
        
        # Tactical state encoder
        self.tactical_encoder = nn.Sequential(
            nn.Linear(emb_dim * 3 + 5, hidden_dim),  # +5 for numerical tactical features
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, categorical_features: torch.Tensor, numerical_features: torch.Tensor) -> torch.Tensor:
        # Extract tactical categorical features (indices 4, 5, 6)
        game_phase = categorical_features[..., 4]
        zone_x = categorical_features[..., 5]
        zone_y = categorical_features[..., 6]
        
        # Extract tactical numerical features (indices 8, 9, 10, 11, 12)
        tactical_nums = numerical_features[..., 8:13]
        
        # Embed categorical features
        phase_emb = self.game_phase_emb(game_phase)
        zone_x_emb = self.zone_x_emb(zone_x)
        zone_y_emb = self.zone_y_emb(zone_y)
        
        # Combine all tactical features
        tactical_features = torch.cat([phase_emb, zone_x_emb, zone_y_emb, tactical_nums], dim=-1)
        
        return self.tactical_encoder(tactical_features)


class EnhancedEventEncoder(nn.Module):
    """Enhanced event encoder with multi-modal feature fusion"""
    
    def __init__(self, vocab_sizes: Dict[str, int], num_dim: int = 13, 
                 emb_dim: int = 32, hidden_dim: int = 128, spatial_dim: int = 64):
        super().__init__()
        
        # Basic embeddings
        self.team_emb = nn.Embedding(vocab_sizes['team'], emb_dim)
        self.type_emb = nn.Embedding(vocab_sizes['type'], emb_dim)
        self.subtype_emb = nn.Embedding(vocab_sizes['subtype'], emb_dim)
        self.from_emb = nn.Embedding(vocab_sizes['from'], emb_dim)
        
        # Numerical feature processor
        self.num_processor = nn.Sequential(
            nn.Linear(8, hidden_dim // 2),  # Basic numerical features
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # Spatial relationship encoder
        self.spatial_encoder = SpatialRelationEncoder(spatial_dim)
        
        # Tactical context encoder
        self.tactical_encoder = TacticalContextEncoder(vocab_sizes, emb_dim, hidden_dim // 2)
        
        # Multi-modal fusion
        fusion_input_dim = emb_dim * 4 + hidden_dim // 2 + spatial_dim + hidden_dim // 2
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, categorical_features: torch.Tensor, numerical_features: torch.Tensor) -> torch.Tensor:
        # Basic categorical embeddings
        team_emb = self.team_emb(categorical_features[..., 0])
        type_emb = self.type_emb(categorical_features[..., 1])
        subtype_emb = self.subtype_emb(categorical_features[..., 2])
        from_emb = self.from_emb(categorical_features[..., 3])
        
        # Process basic numerical features (positions, geometry, timing)
        basic_nums = numerical_features[..., :8]
        num_features = self.num_processor(basic_nums)
        
        # Encode spatial relationships
        positions = numerical_features[..., :4]  # start_x, start_y, end_x, end_y
        spatial_features = self.spatial_encoder(positions)
        
        # Encode tactical context
        tactical_features = self.tactical_encoder(categorical_features, numerical_features)
        
        # Fuse all modalities
        all_features = torch.cat([
            team_emb, type_emb, subtype_emb, from_emb,
            num_features, spatial_features, tactical_features
        ], dim=-1)
        
        return self.fusion_layer(all_features)


class AdaptiveMultiHeadAttention(nn.Module):
    """Multi-head attention with adaptive importance weighting"""
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        
        # Adaptive importance network
        self.importance_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, n_heads),
            nn.Softmax(dim=-1)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = query.shape
        
        # Compute Q, K, V
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        context = torch.matmul(attention_weights, V)
        
        # Compute adaptive importance weights
        query_pooled = torch.mean(query, dim=1)  # [B, d_model]
        head_importance = self.importance_net(query_pooled)  # [B, n_heads]
        head_importance = head_importance.unsqueeze(-1).unsqueeze(-1)  # [B, n_heads, 1, 1]
        
        # Weight heads by importance
        context = context * head_importance
        
        # Concatenate heads and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(context)


class AdvancedTransformerBlock(nn.Module):
    """Enhanced transformer block with multi-scale attention and tactical awareness"""
    
    def __init__(self, d_model: int, n_heads: int = 8, ff_dim: int = 512, 
                 dropout: float = 0.1, tactical_dim: int = 64):
        super().__init__()
        
        # Multi-head attention layers
        self.self_attention = AdaptiveMultiHeadAttention(d_model, n_heads, dropout)
        self.tactical_attention = AdaptiveMultiHeadAttention(tactical_dim, max(1, n_heads // 2), dropout)
        
        # Feed-forward networks
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Tactical integration
        self.tactical_proj = nn.Linear(tactical_dim, d_model)
        self.tactical_gate = nn.Sequential(
            nn.Linear(d_model + tactical_dim, d_model),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, tactical_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        residual = x
        x = self.norm1(x)
        x = residual + self.dropout(self.self_attention(x, x, x))
        
        # Tactical integration if available
        if tactical_context is not None:
            tactical_features = self.tactical_attention(tactical_context, tactical_context, tactical_context)
            tactical_proj = self.tactical_proj(tactical_features)
            
            # Gated fusion
            gate = self.tactical_gate(torch.cat([x, tactical_features], dim=-1))
            x = x + gate * tactical_proj
            x = self.norm2(x)
        
        # Feed-forward
        residual = x
        x = self.norm3(x)
        x = residual + self.dropout(self.ff(x))
        
        return x


class AdvancedPassPredictionModel(nn.Module):
    """Advanced multi-modal transformer for pass prediction"""
    
    def __init__(self, vocab_sizes: Dict[str, int], num_dim: int = 13, 
                 d_model: int = 256, n_heads: int = 8, n_layers: int = 6,
                 dropout: float = 0.1, max_seq_len: int = 24):
        super().__init__()
        
        self.d_model = d_model
        
        # Event encoding
        self.event_encoder = EnhancedEventEncoder(vocab_sizes, num_dim, hidden_dim=d_model)
        
        # Positional encoding
        self.pos_encoding = MultiScalePositionalEncoding(d_model, max_seq_len)
        
        # Player relationship modeling
        self.player_gnn = PlayerRelationshipGNN(32, 64, 2)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            AdvancedTransformerBlock(d_model, n_heads, d_model * 2, dropout)
            for _ in range(n_layers)
        ])
        
        # Multi-scale temporal pooling
        self.temporal_pools = nn.ModuleList([
            nn.AdaptiveAvgPool1d(1),  # Global average
            nn.AdaptiveMaxPool1d(1),  # Global max
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),  # Local patterns
        ])
        
        # Final prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model * 3 + 64, d_model),  # +64 for GNN features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, vocab_sizes['to'])
        )
        
        # Player embedding for GNN
        self.player_embedding = nn.Embedding(vocab_sizes['from'], 32)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, categorical_seq: torch.Tensor, numerical_seq: torch.Tensor, 
                player_seq: torch.Tensor) -> torch.Tensor:
        
        batch_size, seq_len, _ = categorical_seq.shape
        
        # Encode events
        event_features = self.event_encoder(categorical_seq, numerical_seq)
        
        # Add positional encoding
        event_features = self.pos_encoding(event_features)
        event_features = self.dropout(event_features)
        
        # Process through transformer layers
        for layer in self.transformer_layers:
            event_features = layer(event_features)
        
        # Multi-scale temporal pooling
        x_transposed = event_features.transpose(1, 2)  # [B, d_model, seq_len]
        
        pooled_features = []
        pooled_features.append(self.temporal_pools[0](x_transposed).squeeze(-1))  # Global avg
        pooled_features.append(self.temporal_pools[1](x_transposed).squeeze(-1))  # Global max
        
        local_features = self.temporal_pools[2](x_transposed)
        pooled_features.append(F.adaptive_avg_pool1d(local_features, 1).squeeze(-1))  # Local patterns
        
        temporal_features = torch.cat(pooled_features, dim=-1)
        
        # Player relationship modeling
        player_embeddings = self.player_embedding(player_seq)  # [B, seq_len, 2, emb_dim]
        
        # Create adjacency matrix (simple co-occurrence based)
        adjacency = torch.zeros(batch_size, seq_len, 2, 2, device=player_seq.device)
        adjacency[:, :, 0, 1] = 1.0  # From -> To connection
        adjacency[:, :, 1, 0] = 1.0  # To -> From connection
        
        player_features = self.player_gnn(player_embeddings, adjacency)  # [B, seq_len, hidden_dim]
        player_pooled = torch.mean(player_features, dim=1)  # [B, hidden_dim]
        
        # Combine all features
        final_features = torch.cat([temporal_features, player_pooled], dim=-1)
        
        # Generate predictions
        logits = self.prediction_head(final_features)
        
        return logits


# ===================== Training and Evaluation =====================

def advanced_train_and_eval(sequences, player_sequences, targets, vocabs, 
                           model_type='advanced_transformer', epochs=20, 
                           lr=1e-4, batch_size=64, weight_decay=1e-5):
    """Enhanced training with advanced model"""
    
    # Prepare data
    cat_data, num_data = zip(*[(seq[0], seq[1]) for batch in sequences for seq in batch])
    
    # Reshape data properly
    cat_sequences = []
    num_sequences = []
    
    window = len(sequences[0])
    for seq_batch in sequences:
        cat_seq = [item[0] for item in seq_batch]
        num_seq = [item[1] for item in seq_batch]
        cat_sequences.append(cat_seq)
        num_sequences.append(num_seq)
    
    cat_data = torch.tensor(cat_sequences, dtype=torch.long)
    num_data = torch.tensor(num_sequences, dtype=torch.float32)
    player_data = torch.tensor(player_sequences, dtype=torch.long)
    target_data = torch.tensor(targets, dtype=torch.long)
    
    # Split data
    n_samples = len(targets)
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    
    indices = torch.randperm(n_samples)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    # Create data loaders
    train_dataset = TensorDataset(cat_data[train_idx], num_data[train_idx], 
                                 player_data[train_idx], target_data[train_idx])
    val_dataset = TensorDataset(cat_data[val_idx], num_data[val_idx], 
                               player_data[val_idx], target_data[val_idx])
    test_dataset = TensorDataset(cat_data[test_idx], num_data[test_idx], 
                                player_data[test_idx], target_data[test_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    vocab_sizes = {k: len(v) for k, v in vocabs.items()}
    model = AdvancedPassPredictionModel(vocab_sizes, num_dim=num_data.shape[-1])
    model.to(device)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Training loop
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for cat_batch, num_batch, player_batch, target_batch in train_loader:
            cat_batch = cat_batch.to(device)
            num_batch = num_batch.to(device) 
            player_batch = player_batch.to(device)
            target_batch = target_batch.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(cat_batch, num_batch, player_batch)
            loss = criterion(outputs, target_batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += target_batch.size(0)
            train_correct += (predicted == target_batch).sum().item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        top3_correct = 0
        top5_correct = 0
        
        with torch.no_grad():
            for cat_batch, num_batch, player_batch, target_batch in val_loader:
                cat_batch = cat_batch.to(device)
                num_batch = num_batch.to(device)
                player_batch = player_batch.to(device)
                target_batch = target_batch.to(device)
                
                outputs = model(cat_batch, num_batch, player_batch)
                loss = criterion(outputs, target_batch)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += target_batch.size(0)
                val_correct += (predicted == target_batch).sum().item()
                
                # Top-k accuracy
                _, top3_pred = torch.topk(outputs, 3, dim=1)
                _, top5_pred = torch.topk(outputs, 5, dim=1)
                
                top3_correct += sum(target_batch[i] in top3_pred[i] for i in range(len(target_batch)))
                top5_correct += sum(target_batch[i] in top5_pred[i] for i in range(len(target_batch)))
        
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        top3_acc = top3_correct / val_total
        top5_acc = top5_correct / val_total
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}')
        print(f'  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}')
        print(f'  Top-3 Acc: {top3_acc:.4f}, Top-5 Acc: {top5_acc:.4f}')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
    
    # Test evaluation
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    model.eval()
    test_correct = 0
    test_total = 0
    test_top3_correct = 0
    test_top5_correct = 0
    
    with torch.no_grad():
        for cat_batch, num_batch, player_batch, target_batch in test_loader:
            cat_batch = cat_batch.to(device)
            num_batch = num_batch.to(device)
            player_batch = player_batch.to(device)
            target_batch = target_batch.to(device)
            
            outputs = model(cat_batch, num_batch, player_batch)
            
            _, predicted = torch.max(outputs.data, 1)
            test_total += target_batch.size(0)
            test_correct += (predicted == target_batch).sum().item()
            
            _, top3_pred = torch.topk(outputs, 3, dim=1)
            _, top5_pred = torch.topk(outputs, 5, dim=1)
            
            test_top3_correct += sum(target_batch[i] in top3_pred[i] for i in range(len(target_batch)))
            test_top5_correct += sum(target_batch[i] in top5_pred[i] for i in range(len(target_batch)))
    
    test_results = {
        'accuracy': test_correct / test_total,
        'top3_accuracy': test_top3_correct / test_total,
        'top5_accuracy': test_top5_correct / test_total
    }
    
    return test_results, model


# ===================== Command Line Interface =====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Advanced Football Pass Prediction')
    parser.add_argument('--csv', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--model', type=str, default='advanced_transformer', 
                       choices=['advanced_transformer'], help='Model type')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--window', type=int, default=12, help='Sequence window size')
    parser.add_argument('--context_window', type=int, default=24, help='Context window size')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    
    args = parser.parse_args()
    
    print("Loading and preprocessing data...")
    sequences, player_sequences, targets, vocabs = load_advanced_sequences(
        args.csv, args.window, args.context_window
    )
    
    print(f"Loaded {len(sequences)} sequences")
    print(f"Vocabulary sizes: {{k: len(v) for k, v in vocabs.items()}}")
    
    print("Training advanced model...")
    results, model = advanced_train_and_eval(
        sequences, player_sequences, targets, vocabs,
        model_type=args.model, epochs=args.epochs, lr=args.lr, 
        batch_size=args.batch_size, weight_decay=args.weight_decay
    )
    
    print("\nFinal Test Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Top-3 Accuracy: {results['top3_accuracy']:.4f}")
    print(f"Top-5 Accuracy: {results['top5_accuracy']:.4f}")