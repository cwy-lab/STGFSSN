import math
import torch
import torch.nn as nn

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        
        super(GraphAttentionLayer, self).__init__()
        self.out_features = out_features
        self.W_q = nn.Linear(in_features, out_features)
        self.W_k = nn.Linear(in_features, out_features)
        self.W_v = nn.Linear(in_features, out_features)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, query_features, key_features):
        """
        query_features: shape (B, num_nodes, in_features)
        key_features: shape (B, num_nodes, in_features)
        """
        Q = self.W_q(query_features)  # (B, num_nodes, out_features)
        K = self.W_k(key_features)    # (B, num_nodes, out_features)
        V = self.W_v(key_features)    # (B, num_nodes, out_features)
        scores = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(self.out_features)
        attn = self.softmax(scores)
        out = torch.matmul(attn, V)   # (B, num_nodes, out_features)
        return out

class MultiHeadGraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads=4):
        super(MultiHeadGraphAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.attn_heads = nn.ModuleList(
            [GraphAttentionLayer(in_features, out_features) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * out_features, out_features)
        self.layer_norm = nn.LayerNorm(out_features)
    
    def forward(self, query_features, key_features):
        
        head_outputs = [head(query_features, key_features) for head in self.attn_heads]
        concatenated = torch.cat(head_outputs, dim=-1)
        out = self.linear(concatenated)
        
        out = self.layer_norm(out + query_features)
        return out

class GraphMambaImproved(nn.Module):
    def __init__(self, num_nodes, seq_len, hidden_dim, forecast_horizon, dropout_rate=0.2, num_heads=4):
        
        super(GraphMambaImproved, self).__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len

        self.main_fc = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(hidden_dim)
        )
        self.graph_attn = MultiHeadGraphAttentionLayer(hidden_dim, hidden_dim, num_heads=num_heads)

        self.state_cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.skip_cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.fusion_fc = nn.Linear(3 * hidden_dim, forecast_horizon)

    def forward(self, x):
        """
        x: (B, seq_len, num_nodes)
        """
        B, L, N = x.shape  # (B, seq_len, num_nodes)

        x_main = x.permute(0, 2, 1)  # (B, num_nodes, seq_len)
        main_features = self.main_fc(x_main)  # (B, num_nodes, hidden_dim)
        main_out = self.graph_attn(main_features, main_features)  # (B, num_nodes, hidden_dim)

        x_fft = torch.fft.rfft(x, dim=1)  # (B, freq_bins, num_nodes)
        x_fft = torch.abs(x_fft)           # (B, freq_bins, num_nodes)
        x_fft = x_fft.permute(0, 2, 1)       # (B, num_nodes, freq_bins)
        B, N, F = x_fft.shape
        x_fft_reshaped = x_fft.reshape(B * N, 1, F)  # (B*N, 1, freq_bins)
        state_features = self.state_cnn(x_fft_reshaped)  # (B*N, hidden_dim, freq_bins)
        state_features = torch.mean(state_features, dim=2)  # (B*N, hidden_dim)
        state_features = state_features.reshape(B, N, -1)     # (B, num_nodes, hidden_dim)

        x_skip = x.permute(0, 2, 1)  # (B, num_nodes, seq_len)
        x_skip_reshaped = x_skip.reshape(B * N, 1, L)  # (B*N, 1, seq_len)
        skip_features = self.skip_cnn(x_skip_reshaped)  # (B*N, hidden_dim, seq_len)
        skip_features = torch.mean(skip_features, dim=2)  # (B*N, hidden_dim)
        skip_features = skip_features.reshape(B, N, -1)     # (B, num_nodes, hidden_dim)

        fused = torch.cat([main_out, state_features, skip_features], dim=-1)  # (B, num_nodes, 3*hidden_dim)
        out = self.fusion_fc(fused)  # (B, num_nodes, forecast_horizon)
        return out
