import torch
import torch.nn as nn


class RankingHead(nn.Module):
    def __init__(self, hidden_dim=3584, mlp_dim=1024, num_heads=8):
        super().__init__()
        
        # 더 큰 mlp_dim (512 -> 1024)
        self.mlp1 = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.LayerNorm(mlp_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_dim, mlp_dim)
        )
        
        self.self_attn = nn.MultiheadAttention(
            embed_dim=mlp_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        self.attn_norm = nn.LayerNorm(mlp_dim)
        
        self.mlp2 = nn.Sequential(
            nn.Linear(mlp_dim, mlp_dim // 2),
            nn.LayerNorm(mlp_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_dim // 2, 1)
        )
    
    def forward(self, embeddings):
        # MLP1 with residual
        x = self.mlp1(embeddings)
        
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x)
        x = self.attn_norm(x + attn_out)  # Residual
        
        # MLP2
        scores = self.mlp2(x).squeeze(-1)
        return scores