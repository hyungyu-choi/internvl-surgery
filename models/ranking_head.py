import torch
import torch.nn as nn


class RankingHead(nn.Module):
    def __init__(self, hidden_dim, mlp_dim=1024, num_heads=8, dtype=torch.bfloat16):
        super().__init__()
        self.dtype = dtype
        
        self.mlp1 = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.LayerNorm(mlp_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_dim, mlp_dim)
        ).to(dtype)
        
        self.self_attn = nn.MultiheadAttention(
            embed_dim=mlp_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        ).to(dtype)
        
        self.attn_norm = nn.LayerNorm(mlp_dim).to(dtype)
        
        self.mlp2 = nn.Sequential(
            nn.Linear(mlp_dim, mlp_dim // 2),
            nn.LayerNorm(mlp_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_dim // 2, 1)
        ).to(dtype)
    
    def forward(self, embeddings):
        """
        Args:
            embeddings: [B, K, hidden_dim]
        Returns:
            scores: [B, K]
        """
        x = self.mlp1(embeddings)
        attn_out, _ = self.self_attn(x, x, x)
        x = self.attn_norm(x + attn_out)
        scores = self.mlp2(x).squeeze(-1)
        return scores