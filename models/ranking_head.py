import torch
import torch.nn as nn


class RankingHead(nn.Module):
    def __init__(self, hidden_dim=3584, mlp_dim=512, num_heads=8):
        super().__init__()
        
        self.mlp1 = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim)
        )
        
        self.self_attn = nn.MultiheadAttention(
            embed_dim=mlp_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.mlp2 = nn.Sequential(
            nn.Linear(mlp_dim, mlp_dim // 2),
            nn.ReLU(),
            nn.Linear(mlp_dim // 2, 1)
        )
    
    def forward(self, embeddings):
        """
        Args:
            embeddings: [batch_size, K, hidden_dim]
        Returns:
            scores: [batch_size, K]
        """
        x = self.mlp1(embeddings)
        print(x.shape)
        x, _ = self.self_attn(x, x, x)
        scores = self.mlp2(x).squeeze(-1)
        return scores