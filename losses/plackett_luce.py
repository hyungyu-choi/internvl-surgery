import torch
import torch.nn as nn


class PlackettLuceLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, scores, ground_truth_order):
        """
        Args:
            scores: [B, K] predicted scores for K items
            ground_truth_order: [B, K] ground truth permutation (0 to K-1)
        Returns:
            loss: scalar
        """
        batch_size, K = scores.shape
        loss = 0.0
        
        for b in range(batch_size):
            print(scores)
            s = scores[b]
            print(s)
            r_star = ground_truth_order[b]
            
            log_prob = 0.0
            for i in range(K):
                item_idx = r_star[i]
                remaining_indices = r_star[i:]
                
                numerator = torch.exp(s[item_idx])
                denominator = torch.sum(torch.exp(s[remaining_indices]))
                
                log_prob += torch.log(numerator / denominator)
            
            loss -= log_prob
        
        return loss / batch_size