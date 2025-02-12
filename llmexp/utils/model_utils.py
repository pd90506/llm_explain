import torch
import torch.nn.functional as F


class GumbelKHotDistribution:
    def __init__(self, logits, context_mask, k, temperature=1.0):
        """
        logits: [batch_size, N] unnormalized log probabilities
        context_mask: [batch_size, N] binary mask (0 for masked positions)
        k: number of elements to select
        temperature: temperature parameter for Gumbel-Softmax
        """
        self.logits = logits
        self.context_mask = context_mask
        
        # Apply mask by setting masked logits to large negative value
        self.masked_logits = logits.masked_fill(~context_mask.bool(), float('-inf'))
        self.k = k
        self.temperature = temperature
        
    def sample(self):
        """Sample k-hot vectors using Gumbel-Softmax trick"""
        # Add Gumbel noise
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(self.logits) + 1e-10) + 1e-10)
        noisy_logits = (self.logits + gumbel_noise) / self.temperature
        
        # Apply mask
        masked_noisy_logits = noisy_logits.masked_fill(~self.context_mask.bool(), float('-inf'))
        
        # Get top-k values
        _, top_k_indices = torch.topk(masked_noisy_logits, self.k, dim=-1)
        
        # Convert to k-hot
        k_hot = torch.zeros_like(self.logits)
        k_hot.scatter_(1, top_k_indices, 1)
        
        # Ensure we only select valid positions
        k_hot = k_hot * self.context_mask
        
        return k_hot
    
    def log_prob(self, k_hot):
        """
        Compute log probability of a k-hot configuration
        k_hot: [batch_size, N] binary tensor with exactly k 1s per row
        """
        # Ensure k_hot only selects valid positions
        # k_hot = k_hot * self.context_mask
        
        selected_logits = self.logits * k_hot
        log_numerator = selected_logits.sum(dim=-1)  # sum of selected logits
        
        # Compute log of normalization constant only over valid positions
        log_denominator = torch.logsumexp(self.masked_logits, dim=-1)  # [batch_size]
        
        return log_numerator - log_denominator
    
    def entropy(self):
        """
        Compute entropy of the k-hot distribution over valid positions
        Returns entropy per batch element
        """
        # Compute softmax probabilities from masked logits
        probs = F.softmax(self.masked_logits, dim=-1)
        
        # Compute entropy only for valid positions
        element_entropy = -(probs * torch.log(probs + 1e-10))  # [batch_size, N]
        total_entropy = (element_entropy * self.context_mask).sum(dim=-1)  # [batch_size]
        
        # Scale by k/N where N is the number of valid positions
        n_valid = self.context_mask.sum(dim=-1)
        scaled_entropy = total_entropy #* (self.k / (n_valid + 1e-10))
        
        return scaled_entropy
