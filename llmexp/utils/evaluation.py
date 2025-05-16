import torch
from scipy.stats import spearmanr
import numpy as np

def calculate_avg_log_prob_diff(log_prob1: torch.Tensor, log_prob2: torch.Tensor) -> torch.Tensor:
    """
    Calculate the average log probability difference between two log probability tensors.
    
    Args:
        log_prob1 (torch.Tensor): First log probability tensor of shape (1, N) or (N,)
        log_prob2 (torch.Tensor): Second log probability tensor of shape (1, N) or (N,)
        
    Returns:
        torch.Tensor: Average log probability difference
        
    Raises:
        ValueError: If shapes don't match or tensors have invalid shapes
    """
    # Ensure both tensors are on the same device
    if log_prob1.device != log_prob2.device:
        log_prob2 = log_prob2.to(log_prob1.device)
    
    # Handle different input shapes
    if log_prob1.dim() == 2 and log_prob1.shape[0] == 1:
        log_prob1 = log_prob1.squeeze(0)
    if log_prob2.dim() == 2 and log_prob2.shape[0] == 1:
        log_prob2 = log_prob2.squeeze(0)
    
    # Verify shapes match
    if log_prob1.shape != log_prob2.shape:
        raise ValueError(f"Shape mismatch: log_prob1 {log_prob1.shape} != log_prob2 {log_prob2.shape}")
    
    # Calculate difference and mean
    diff = log_prob1 - log_prob2
    avg_diff = diff.mean()
    
    return avg_diff.item()

def calculate_bertscore(pred: str, ref: str) -> float:
    """
    Calculate BERTScore between prediction and reference strings.
    
    Args:
        pred (str): Prediction text
        ref (str): Reference text
        
    Returns:
        float: BERTScore F1 score between prediction and reference
    """
    from bert_score import score
    
    # Calculate BERTScore
    P, R, F1 = score([pred], [ref], lang='en', rescale_with_baseline=True)
    
    # Return the F1 score (harmonic mean of precision and recall)
    return F1.item()
