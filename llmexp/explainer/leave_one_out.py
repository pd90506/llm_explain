import torch
import os
import json
import numpy as np
from typing import List, Tuple, Optional

class LeaveOneOutWrapper:
    def __init__(self, model, tokenizer, device):
        """
        Initialize the Leave-One-Out attribution wrapper.
        
        Args:
            model: The LLM model
            tokenizer: The tokenizer for the model
            device: The device to run the model on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device


    def attribute(self, sentences: List[str], query: str, response: str, num_samples: int = 1) -> Tuple[List[str], List[float]]:
        """
        Perform leave-one-out attribution on the sentences with respect to the query.
        
        Args:
            sentences: List of input sentences
            query: The query text
            num_samples: Number of samples to use for averaging (default: 1)
            
        Returns:
            Tuple of (sources, attributions) where sources are the sentences and
            attributions are their importance scores
        """
        # Get baseline output with full context
        baseline_log_likelihood = self.model.get_log_likelihood(sentences, query, response)
        
        # Calculate importance scores for each sentence
        importance_scores = []
        
        for sentence in sentences:
            # Create context without the current sentence
            remaining_sentences = [s for s in sentences if s != sentence]

            # Get output with ablated context
            ablated_log_likelihood = self.model.get_log_likelihood(remaining_sentences, query, response)
            
            # Calculate importance as the difference in logits
            importance = torch.abs(baseline_log_likelihood - ablated_log_likelihood).mean().item()
            importance_scores.append(importance)
        
        # Normalize importance scores
        importance_scores = np.array(importance_scores)
        if importance_scores.sum() > 0:
            importance_scores = importance_scores / importance_scores.sum()
        
        return sentences, importance_scores.tolist()
