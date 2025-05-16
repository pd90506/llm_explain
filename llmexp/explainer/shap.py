import torch
import numpy as np
from typing import List
import shap
from llmexp.llm.smollm import LLMWrapper

class SHAPExplainer:
    def __init__(self, model: LLMWrapper, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def attribute(self, sentences: List[str], query: str, response: str, num_ablations: int = 20):
        """
        Attribute importance to sentences in the context using SHAP values.
        
        Args:
            sentences: List of sentences to analyze
            query: The query/question
            response: The model's response to evaluate
            num_ablations: Number of background samples to use for SHAP
            
        Returns:
            sources: List of sentences from the context
            attributions: SHAP values for each sentence
        """
        # Create background samples by randomly removing sentences
        zero_mask = np.zeros(len(sentences)).reshape(1, -1)
        
        # Create explainer
        def model_fn(masks):
            # Convert list of sentence lists to list of log likelihoods
            log_likelihoods = []
            for mask in masks:
                mask = mask.reshape(-1).tolist()
                sentences_to_keep = [sent for sent, m in zip(sentences, mask) if m]
                log_likelihood = self.model.get_log_likelihood(sentences_to_keep, query, response)
                log_likelihoods.append(log_likelihood.item())
            return np.array(log_likelihoods)
            # mask = mask.reshape(-1).tolist()
            # sentences_to_keep = [sent for sent, m in zip(sentences, mask) if m]
            # log_likelihood = self.model.get_log_likelihood(sentences_to_keep, query, response)
            # # Return a 2D array with shape (1, 1) for single output
            # return np.array([[log_likelihood.item()]])
            
        explainer = shap.KernelExplainer(model_fn, zero_mask)
        
        # Create a mask of all ones for the full input
        full_mask = np.ones(len(sentences))
        
        # Get SHAP values for all sentences
        shap_values = explainer.shap_values(full_mask, nsamples=num_ablations)
        
        # Normalize SHAP values to sum to 1
        attributions = np.array(shap_values)
        attributions = attributions / np.sum(np.abs(attributions))
        
        return sentences, attributions
