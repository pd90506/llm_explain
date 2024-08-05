import torch.nn as nn 
from models import MLP
import torch.nn.functional as F
import torch

def similarity_measure(logits, labels, attention_mask):
    """ 
    args:
        logis: torch.Tensor, shape (batch_size, seq_len, vocab_size)
        labels: torch.Tensor, shape (batch_size, seq_len)
        attention_mask: torch.Tensor, shape (batch_size, seq_len) the original input text is masked with 0

    return:
        mean_log_probs: torch.Tensor, shape (batch_size,)
    """
    log_probs = torch.log_softmax(logits, dim=-1)

    # Flatten the tensors to simplify indexing
    batch_size, seq_len, vocab_size = logits.size()
    labels = labels.view(-1)  # flatten to (batch_size * seq_len)
    log_probs_flat = log_probs.view(-1, vocab_size)

    # Extract the log probabilities for the correct labels
    correct_log_probs_flat = log_probs_flat[range(batch_size * seq_len), labels]

    # Reshape to (batch_size, seq_len)
    correct_log_probs = correct_log_probs_flat.view(batch_size, seq_len)

    # Mask out the original input texts, only keep the generated tokens
    masked_log_probs = correct_log_probs * attention_mask

    # Calculate the mean log probability for each sequence
    mean_log_probs = masked_log_probs.sum(dim=-1) / attention_mask.sum(dim=-1)

    return mean_log_probs


class MaskGeneratingModel(nn.Module):
    def __init__(self, hidden_size=4096, mlp_hidden_dim=1024, mlp_bottleneck_dim=768, mlp_num_blocks=2):
        """ 
        hidden_size: int
            The hidden size of the output of the generative model, 4096 for llama3
        """
        super().__init__()

        self.hidden_size = hidden_size

        self.explain_map = MLP(input_dim=hidden_size, 
                               hidden_dim=mlp_hidden_dim, 
                               output_dim=1, 
                               num_blocks=mlp_num_blocks, 
                               bottleneck_dim=mlp_bottleneck_dim) # takes [N, L, hidden_size] outputs [N, L, 1]
    
    def forward(self, pred_features):
        """ 
        pred_features: torch.Tensor of shape [N, L, hidden_size]
        """
        mask_logits = self.explain_map(pred_features) # [N, L, 1]
        return mask_logits.squeeze(-1) # [batch_size, seq_len]
    
    @torch.no_grad()
    def generate_mask(self, mask_logits, context_mask):
        """
        generate mask based on a Bernoulli distribution with the probabilities of the given logits 
        args:
            mask_logits: torch.Tensor, shape (batch_size, seq_len)
            context_mask: torch.Tensor, shape (batch_size, seq_len), the context texts 
                (for example, the chatbot instruction template, not including the user inputs) are masked with 0.
                This mask strategy is to ensure we only focus on the real user inputs.
        return:
            mask: torch.Tensor, shape (batch_size, seq_len), with contexts being all 1s and user inputs being randomly masked.
        """
        mask_probs = torch.sigmoid(mask_logits) # (batch_size, seq_len)
        mask = torch.bernoulli(mask_probs) # (batch_size, seq_len)
        mask = context_mask * mask + (1. - context_mask) # (batch_size, seq_len)
        return mask # this could be used as the attention mask for the generative model
    
    @torch.no_grad()
    def sample_one_batch(self, input_ids, attention_mask, mask_logits, context_mask):
        """ 
        args:
            input_ids: torch.Tensor, shape (batch_size, seq_len)
            attention_mask: torch.Tensor, shape (batch_size, seq_len), the attention_mask generated automatically by tokenizer, usually all 1s
            mask_logits: torch.Tensor, shape (batch_size, prompt_len), the logits for the mask generation
            context_mask: torch.Tensor, shape (batch_size, prompt_len), the context texts 
                (for example, the chatbot instruction template, not including the user inputs) are masked with 0.
        """
        #  get the mask of interest, i.e., for the user inputs, then pad to match the shape of other masks (of the full length)
        seq_len = input_ids.size(1)
        prompt_len = mask_logits.size(1)
        mask = self.generate_mask(mask_logits=mask_logits, context_mask=context_mask) # (batch_size, prompt_len)
        pad_length = seq_len - prompt_len
        padded_mask = F.pad(mask, (0, pad_length), mode='constant', value=1)

        # get the masked attention_mask
        masked_attention_mask = attention_mask * padded_mask

        return input_ids, masked_attention_mask, mask
    
    def compute_similarity(self, logits, labels, attention_mask):
        """ 
        compute the mean log_probs for the generated tokens
        """
        mean_log_probs = similarity_measure(logits, labels, attention_mask)
        return mean_log_probs
