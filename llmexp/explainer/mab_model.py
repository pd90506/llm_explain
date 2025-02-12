import torch 
import torch.nn as nn 
import torch.nn.functional as F
from llmexp.llm.smollm import LLMWrapper


def scale_to_0_1(x, dim=-1):
    x_min = x.min(dim=dim, keepdim=True).values
    x_max = x.max(dim=dim, keepdim=True).values
    return (x - x_min) / (x_max - x_min + 1e-8)  # Adding epsilon to avoid division by zero


class MABModel(nn.Module):
    """
    Multi-Armed Bandit Model for explanation.
    """
    def __init__(self, base_model:LLMWrapper, hidden_size, freeze_base=True):
        super().__init__()

        # Register base_model as a buffer without saving it
        self._register_base_model(base_model)
        self.hidden_size = hidden_size  # Save for reconstruction

        # Actor network for policy generation
        self.prompt_actor = nn.Sequential(
            nn.Linear(self.base_model.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), # output a vector of length hidden_size for each prompt token
        )  # append an element-wise product between the last token and the every input token, the append an MLP to get the final attribution score

        self.generated_actor = nn.Sequential(
            nn.Linear(self.base_model.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), # output a vector of length hidden_size for the last token
        )

        self.critic = nn.Sequential(
            nn.Linear(self.base_model.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        self.actor_head = nn.Linear(hidden_size, 1)


        # Freeze the base model if required
        if freeze_base:
            self._freeze_base_model()
    

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        """Forward pass of the Multi-Armed Bandit Model.

        input_ids: 
                torch.Tensor, shape [batch_size, sequence_length], 
                the left padded input_ids with the last token being the generated token
        attention_mask: 
                torch.Tensor, shape [batch_size, sequence_length], 
                the attention_mask, left padded with 0s
        """
        
        hidden_states = self.base_model.get_hidden_states(input_ids, attention_mask)

        prompt_hidden_states = hidden_states[:, :-1, :] # [batch_size, sequence_length-1, hidden_size]
        prompt_hidden_states = self.prompt_actor(prompt_hidden_states) # [batch_size, sequence_length-1, hidden_size]

        generated_hidden_state = hidden_states[:, -1, :].unsqueeze(1) # [batch_size, 1, hidden_size]
        generated_hidden_state = self.generated_actor(generated_hidden_state) # [batch_size, 1, hidden_size]

        # Hadamard product between the generated hidden state and the prompt hidden states
        correlated_hidden_states = generated_hidden_state * prompt_hidden_states # [batch_size, sequence_length-1, hidden_size]
        logits = self.actor_head(correlated_hidden_states).squeeze(-1) # [batch_size, sequence_length-1]

        value = self.critic(hidden_states[:, -1, :]) # [batch_size, 1]

        return logits, value
    
    def get_logits_value(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        """
        Get the logits and the MAB values.
        """
        logits, value = self.forward(input_ids, attention_mask)

        return logits, value # logits: [batch_size, sequence_length-1], value: [batch_size, 1]

    # Public methods that users will commonly use
    def state_dict(self, *args, **kwargs):
        """Public method for saving model."""
        state_dict = super().state_dict(*args, **kwargs)
        # Remove base_model related keys
        keys_to_remove = [k for k in state_dict.keys() if k.startswith('base_model')]
        for k in keys_to_remove:
            del state_dict[k]
        return state_dict

    @classmethod
    def load_with_base_model(cls, state_dict, base_model, hidden_size=None):
        """Public class method for loading model."""
        model = cls(base_model, hidden_size)
        model.load_state_dict(state_dict, strict=False)
        return model

    # Private utility methods at the end
    def _register_base_model(self, base_model):
        """Private utility method for internal use."""
        self.register_buffer('base_model', torch.tensor([]), persistent=False)
        self.base_model = base_model

    def _freeze_base_model(self):
        """Private utility method for internal use."""
        for param in self.base_model.parameters():
            param.requires_grad = False