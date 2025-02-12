from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
class LLMWrapper(nn.Module):
    def __init__(self, checkpoint=None, device="cuda"):
        super().__init__()
        if checkpoint is None:
            checkpoint = "HuggingFaceTB/SmolLM-1.7B-Instruct"

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
        self.device = device

        self.hidden_size = self.model.config.hidden_size

    def forward(self, input_ids, attention_mask, **kwargs):
        return self.model(input_ids, attention_mask=attention_mask, **kwargs)

    def get_logits(self, input_ids, attention_mask, **kwargs):
        outputs = self.model.forward(input_ids, attention_mask, **kwargs)
        return outputs.logits
    
    def generate(self, 
                 input_ids,
                 attention_mask,
                 max_new_tokens=50, 
                 temperature=0.2, 
                 top_p=0.9, 
                 do_sample=True,
                 **kwargs):
        generated = self.model.generate(input_ids, 
                                    attention_mask=attention_mask,
                                    max_new_tokens=max_new_tokens, 
                                    temperature=temperature, 
                                    top_p=top_p, 
                                    do_sample=do_sample,
                                    pad_token_id=self.tokenizer.pad_token_id)
        
        # Create attention mask for the full sequence
        new_attention_mask = torch.zeros_like(generated)
        
        # Copy original attention mask
        new_attention_mask[:, :input_ids.shape[1]] = attention_mask
        
        # Set 1s for new tokens, except for padding and after EOS
        new_tokens = generated[:, input_ids.shape[1]:]
        new_tokens_mask = torch.ones_like(new_tokens)
        is_eos = (new_tokens == self.tokenizer.eos_token_id)
        is_pad = (new_tokens == self.tokenizer.pad_token_id)
        
        # Create mask for positions after EOS
        eos_indices = is_eos.float().cumsum(dim=1)
        after_eos = eos_indices > 0
        
        new_tokens_mask[after_eos | is_pad] = 0
        new_attention_mask[:, input_ids.shape[1]:] = new_tokens_mask
        
        return {
            'input_ids': generated,
            'attention_mask': new_attention_mask
        }

    def generate_from_texts(self, 
                 messages,
                 max_new_tokens=50, 
                 temperature=0.2, 
                 top_p=0.9, 
                 do_sample=True):
        input_text = messages

        inputs = self.tokenizer(
            input_text,
            truncation=True,
            max_length=512,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.device)
        
        outputs = self.generate(**inputs, 
                                      max_new_tokens=max_new_tokens, 
                                      temperature=temperature, 
                                      top_p=top_p, 
                                      do_sample=do_sample)

        return self.tokenizer.decode(outputs['input_ids'][0])

    def get_hidden_states(self, input_ids, attention_mask, **kwargs):
        """
        input_ids: torch.Tensor, shape [batch_size, sequence_length]
        attention_mask: torch.Tensor, shape [batch_size, sequence_length]
        """
        # Get model outputs with output_hidden_states=True
        with torch.no_grad():  # No need for gradients
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
        # Get last hidden state
        last_hidden_state = outputs.hidden_states[-1]  # Shape: [batch_size, sequence_length, hidden_size]
        
        return last_hidden_state
