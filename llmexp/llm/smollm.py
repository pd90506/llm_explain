from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
from typing import List


class Template:
    def __init__(self, tokenizer, task='qa'):
        self.tokenizer = tokenizer
        if task == 'qa':
            self.role_system = "You are a helpful assistant that can answer questions concisely based only on the context provided."
        elif task == 'sentiment_analysis':
            self.role_system = "You are a helpful assistant that can analyze the sentiment of a given text. Answer with only a single word, either 'positive' or 'negative'."
    def __call__(self, sentences: List[str], question: str) -> str:
        """
        Format the query into a template.
        """
        query = "Context: " + " ".join(sentences) + "\n\n" + " Query: " + question
        
        messages = [
                    # {"role": "system", 
                    # "content": self.role_system
                    # },
                    {"role": "user", 
                    "content": query 
                    }
                ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


class LLMWrapper(nn.Module):
    def __init__(self, checkpoint=None, device="cuda", access_token=None, task='qa'):
        super().__init__()
        if checkpoint is None:
            checkpoint = "HuggingFaceTB/SmolLM-1.7B-Instruct"

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, token=access_token)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint, token=access_token).to(device)
        self.device = device
        self.template = Template(self.tokenizer, task=task)

        self.hidden_size = self.model.config.hidden_size

    def forward(self, input_ids, attention_mask, **kwargs):
        return self.model(input_ids, attention_mask=attention_mask, **kwargs)

    def get_logits(self, input_ids, attention_mask, **kwargs):
        outputs = self.model.forward(input_ids, attention_mask, **kwargs)
        return outputs.logits

    def get_response(self, sentences: List[str], question: str):
        response_tokens = self.get_response_tokens(sentences, question)
        response = self.tokenizer.decode(response_tokens[0])
        return response
        
    def get_response_tokens(self, sentences: List[str], question: str):
        messages = self.template(sentences, question)
        inputs = self.tokenizer(messages, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        response_output = self.generate(**inputs, max_new_tokens=256)
        # get the response tokens
        response_tokens = response_output['input_ids']
        response_tokens = response_tokens[:, inputs['input_ids'].shape[1]:]
        return response_tokens

    @torch.no_grad()
    def get_response_logits(self, sentences: List[str], question: str, response: str):
        messages = self.template(sentences, question)
        
        input_tokenized = self.tokenizer(messages, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        og_input_ids = input_tokenized.input_ids
        input_attention_mask = input_tokenized.attention_mask

        response_tokenized = self.tokenizer(response, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        response_ids = response_tokenized.input_ids
        response_attention_mask = response_tokenized.attention_mask
        
        # note that the batch size is 1, so they can be concatenated
        input_ids = torch.cat([og_input_ids, response_ids], dim=1)
        attention_mask = torch.cat([input_attention_mask, response_attention_mask], dim=1)

        outputs = self.get_logits(input_ids, attention_mask)
        
        # Extract the logits for the response tokens
        response_logits = outputs[:, og_input_ids.shape[1]-1:-1, :]
        
        return response_logits
    
    @torch.no_grad()
    def get_log_likelihood(self, sentences: List[str], question: str, response: str):
        response_tokens = self.tokenizer(response, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        response_ids = response_tokens.input_ids
        sentence_logits = self.get_response_logits(sentences, question, response)
        log_probs = torch.log_softmax(sentence_logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=response_ids.unsqueeze(-1)).squeeze(-1)
        log_likelihood = token_log_probs.mean()
        return log_likelihood

    @torch.no_grad()
    def generate(self, 
                 input_ids,
                 attention_mask=None,
                 max_new_tokens=50, 
                 temperature=None, 
                 top_p=None, 
                 do_sample=False,
                 **kwargs):
        generated = self.model.generate(input_ids, 
                                    attention_mask=attention_mask,
                                    max_new_tokens=max_new_tokens, 
                                    temperature=temperature, 
                                    top_p=top_p, 
                                    do_sample=do_sample,
                                    pad_token_id=self.tokenizer.pad_token_id)
        
        # # Create attention mask for the full sequence
        # new_attention_mask = torch.zeros_like(generated)
        
        # # Copy original attention mask
        # new_attention_mask[:, :input_ids.shape[1]] = attention_mask
        
        # # Set 1s for new tokens, except for padding and after EOS
        # new_tokens = generated[:, input_ids.shape[1]:]
        # new_tokens_mask = torch.ones_like(new_tokens)
        # is_eos = (new_tokens == self.tokenizer.eos_token_id)
        # is_pad = (new_tokens == self.tokenizer.pad_token_id)
        
        # # Create mask for positions after EOS
        # eos_indices = is_eos.float().cumsum(dim=1)
        # after_eos = eos_indices > 0
        
        # new_tokens_mask[after_eos | is_pad] = 0
        # new_attention_mask[:, input_ids.shape[1]:] = new_tokens_mask
        
        return {
            'input_ids': generated,
            # 'attention_mask': new_attention_mask
        }

    def generate_from_texts(self, 
                 messages,
                 max_new_tokens=50, 
                 temperature=None, 
                 top_p=None, 
                 do_sample=False):
        input_text = messages

        inputs = self.tokenizer(
            input_text,
            truncation=True,
            max_length=2048,
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
