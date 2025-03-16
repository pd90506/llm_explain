import torch
import torch.nn as nn
import torch.nn.functional as F
from llmexp.explainer.mab_model import MABModel
from typing import Dict, Any, List, Tuple
import transformers
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import random


class MultiLevelInputMapper:
    def __init__(self):
        """Initialize the MultiLevelInputMapper with empty dictionaries for sentences, phrases, and words."""
        self.sentences = {}  # Keys: (a,) where a is sentence index
        self.phrases = {}    # Keys: (a,b) where a is sentence index, b is phrase index
        self.words = {}      # Keys: (a,b,c) where a is sentence index, b is phrase index, c is word index
        
        # Ensure NLTK resources are available
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def phrase_extract(self, sentence: str) -> List[str]:
        """
        Extract phrases from a sentence, keeping delimiters (commas, semicolons, colons).
        
        Args:
            sentence: The input sentence to process
            
        Returns:
            List of phrases with their delimiters
        """
        # Split sentence into phrases while keeping the delimiters
        delimiters_pattern = r'([,;:])'
        split_parts = re.split(delimiters_pattern, sentence)
        
        # Reconstruct phrases with their delimiters
        phrases = []
        i = 0
        while i < len(split_parts):
            if i + 1 < len(split_parts) and split_parts[i+1] in [',', ';', ':']:
                # Combine content with its following delimiter
                phrase = split_parts[i] + split_parts[i+1]
                # Strip spaces but preserve newlines
                phrase = phrase.strip()
                phrases.append(phrase)
                i += 2
            else:
                # Handle the last part if it doesn't have a delimiter
                if split_parts[i].strip():
                    phrase = split_parts[i].strip()
                    phrases.append(phrase)
                i += 1
        
        return [phrase for phrase in phrases if phrase]

    def process_text(self, text: str) -> None:
        """
        Process text into hierarchical levels: sentences, phrases, and words.
        
        Args:
            text: The input text to process
        """
        # Clear previous mappings
        self.sentences = {}
        self.phrases = {}
        self.words = {}
        
        # Split text into sentences
        sentences = sent_tokenize(text)
        
        for sent_idx, sentence in enumerate(sentences):
            # Store the sentence
            self.sentences[(sent_idx,)] = sentence
            
            # Extract phrases using the dedicated function
            phrases = self.phrase_extract(sentence)
            
            for phrase_idx, phrase in enumerate(phrases):
                # Store the phrase
                self.phrases[(sent_idx, phrase_idx)] = phrase
                
                # Split phrase into words
                words = word_tokenize(phrase)
                
                for word_idx, word in enumerate(words):
                    # Store the word
                    self.words[(sent_idx, phrase_idx, word_idx)] = word
    
    def get_content(self, *args) -> str:
        """
        Get content at the specified level based on indices.
        
        Args:
            *args: Variable length argument list:
                - (a,): Return the ath sentence
                - (a,b): Return the bth phrase of the ath sentence
                - (a,b,c): Return the cth word of the bth phrase of the ath sentence
                
        Returns:
            The requested content as a string
        
        Raises:
            KeyError: If the requested indices don't exist
            ValueError: If invalid number of arguments provided
        """
        if len(args) == 1:
            key = (args[0],)
            return self.sentences.get(key, "")
        elif len(args) == 2:
            key = (args[0], args[1])
            return self.phrases.get(key, "")
        elif len(args) == 3:
            key = (args[0], args[1], args[2])
            return self.words.get(key, "")
        else:
            raise ValueError("Invalid number of arguments. Expected 1, 2, or 3 arguments.")

    def sample_sentence(self, p=0.5) -> List[int]:
        """
        Sample sentences with probability p.
        
        Args:
            p: Probability of selecting each sentence (default: 0.5)
            
        Returns:
            List of sampled sentence indices
        """
        sampled_indices = []
        for key in self.sentences.keys():
            if random.random() < p:
                sampled_indices.append(key[0])
        return sampled_indices

    def sample_phrases(self, p=0.5, a=None) -> List[Tuple[int, int]]:
        """
        Sample phrases with probability p.
        
        Args:
            p: Probability of selecting each phrase (default: 0.5)
            a: If specified, only sample phrases from sentence a
            
        Returns:
            List of sampled phrase indices as (sentence_idx, phrase_idx) tuples
        """
        sampled_indices = []
        for key in self.phrases.keys():
            sent_idx, phrase_idx = key
            # If a is specified, only sample from sentence a
            if a is not None and sent_idx != a:
                continue
            
            if random.random() < p:
                sampled_indices.append((sent_idx, phrase_idx))
        return sampled_indices

    def sample_words(self, p=0.5, a=None, b=None) -> List[Tuple[int, int, int]]:
        """
        Sample words with probability p.
        
        Args:
            p: Probability of selecting each word (default: 0.5)
            a: If specified, only sample words from sentence a
            b: If specified, only sample words from phrase b of sentence a
            
        Returns:
            List of sampled word indices as (sentence_idx, phrase_idx, word_idx) tuples
        """
        sampled_indices = []
        for key in self.words.keys():
            sent_idx, phrase_idx, word_idx = key
            # If a is specified, only sample from sentence a
            if a is not None and sent_idx != a:
                continue
            # If b is specified, only sample from phrase b
            if b is not None and phrase_idx != b:
                continue
            
            if random.random() < p:
                sampled_indices.append((sent_idx, phrase_idx, word_idx))
        return sampled_indices


class MABTemplate:
    def __init__(self, tokenizer, instruction: str):
        self.tokenizer = tokenizer
        self.instruction = instruction

    def format(self, query: str) -> str:
        """
        Format the query into a template.
        """
        content = [
                    {"role": "system", 
                    "content": self.instruction
                    },

                    {"role": "user", 
                    "content": query
                    }
                ]
        return self.tokenizer.apply_chat_template(content, tokenize=False, add_generation_prompt=True)


class MABExplainer:
    def __init__(
        self,
        target_model: torch.nn.Module,
        tokenizer: transformers.PreTrainedTokenizer,
        template: MABTemplate,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        # Initialize the target model
        self.target_model = target_model.to(device)

        self.tokenizer = tokenizer
        self.template = template
        self.device = device

        self.input_mapper = MultiLevelInputMapper()

    @torch.no_grad()
    def get_response_logits_mean(self, 
                                input_ids: torch.Tensor, 
                                attention_mask: torch.Tensor, 
                                response_mask: torch.Tensor,
                                normalize: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the predictions of the model with the masks applied.
        """ 
        labels = torch.cat([input_ids[..., 1:], input_ids[..., :1]], dim=-1) # [batch_size, sequence_length]
        response_logits, shifted_response_mask = self.get_response_logits(input_ids, attention_mask, response_mask, normalize)
        # get the logits of the labels
        labels_logits = response_logits.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1) # [batch_size, sequence_length]
        # get the mean of the logits
        labels_logits_mean = labels_logits.sum(dim=-1) / shifted_response_mask.sum(dim=-1)
        return labels_logits_mean
    
    @torch.no_grad()
    def get_response_logits(self, 
                        input_ids: torch.Tensor, 
                        attention_mask: torch.Tensor, 
                        response_mask: torch.Tensor,
                        normalize: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the predictions of the model with the masks applied.
        Args:
            input_ids: [batch_size, sequence_length] The input ids of the model.
            attention_mask: [batch_size, sequence_length] The attention mask of the model.
            response_mask: [batch_size, sequence_length] The response mask of the model.
        Returns:
            [batch_size,] The logit of the last token prediction
        """
        logits_all = self.target_model(input_ids, attention_mask).logits # [batch_size, sequence_length, vocab_size]
        shifted_response_mask = torch.cat([response_mask[..., 1:], response_mask[..., :1]], dim=-1)
        if normalize:
            normalized_logits_all = logits_all - logits_all.mean(dim=-1, keepdim=True)
            response_logits = normalized_logits_all * shifted_response_mask.unsqueeze(-1) # [batch_size, sequence_length, vocab_size]
        else:
            response_logits = logits_all * shifted_response_mask.unsqueeze(-1) # [batch_size, sequence_length, vocab_size]
        return response_logits, shifted_response_mask # [batch_size, sequence_length, vocab_size], [batch_size, sequence_length]
    

    @torch.no_grad()
    def get_response_mask(self, attention_mask: torch.Tensor, 
                          output_attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Get the response mask of the model.
        """
        response_mask = torch.zeros_like(output_attention_mask)
        # pad the attention_mask to the same length as the output_attention_mask
        attention_mask = F.pad(attention_mask, (0, output_attention_mask.size(1) - attention_mask.size(1), 0, 0), value=0)
        response_mask[(attention_mask == 0) & (output_attention_mask == 1)] = 1
        
        return response_mask
    
    
    # @torch.no_grad() 
    # def random_clip_query(self, query: str, num_trials: int = 5):
    #     """
    #     Process the query into hierarchical levels: sentences, phrases, and words.
    #     """
    #     self.input_mapper.process_text(query)
    #     for _ in range(num_trials):
    #         clipped_query_indices = self.input_mapper.sample_phrases(p=0.5)
    #         clipped_query = " ".join([self.input_mapper.get_content(*i) for i in clipped_query_indices])
    #         template = self.template.format(clipped_query)
    #         yield template
    
    @torch.no_grad()
    def random_clip_query_words(self, query: str, response: str, num_trials: int = 5):
        """
        Randomly clip the query into words.
        """
        self.input_mapper.process_text(query)
        for _ in range(num_trials):
            clipped_query_indices = self.input_mapper.sample_words(p=0.5)
            clipped_query = " ".join([self.input_mapper.get_content(*i) for i in clipped_query_indices])
            template = self.template.format(clipped_query)
            clip = f"{template}{response}"
            yield clip
