from datasets import load_dataset, Dataset
from typing import Tuple, Dict, List, Any
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase, BatchEncoding
import torch.nn.functional as F
from abc import ABC, abstractmethod


def load_sst2(split: str = "train") -> Tuple[List[str], List[int]]:
    """
    Load the SST-2 dataset.
    
    Args:
        split (str): Dataset split to load ('train', 'validation', or 'test')
        
    Returns:
        Tuple[List[str], List[int]]: Tuple containing:
            - List of sentences
            - List of labels (0: negative, 1: positive)
    """
    # Load dataset
    dataset = load_dataset("sst2", split=split)
    
    # Extract sentences and labels
    sentences = dataset["sentence"]
    labels = dataset["label"]
    
    return sentences, labels


class LLMDataset(Dataset):
    def __init__(self, dataset: str, split="train"):
        """
        Initialize LLMDataset.
        
        Args:
            dataset (str): Name of the dataset to load
        """
        self.dataset_name = dataset
        if dataset == "sst2":
            self.dataset = load_dataset(dataset, split=split)
        
        elif dataset == "hotpot_qa":
            self.dataset = load_dataset('hotpotqa/hotpot_qa', 'fullwiki')
            self.dataset = self.dataset[split]
        
        else:
            raise ValueError(f"Dataset {dataset} not supported")
        
    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.dataset)
    
    def __getitem__(self, idx) -> Dict:
        """
        Get a single item from the dataset.
        
        Args:
            idx (int): Index of the item
            
        Returns:
            Dict: Dictionary containing the sentence and label (if available)
        """
        item = self.dataset[idx]
            
        return item
    

class DataCollator(ABC):
    """Abstract base class for data collators."""
    
    def __init__(self, tokenizer: Any, max_length: int = 512, instruction=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction = instruction
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
    
    @abstractmethod
    def _create_prompt_parts(self, user_input: str) -> Tuple[str, str, str]:
        """Return (prefix, user_content, suffix) with explicit markers"""
        pass
    
    @abstractmethod
    def __call__(self, examples: List[Dict]) -> BatchEncoding:
        """Process and batch examples"""
        pass


class DataCollatorSST2(DataCollator):
    def __init__(self, tokenizer: Any, max_length: int = 512, instruction=None):
        super().__init__(tokenizer, max_length, instruction)
        if instruction is None:
            self.instruction = "Analyze the sentiment of the following sentence and respond with only one word: 'positive,' 'negative,' or 'neutral,' based on the overall tone and meaning of the sentence. Do not provide any additional explanation."
        else:
            self.instruction = instruction

        self.begin_marker = "<|begin_of_text|>"
        self.system_markers = ("<|start_header_id|>system<|end_header_id|>", "<|eot_id|>")
        self.time_marker = "\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n"
        self.user_markers = ("<|start_header_id|>sentence<|end_header_id|>", "<|eot_id|>")
        self.prompt_marker = "<|start_header_id|>assistant<|end_header_id|>"

    def _create_prompt_parts(self, user_input: str) -> Tuple[str, str, str]:
        """Return (prefix, user_content, suffix) with explicit markers"""
        prefix = self.begin_marker + self.system_markers[0] + self.time_marker + self.instruction + self.system_markers[1] 
        user_prefix = self.user_markers[0] + "\n\n"
        user_suffix = self.user_markers[1]
        suffix = self.prompt_marker
        return prefix + user_prefix, user_input, user_suffix + suffix

    def __call__(self, examples: List[Dict]) -> BatchEncoding:
        """
        Tokenize and batch examples.
        
        Args:
            examples: List of dictionaries containing 'sentence' and optionally 'label'
            
        Returns:
            BatchEncoding: BatchEncoding containing batched tensors
        """
        # Process each example
        input_ids_list = []
        attention_masks_list = []
        context_masks_list = []

        for example in examples:
            prefix, user_input, suffix = self._create_prompt_parts(example['sentence'])

            # Tokenize each part separately
            prefix_tokens = self.tokenizer.encode(
                prefix, 
                add_special_tokens=False, 
                truncation=False
            )
            
            suffix_tokens = self.tokenizer.encode(
                suffix,
                add_special_tokens=False,
                truncation=False
            )

            user_input_tokens = self.tokenizer.encode(
                user_input,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length - len(prefix_tokens) - len(suffix_tokens)
            )
        
            # Combine tokens
            full_tokens = prefix_tokens + user_input_tokens + suffix_tokens
            input_ids = torch.tensor(full_tokens[:self.max_length]) # truncate to max length or simply return the full tokens
            
            # Create context mask
            context_mask = torch.zeros(len(full_tokens), dtype=torch.long)
            user_start = len(prefix_tokens)
            user_end = user_start + len(user_input_tokens)
            context_mask[user_start:user_end] = 1
            
            # Create attention mask
            attention_mask = torch.ones_like(input_ids).long().to(input_ids.device)
            
            input_ids_list.append(input_ids)
            attention_masks_list.append(attention_mask)
            context_masks_list.append(context_mask)
        
        # left pad the sequences to the max length
        batch_max_length = max(len(input_ids) for input_ids in input_ids_list)
        for i in range(len(input_ids_list)):
            input_ids_list[i] = F.pad(input_ids_list[i], (batch_max_length - len(input_ids_list[i]), 0), value=self.tokenizer.pad_token_id)
            attention_masks_list[i] = F.pad(attention_masks_list[i], (batch_max_length - len(attention_masks_list[i]), 0), value=0)
            context_masks_list[i] = F.pad(context_masks_list[i], (batch_max_length - len(context_masks_list[i]), 0), value=0)

        # Convert to batches
        batch = {
            'input_ids': torch.stack(input_ids_list),
            'attention_mask': torch.stack(attention_masks_list),
            'context_mask': torch.stack(context_masks_list)
        }
        
        if 'label' in examples[0]:
            batch['labels'] = torch.tensor([ex['label'] for ex in examples])
            
        return BatchEncoding(batch)


class DataCollatorHotpotQA(DataCollator):
    def __init__(self, tokenizer: Any, max_length: int = 512, instruction=None):
        super().__init__(tokenizer, max_length, instruction)
        if instruction is None:
            self.instruction = "Answer the question based on the context provided."
        else:
            self.instruction = instruction

        self.begin_marker = "<|begin_of_text|>"
        self.system_markers = ("<|start_header_id|>system<|end_header_id|>", "<|eot_id|>")
        # self.time_marker = "\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n"
        self.time_marker = "\n\n"
        self.user_markers = ("<|start_header_id|>user<|end_header_id|>", "<|eot_id|>")
        self.prompt_marker = "<|start_header_id|>assistant<|end_header_id|>"

    def _create_prompt_parts(self, user_input: str) -> Tuple[str, str, str]:
        """Return (prefix, user_content, suffix) with explicit markers"""
        prefix = self.begin_marker + self.system_markers[0] + self.time_marker + self.instruction + self.system_markers[1] 
        user_prefix = self.user_markers[0] + "\n\n"
        user_suffix = self.user_markers[1]
        suffix = self.prompt_marker
        return prefix + user_prefix, user_input, user_suffix + suffix
    
    def _create_user_input(self, sample: Dict) -> str:
        """Create the user input for the prompt"""
        titles = sample["context"]["title"]
        contexts = sample["context"]["sentences"]
        question = sample["question"]
        answer = sample["answer"]

        # Build the Context Sentences section.
        contexts_str = ""
        for title, sentences in zip(titles, contexts):
            # Join the list of sentences into one string.
            sentence_str = " ".join(sentences)
            contexts_str += f"\n- *{title}:* \"{sentence_str}\""
        
        context_str = f"Context Sentences:\n{contexts_str}"
        question_str = f"Question: {question}"

        return context_str + "\n\n" + question_str + "\n\n"

    def __call__(self, examples: List[Dict]) -> BatchEncoding:
        """
        Tokenize and batch examples.
        
        Args:
            examples: List of dictionaries containing 'sentence' and optionally 'label'
            
        Returns:
            BatchEncoding: BatchEncoding containing batched tensors
        """
        # Process each example
        input_ids_list = []
        attention_masks_list = []
        context_masks_list = []

        for example in examples:
            user_input = self._create_user_input(example)
            prefix, user_input, suffix = self._create_prompt_parts(user_input)

            # Tokenize each part separately
            prefix_tokens = self.tokenizer.encode(
                prefix, 
                add_special_tokens=False, 
                truncation=False
            )
            
            suffix_tokens = self.tokenizer.encode(
                suffix,
                add_special_tokens=False,
                truncation=False
            )

            user_input_tokens = self.tokenizer.encode(
                user_input,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length - len(prefix_tokens) - len(suffix_tokens)
            )
        
            # Combine tokens
            full_tokens = prefix_tokens + user_input_tokens + suffix_tokens
            input_ids = torch.tensor(full_tokens[:self.max_length]) # truncate to max length or simply return the full tokens
            
            # Create context mask
            context_mask = torch.zeros(len(full_tokens), dtype=torch.long)
            user_start = len(prefix_tokens)
            user_end = user_start + len(user_input_tokens)
            context_mask[user_start:user_end] = 1
            
            # Create attention mask
            attention_mask = torch.ones_like(input_ids).long().to(input_ids.device)
            
            input_ids_list.append(input_ids)
            attention_masks_list.append(attention_mask)
            context_masks_list.append(context_mask)
        
        # left pad the sequences to the max length
        batch_max_length = max(len(input_ids) for input_ids in input_ids_list)
        for i in range(len(input_ids_list)):
            input_ids_list[i] = F.pad(input_ids_list[i], (batch_max_length - len(input_ids_list[i]), 0), value=self.tokenizer.pad_token_id)
            attention_masks_list[i] = F.pad(attention_masks_list[i], (batch_max_length - len(attention_masks_list[i]), 0), value=0)
            context_masks_list[i] = F.pad(context_masks_list[i], (batch_max_length - len(context_masks_list[i]), 0), value=0)

        # Convert to batches
        batch = {
            'input_ids': torch.stack(input_ids_list),
            'attention_mask': torch.stack(attention_masks_list),
            'context_mask': torch.stack(context_masks_list)
        }
        
        if 'label' in examples[0]:
            batch['labels'] = torch.tensor([ex['label'] for ex in examples])
            
        return BatchEncoding(batch)
    

def create_dataloader(
    dataset,
    tokenizer: PreTrainedTokenizerBase,
    instruction: str = None,
    batch_size: int = 32,
    shuffle: bool = True,
    max_length: int = 512,
    split: str = None
) -> DataLoader:
    """
    Create a DataLoader for the LLM dataset.
    
    Args:
        dataset: Hugging Face dataset
        tokenizer: Model tokenizer
        batch_size: Size of each batch
        shuffle: Whether to shuffle the data
        max_length: Maximum sequence length
        
    Returns:
        DataLoader: PyTorch DataLoader
    """
    # Create dataset and collator
    llm_dataset = LLMDataset(dataset, split=split)
    if dataset == "sst2":
        collator = DataCollatorSST2(tokenizer=tokenizer, max_length=max_length, instruction=instruction)
    elif dataset == "hotpot_qa":
        collator = DataCollatorHotpotQA(tokenizer=tokenizer, max_length=max_length, instruction=instruction)
    else:
        raise ValueError(f"Dataset {dataset} not supported")
    
    # Create dataloader
    dataloader = DataLoader(
        llm_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator
    )
    
    return dataloader

# Usage example:
"""
# Load dataset
dataset = load_dataset("sst2", split="train")

# Create dataloader
train_dataloader = create_dataloader(
    dataset=dataset,
    tokenizer=model.tokenizer,
    batch_size=32,
    shuffle=True
)

# Iterate over batches
for batch in train_dataloader:
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['labels']
    # Your training/inference code here
"""
    