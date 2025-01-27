from datasets import load_dataset, Dataset
from typing import Tuple, Dict, List, Any
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase


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
    

class DataCollator:
    """Collate function for batching examples."""
    def __init__(self, tokenizer: Any, max_length: int = 512, instruction: str = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction = instruction

        # Set pad token to eos token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    
    def format_example(self, example: str, max_length: int = 512) -> str:
        # Truncate the input text to fit within token limit
        truncated_input = self.tokenizer.decode(
            self.tokenizer.encode(example, truncation=True, max_length=max_length),  
            skip_special_tokens=True
        )
        
        if self.instruction is not None:
            instruction = self.instruction
        else:
            instruction = "Analyze the sentiment of the following sentence and respond with only one word: 'positive,' 'negative,' or 'neutral,' based on the overall tone and meaning of the sentence. Do not provide any additional explanation."
        content = [
            {"role": "system", 
             "content": instruction
            },
            {"role": "user", 
             "content": truncated_input
            }
        ]
        return self.tokenizer.apply_chat_template(content, tokenize=False, add_generation_prompt=True)
    
    def __call__(self, examples: List[Dict]) -> Dict:
        """
        Tokenize and batch examples.
        
        Args:
            examples: List of dictionaries containing 'sentence' and optionally 'label'
            
        Returns:
            Dict containing batched tensors
        """
        # Extract sentence and labels
        prompts = [self.format_example(example['sentence'], max_length=self.max_length) for example in examples]
        
        # Tokenize all prompts in the batch
        batch = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # Add labels if they exist
        if 'label' in examples[0]:
            batch['labels'] = torch.tensor([example['label'] for example in examples])
            
        return batch
    

def create_dataloader(
    dataset,
    tokenizer: PreTrainedTokenizerBase,
    instruction: str = None,
    batch_size: int = 32,
    shuffle: bool = True,
    max_length: int = 512
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
    llm_dataset = LLMDataset(dataset)
    collator = DataCollator(tokenizer=tokenizer, max_length=max_length, instruction=instruction)
    
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
    