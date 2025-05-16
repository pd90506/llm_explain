from typing import Dict
from datasets import load_dataset, Dataset
import re

class SST2Dataset(Dataset):
    def __init__(self, split="train"):
        """
        Initialize Dataset.
        
        Args:
            dataset (str): Name of the dataset to load
        """
        self.dataset = load_dataset('sst2', split=split)
        
    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.dataset)
    
    def __getitem__(self, idx) -> Dict:
        """
        Get a single item from the dataset.
        
        Args:
            idx (int): Index of the item
            
        Returns:
            Dict: Dictionary containing the tokenized text and label (if available)
        """
        item = self.dataset[idx]
        # Split text into words and punctuation using regex
        # This pattern matches either words or individual punctuation marks
        words = re.findall(r'\w+|[^\w\s]', item['sentence'])
        label = item['label']
            
        return {"words": words, "label": label}