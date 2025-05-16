from llmexp.utils.hotpot_helper import HotpotSample
from typing import Dict
from datasets import load_dataset, Dataset

class HotpotDataset(Dataset):
    def __init__(self, split='test'):
        """
        Initialize Dataset.
        
        Args:
            dataset (str): Name of the dataset to load
        """
        self.dataset = load_dataset('hotpotqa/hotpot_qa', "fullwiki", split=split)
        
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
        hotpot_sample = HotpotSample(item)
        sentences = hotpot_sample.flattened_contexts
        question = hotpot_sample.question
            
        return {"sentences": sentences, "question": question}


