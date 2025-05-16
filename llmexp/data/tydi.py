from typing import Dict
from datasets import load_dataset, Dataset
import re

def split_into_sentences(text: str) -> list:
    """
    Split text into sentences while preserving punctuation.
    Handles multiple newlines by converting them to single newlines.
    Also handles reference brackets like [3] as part of the sentence.
    
    Args:
        text (str): Input text to split
        
    Returns:
        list: List of sentences
    """
    # First normalize newlines (convert multiple newlines to single newline)
    text = re.sub(r'\n+', '\n', text)
    
    # Split on sentence endings followed by space or newline
    # This pattern looks for .!? followed by optional reference brackets and then space or newline
    sentences = re.split(r'([.!?](?:\[\d+\])?)\s+', text)
    
    # Combine the punctuation (and reference brackets) with the previous sentence
    result = []
    for i in range(0, len(sentences)-1, 2):
        if i+1 < len(sentences):
            result.append(sentences[i] + sentences[i+1])
        else:
            result.append(sentences[i])
    
    # Clean up any remaining whitespace
    result = [s.strip() for s in result if s.strip()]
    
    return result

class TydiDataset(Dataset):
    def __init__(self, split="train"):
        """
        Initialize Dataset.
        
        Args:
            dataset (str): Name of the dataset to load
        """
        self.dataset = load_dataset("google-research-datasets/tydiqa", "primary_task", split='train')
        self.dataset = self.dataset.filter(lambda x: x["language"] == "english")
        
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
        text = item['document_plaintext']
        sentences = split_into_sentences(text)
        
        question = item['question_text']
            
        return {"sentences": sentences, "question": question}