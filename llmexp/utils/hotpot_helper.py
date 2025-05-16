from typing import Dict
import numpy as np

class HotpotSample:
    def __init__(self, example: Dict):
        self.contexts = example["context"]["sentences"]
        self.supporting_facts = example["supporting_facts"]
        self.titles = example["context"]["title"]
        self.question = example["question"]
        self.answer = example["answer"]
        
        self.indices_tuple_list = self.process_indices()
        self.flattened_contexts = self.flatten_contexts()
        self.flattened_supporting_facts_indices = self.flatten_supporting_facts_indices()
        
        self.sentence_mask = self.get_sentence_mask()
        
    def get_contexts_from_mask(self, mask: np.ndarray):
        """
        Get the contexts from the mask.
        """
        contexts = self.flattened_contexts
        selected_contexts = []
        for context, mask_val in zip(contexts, mask):
            if mask_val == 1:
                selected_contexts.append(context)
        return selected_contexts
    
    def process_indices(self):
        """
        Process the indices of the contexts, return a list of tuples of the form (title_idx, sent_idx).
        The flattened indices can be found by `self.indices_tuple_list.index((title_idx, sent_idx))`.
        """
        contexts = self.contexts
        indices_tuple_list = []
        for idx, context in enumerate(contexts):
            for j, sent in enumerate(context):
                indices_tuple_list.append((idx, j))
        return indices_tuple_list
    
    def flatten_contexts(self):
        """
        Flatten the contexts into a single list of sentences.
        """
        contexts = self.contexts
        flattened_contexts = []
        for context in contexts:
            flattened_contexts.extend(context)
        return flattened_contexts
    
    def flatten_supporting_facts_indices(self):
        """ 
        Convert supporting facts indices to a list of flattened indices.
        """
        supporting_facts = self.supporting_facts
        flattened_supporting_facts_indices = []
        titles = supporting_facts["title"]
        sent_ids = supporting_facts["sent_id"]
        for title, sent_id in zip(titles, sent_ids):
            title_idx = self.titles.index(title)
            tuple_idx = (title_idx, sent_id)
            flattened_index = self.indices_tuple_list.index(tuple_idx)
            flattened_supporting_facts_indices.append(flattened_index)
        return flattened_supporting_facts_indices
    
    def get_sentence_mask(self):
        """
        Get the sentence mask for the contexts.
        """
        sentence_mask = [1] * len(self.flattened_contexts)
        return np.array(sentence_mask, dtype=int)
    



class HotpotHelper:
    def __init__(self):
        pass

    def get_supporting_facts_indices(self, example: Dict):
        titles = example["context"]["title"]
        supporting_facts = example["supporting_facts"]
        supporting_facts_title_indices = [titles.index(t) for t in supporting_facts["title"]]
        return list(zip(supporting_facts_title_indices, supporting_facts["sent_id"]))

    def get_iter_indices(self, example: Dict):
        ind_tuples = []
        iter_indices = []
        cur_idx = 0
        for a, context_list in enumerate(example["context"]["sentences"]):
            for b, _ in enumerate(context_list):
                ind_tuples.append((a, b))
                iter_indices.append(cur_idx)
                cur_idx += 1
        ind_map = dict(zip(ind_tuples, iter_indices))
        return ind_map
    
