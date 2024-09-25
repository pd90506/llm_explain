import pytest
import torch
from transformers import AutoTokenizer
from llmexp.helper import DataHelper, template_fn
from datasets import load_dataset


def test_template_fn():
    messages = {
        'context': ['The quick brown fox jumps over the lazy dog.'],
        'user_message': ['What does the fox do?']
    }
    results = template_fn(messages)
    assert len(results['result']) == 1
    assert results['result'][0] == "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nThe quick brown fox jumps over the lazy dog.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat does the fox do?<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    assert results['start'][0] == 157
    assert results['end'][0] == 178


class TestDataHelper:
    @classmethod
    def setup_class(cls):
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        cls.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
        cls.tokenizer.pad_token = cls.tokenizer.eos_token
        cls.data_helper = DataHelper(cls.tokenizer)
        ds = load_dataset("imdb")

        cls.imdb_examples = ds['train'].select(range(3))

        cls.sst2_examples = load_dataset("stanfordnlp/sst2")['train'].select(range(3))

        cls.squad_examples = load_dataset("rajpurkar/squad")['train'].select(range(3))
    
    def test_collate_fn_imdb(self):
        batch = self.data_helper.collate_fn_imdb(self.imdb_examples)
        
        # Assert the shapes of the tensors
        assert batch['input_ids'].shape[0] == len(self.imdb_examples['label'])
        
        # Optionally, check specific values or properties
        decoded_texts = [self.tokenizer.decode(ids, skip_special_tokens=False) for ids in batch['input_ids']]
        expected_text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" \
                        "You are a chatbot for sentiment analysis. You can help users with their questions via concise responses of POSITIVE or NEGATIVE." \
                        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" \
                        'I rented I AM CURIOUS-YELLOW from my video store because of all the controversy that surrounded it when it was first released in 1967. I also heard that at first it was seized by U.S. customs if it ever tried to enter this country, therefore being a fan of films considered "controversial" I really had to see this for myself.<br /><br />The plot is centered around a young Swedish drama student named Lena who wants to learn everything she can about life. In particular she wants to focus her attentions to making some sort of documentary on what the average Swede thought about certain political issues such as the Vietnam War and race issues in the United States. In between asking politicians and ordinary denizens of Stockholm about their opinions on politics, she has sex with her drama teacher, classmates, and married men.<br /><br />What kills me about I AM CURIOUS-YELLOW is that 40 years ago, this was considered pornographic. Really, the sex and nudity scenes are few and far between, even then it\'s not shot like some cheaply made porno. While my countrymen mind find it shocking, in reality sex and nudity are a major staple in Swedish cinema. Even Ingmar Bergman, arguably their answer to good old boy John Ford, had sex scenes in his films.<br /><br />I do commend the filmmakers for the fact that any sex shown in the film is shown for artistic purposes rather than just to shock people and make money to be shown in pornographic theaters in America. I AM CURIOUS-YELLOW is a good film for anyone wanting to study the meat and potatoes (no pun intended) of Swedish cinema. But really, this film doesn\'t have much of a plot.' \
                        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        assert decoded_texts[0].endswith(expected_text)
    
    def test_collate_fn_sst2(self):
        
        batch = self.data_helper.collate_fn_sst2(self.sst2_examples)
        
        # Assert the shapes of the tensors
        assert batch['input_ids'].shape[0] == len(self.sst2_examples['idx'])
        
        # Optionally, check specific values or properties
        decoded_texts = [self.tokenizer.decode(ids, skip_special_tokens=False) for ids in batch['input_ids']]
        expected_text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" \
                        "You are a chatbot for sentiment analysis. You can help users with their questions via concise responses of POSITIVE or NEGATIVE." \
                        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" \
                        "hide new secretions from the parental units " \
                        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        assert decoded_texts[0].endswith(expected_text)
    
    def test_collate_fn_squad(self):
        
        batch = self.data_helper.collate_fn_squad(self.squad_examples)
        
        # Assert the shapes of the tensors
        assert batch['input_ids'].shape[0] == len(self.squad_examples['id'])
        
        # Optionally, check specific values or properties
        decoded_texts = [self.tokenizer.decode(ids, skip_special_tokens=False) for ids in batch['input_ids']]
        expected_text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" \
                        "You are a chatbot for answering questions. You can help users with their questions via concise responses." \
                        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" \
                        'Question: To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?\nContext: Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.' \
                        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        assert decoded_texts[0].endswith(expected_text)


    def test_collate_fn_generic(self):
        examples = self.sst2_examples
        text_key = 'sentence'
        sys_context = (
            "You are a chatbot for testing purposes. "
            "You can help users with their questions via concise responses."
        )
        batch = self.data_helper._collate_fn_generic(examples, text_key, sys_context)
        
        # Assert the shapes of the tensors
        assert batch['input_ids'].shape[0] == len(examples[text_key])
        
        # Optionally, check specific values or properties
        decoded_texts = [self.tokenizer.decode(ids, skip_special_tokens=False) for ids in batch['input_ids']]
        expected_text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" \
                        "You are a chatbot for testing purposes. You can help users with their questions via concise responses." \
                        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" \
                        "hide new secretions from the parental units " \
                        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        assert decoded_texts[0].endswith(expected_text)
    
