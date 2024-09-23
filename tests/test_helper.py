import pytest
import torch
from transformers import AutoTokenizer
from llmexp.helper import DataHelper, template_fn

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
        cls.tokenizer = AutoTokenizer.from_pretrained(model_id)
        cls.tokenizer.pad_token = cls.tokenizer.eos_token
        cls.data_helper = DataHelper(cls.tokenizer)

        cls.imdb_examples = {
            'text': ['I rented I AM CURIOUS-YELLOW from my video store because of all the controversy that surrounded it when it was first released in 1967. I also heard that at first it was seized by U.S. customs if it ever tried to enter this country, therefore being a fan of films considered "controversial" I really had to see this for myself.<br /><br />The plot is centered around a young Swedish drama student named Lena who wants to learn everything she can about life. In particular she wants to focus her attentions to making some sort of documentary on what the average Swede thought about certain political issues such as the Vietnam War and race issues in the United States. In between asking politicians and ordinary denizens of Stockholm about their opinions on politics, she has sex with her drama teacher, classmates, and married men.<br /><br />What kills me about I AM CURIOUS-YELLOW is that 40 years ago, this was considered pornographic. Really, the sex and nudity scenes are few and far between, even then it\'s not shot like some cheaply made porno. While my countrymen mind find it shocking, in reality sex and nudity are a major staple in Swedish cinema. Even Ingmar Bergman, arguably their answer to good old boy John Ford, had sex scenes in his films.<br /><br />I do commend the filmmakers for the fact that any sex shown in the film is shown for artistic purposes rather than just to shock people and make money to be shown in pornographic theaters in America. I AM CURIOUS-YELLOW is a good film for anyone wanting to study the meat and potatoes (no pun intended) of Swedish cinema. But really, this film doesn\'t have much of a plot.',
            '"I Am Curious: Yellow" is a risible and pretentious steaming pile. It doesn\'t matter what one\'s political views are because this film can hardly be taken seriously on any level. As for the claim that frontal male nudity is an automatic NC-17, that isn\'t true. I\'ve seen R-rated films with male nudity. Granted, they only offer some fleeting views, but where are the R-rated films with gaping vulvas and flapping labia? Nowhere, because they don\'t exist. The same goes for those crappy cable shows: schlongs swinging in the breeze but not a clitoris in sight. And those pretentious indie movies like The Brown Bunny, in which we\'re treated to the site of Vincent Gallo\'s throbbing johnson, but not a trace of pink visible on Chloe Sevigny. Before crying (or implying) "double-standard" in matters of nudity, the mentally obtuse should take into account one unavoidably obvious anatomical difference between men and women: there are no genitals on display when actresses appears nude, and the same cannot be said for a man. In fact, you generally won\'t see female genitals in an American film in anything short of porn or explicit erotica. This alleged double-standard is less a double standard than an admittedly depressing ability to come to terms culturally with the insides of women\'s bodies.',
            "If only to avoid making this type of film in the future. This film is interesting as an experiment but tells no cogent story.<br /><br />One might feel virtuous for sitting thru it because it touches on so many IMPORTANT issues but it does so without any discernable motive. The viewer comes away with no new perspectives (unless one comes up with one while one's mind wanders, as it will invariably do during this pointless film).<br /><br />One might better spend one's time staring out a window at a tree growing.<br /><br />"],
            'label': [0, 0, 0]
            }

        cls.sst2_examples = {
            'idx': [0, 1, 2],
            'sentence': ['hide new secretions from the parental units ',
            'contains no wit , only labored gags ',
            'that loves its characters and communicates something rather beautiful about human nature '],
            'label': [0, 0, 1]
            }

        cls.squad_examples = {
            'id': ['5733be284776f41900661182',
            '5733be284776f4190066117f',
            '5733be284776f41900661180'],
            'title': ['University_of_Notre_Dame',
            'University_of_Notre_Dame',
            'University_of_Notre_Dame'],
            'context': ['Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.',
            'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.',
            'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.'],
            'question': ['To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?',
            'What is in front of the Notre Dame Main Building?',
            'The Basilica of the Sacred heart at Notre Dame is beside to which structure?'],
            'answers': [{'text': ['Saint Bernadette Soubirous'], 'answer_start': [515]},
            {'text': ['a copper statue of Christ'], 'answer_start': [188]},
            {'text': ['the Main Building'], 'answer_start': [279]}]
            }

    
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
                        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        assert decoded_texts[0].startswith(expected_text)
    
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
                        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        assert decoded_texts[0].startswith(expected_text)
    
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
                        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        assert decoded_texts[0].startswith(expected_text)


    def test_collate_fn_generic(self):
        examples = {
            'idx': [0, 1, 2],
            'sentence': ['hide new secretions from the parental units ',
            'contains no wit , only labored gags ',
            'that loves its characters and communicates something rather beautiful about human nature '],
            'label': [0, 0, 1]
            }
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
                        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        assert decoded_texts[0].startswith(expected_text)
    
