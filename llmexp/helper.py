import torch

class LlmExpHelper:
    def __init__(self, tokenizer, dataset):
        self.tokenizer = tokenizer
        self.dataset = dataset
    
    def get_collate_fun(self):
        return lambda examples: self.collate_fn(examples)

    def collate_fn(self, examples):
        def num_words(x):
            return len(x.split())
        def get_first_k_words(x, k):
            return ' '.join(x.split()[:k])
        def get_cliped_text(texts, max_len):
            return [text if num_words(text) <= max_len else get_first_k_words(text, max_len) for text in texts]
        tokenizer = self.tokenizer
        max_len = 512 # characters limit other than token limit
        if self.dataset == 'imdb':
            texts = [example['text'] for example in examples]
            texts = get_cliped_text(texts, max_len)
            sys_context = "You are a chatbot for sentiment analysis. You can help users with their questions via concise responses of POSITIVE, or NEGATIVE."
        elif self.dataset == 'sst2':
            texts = [example['sentence'] for example in examples]
            texts = get_cliped_text(texts, max_len)
            sys_context = "You are a chatbot for sentiment analysis. You can help users with their questions via concise responses of POSITIVE, or NEGATIVE."
        elif self.dataset == 'squad':
            context = [example['context'] for example in examples]
            context = get_cliped_text(context, max_len)
            question = [example['question'] for example in examples]
            # texts = [f"Context: {context[i]}\nQuestion: {question[i]}" for i in range(len(context))]
            texts = [f"Question: {question[i]}\nContext: {context[i]}" for i in range(len(context))]
            sys_context = "You are a chatbot for answering questions. You can help users with their questions via concise responses."

        # labels = [example['label'] for example in examples]
        messages_lambda = lambda texts: [
            {"role": "system", "content": sys_context},
            {"role": "user", "content": texts},
        ]

        messages = list(map(messages_lambda, texts))

        messages_with_template_applied = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        batch = tokenizer(
                    messages_with_template_applied,
                    add_special_tokens=False,
                    padding=True,
                    return_tensors="pt",
                    )
        
        # find the template boundaries
        text_lens = [len(tokenizer.encode(text)) for text in texts]
        text_lens_tensor = torch.tensor(text_lens, dtype=torch.long)

        
        def apply_mask(mask_tensor, text_lens_tensor):
            batch_size, seq_len = mask_tensor.shape
            for i in range(batch_size):
                text_len = text_lens_tensor[i].item()
                mask_tensor[i, -text_len-5:-5] = 0
            return 1- mask_tensor

        mask_tensor = apply_mask(torch.ones_like(batch['input_ids']), text_lens_tensor)

        batch['context_mask'] = mask_tensor

        if self.dataset == 'squad':
            answers_start = [example['answers']['answer_start'][0] for example in examples]
            answers_end = [example['answers']['answer_start'][0] + len(example['answers']['text'][0]) for example in examples]
            batch['answers_start'] = torch.tensor(answers_start).long()
            batch['answers_end'] = torch.tensor(answers_end).long()

            context_lens = [len(tokenizer.encode(context)) for context in context]
            context_lens_tensor = torch.tensor(context_lens, dtype=torch.long)
            mask_tensor_v2 = apply_mask(torch.ones_like(batch['input_ids']), context_lens_tensor)
            batch['context_mask'] = mask_tensor_v2
            
        
        return batch