import torch


def template_fn(messages):
    # Initialize the results dictionary with lists
    results = {
        'result': [],
        'start': [],
        'end': []
    }
    
    # Get the list of contexts and user messages
    contexts = messages.get('context', [])
    user_messages = messages.get('user_message', [])
    
    # Ensure both lists are of the same length
    if len(contexts) != len(user_messages):
        raise ValueError("The 'context' and 'user_message' lists must be of the same length.")
    
    # Iterate over the indices
    for idx in range(len(contexts)):
        context = contexts[idx]
        user_message = user_messages[idx]
        
        # Define the template components
        part1 = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        part2 = f"{context}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        part3 = user_message
        part4 = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        # Combine parts to form the full template
        result = part1 + part2 + part3 + part4
        
        # Calculate start and end indices of the user_message
        user_message_start = len(part1 + part2)
        user_message_end = user_message_start + len(user_message)
        
        # Append to results
        results['result'].append(result)
        results['start'].append(user_message_start)
        results['end'].append(user_message_end)
        
    return results


class DataHelper():
    def __init__(self, tokenizer, template=None, max_seq_len=512):
        self.tokenizer = tokenizer

        self.template = template if template is not None else template_fn
        self.max_seq_len = max_seq_len

    def get_collate_fun(self, dataset):
        if dataset == 'imdb':
            return self.collate_fn_imdb
        elif dataset == 'sst2':
            return self.collate_fn_sst2
        elif dataset == 'squad':
            return self.collate_fn_squad
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

    def collate_fn_imdb(self, examples):
        return self._collate_fn_generic(
            examples,
            text_key='text',
            sys_context=(
                "You are a chatbot for sentiment analysis. "
                "You can help users with their questions via concise responses of POSITIVE or NEGATIVE."
            )
        )

    def collate_fn_sst2(self, examples):
        return self._collate_fn_generic(
            examples,
            text_key='sentence',
            sys_context=(
                "You are a chatbot for sentiment analysis. "
                "You can help users with their questions via concise responses of POSITIVE or NEGATIVE."
            )
        )

    def collate_fn_squad(self, examples):

        # contexts = [example['context'] for example in examples]
        # # contexts = examples['context']
        # questions = [example['question'] for example in examples]
        # # questions = examples['question']
        # texts = [
        #     f"Question: {questions[i]}\nContext: {contexts[i]}\n\n"
        #     for i in range(len(examples['id']))
        # ]
        def combine_question_context(example):
            example['sentence'] = f"Question: {example['question']}\nContext: {example['context']}"
            return example
        # The examples is a list of dictionaries
        # examples_with_sentence = examples.map(combine_question_context)
        examples_with_sentence = [combine_question_context(example) for example in examples]

        batch = self._collate_fn_generic(
            examples_with_sentence, 
            text_key='sentence',
            sys_context=(
            "You are a chatbot for answering questions. "
            "Your reply with a short answer to the question provided in the context."
            )
        )

        # # Update context mask for 'squad' dataset
        # context_lens = [len(tokenizer.encode(ctx)) for ctx in contexts]
        # context_lens_tensor = torch.tensor(context_lens, dtype=torch.long)
        # mask_tensor_v2 = self._apply_mask(
        #     torch.ones_like(batch['input_ids']), context_lens_tensor
        # )
        # batch['context_mask'] = mask_tensor_v2 * batch['attention_mask']

        return batch

    def _collate_fn_generic(self, examples, text_key, sys_context):
        tokenizer = self.tokenizer
        max_seq_len = self.max_seq_len

        def get_clipped_texts(texts):
            """
            Clip texts to max_seq_len tokens.
            """
            max_tokens = max_seq_len
            tokenizer = self.tokenizer 
            clipped_texts = []
            for text in texts:
                token_ids = tokenizer.encode(text)
                if len(token_ids) > max_tokens:
                    token_ids = token_ids[:max_tokens]
                    clipped_text = tokenizer.decode(token_ids, skip_special_tokens=True)
                else:
                    clipped_text = text
                clipped_texts.append(clipped_text)
            return clipped_texts

        texts = [example[text_key] for example in examples]
        # texts = examples[text_key]
        texts = get_clipped_texts(texts)
        sys_context = [sys_context] * len(texts)

        # Create messages
        messages = {
            "context": sys_context,
            "user_message": texts,
        }

        # Apply chat template
        messages_with_template_tuple = self.template(messages)
        messages_with_template = messages_with_template_tuple['result']
        # user_message_starts = messages_with_template_tuple['start']
        # user_message_ends = messages_with_template_tuple['end']   

        # Tokenize messages
        batch = tokenizer(
            messages_with_template,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )

        text_lens = [len(tokenizer.encode(text)) for text in texts]
        text_lens_tensor = torch.tensor(text_lens, dtype=torch.long)


        mask_tensor = self._apply_mask(torch.ones_like(batch['input_ids']), text_lens_tensor)

        batch['context_mask'] = mask_tensor

        return batch

    def _apply_mask(self, mask_tensor, lens_tensor):
        batch_size, seq_len = mask_tensor.shape
        for i in range(batch_size):
            text_len = lens_tensor[i].item()
            mask_tensor[i, -text_len-5+1:-5] = 0
        return 1 - mask_tensor