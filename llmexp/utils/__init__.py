from transformers import PreTrainedTokenizerBase
import torch

def decode_output(input_ids, attention_mask, tokenizer: PreTrainedTokenizerBase):
    # Use attention mask to filter tokens
    filtered_ids = input_ids[attention_mask.bool()]
    decoded_output = tokenizer.decode(filtered_ids)
    return decoded_output

def decode_batch_outputs(input_ids, attention_mask, tokenizer: PreTrainedTokenizerBase):
    texts = []
    # Process each sequence in the batch
    for seq_ids, seq_mask in zip(input_ids, attention_mask):
        # Filter tokens using attention mask
        filtered_ids = seq_ids[seq_mask.bool()]
        # Decode the filtered sequence
        text = tokenizer.decode(filtered_ids, skip_special_tokens=False)
        texts.append(text)
    
    return texts

def get_mab_masked_inputs(input_ids, attention_mask, mab_mask, tokenizer: PreTrainedTokenizerBase):
    masked_input_ids = input_ids.clone()
    
    # Pad mab_mask with zeros at the end to match attention_mask size
    padded_mab_mask = torch.cat([mab_mask, torch.zeros((mab_mask.size(0), 1), device=mab_mask.device)], dim=1)
    
    mask_positions = (attention_mask == 1) & (padded_mab_mask == 0)
    masked_input_ids[mask_positions] = tokenizer.pad_token_id

    masked_attention_mask = attention_mask.clone()
    masked_attention_mask[mask_positions] = 0

    return {'input_ids': masked_input_ids, 
            'attention_mask': masked_attention_mask}