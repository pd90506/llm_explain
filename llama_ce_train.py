import os
import json 
import logging

logging.basicConfig(
    filename='log/app.log',            # Specify the log file name
    level=logging.DEBUG,           # Set the log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s'  # Set the log format
)

# Load the environment configuration JSON data
json_path = 'env_config.json'
with open(json_path, 'r') as file:
    env_config = json.load(file)

hf_home = env_config['HF_HOME']
# Set the HF_HOME environment variable
os.environ['HF_HOME'] = hf_home
# Set the access token to huggingface hub
access_token = env_config['access_token']
os.environ['HUGGINGFACE_HUB_TOKEN'] = access_token

import transformers 
print(transformers.__version__)

from transformers import pipeline
import torch

from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers import LlamaTokenizerFast
import torch.nn.functional as F


accelerator = Accelerator()
device = accelerator.device

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_id = "meta-llama/Meta-Llama-3-8B"  # non-instruct version

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token=access_token,
)

tokenizer = LlamaTokenizerFast.from_pretrained(model_id, token=access_token)

from datasets import load_dataset
from torch.utils.data import DataLoader
from llmexp.helper import LlmExpHelper

imdb = load_dataset("imdb")
train_ds = imdb['train']
llm_exp_helper = LlmExpHelper(tokenizer, model, device)
collate_fn = llm_exp_helper.get_collate_fun()

# Define batch size here!
batch_size = 16
train_dataloader = DataLoader(train_ds, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)


from llmexp.model import MaskGeneratingModel

mask_gen_model = MaskGeneratingModel(hidden_size=4096, mlp_hidden_dim=1024, mlp_bottleneck_dim=768, mlp_num_blocks=2)
mask_gen_model.to(device)


from tqdm import tqdm

# Set pad_token_id if it is not set
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

pad_token_id = tokenizer.pad_token_id

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

optimizer = torch.optim.Adam(mask_gen_model.parameters(), lr=1e-5)


for epoch in range(10):
    pbar = tqdm(train_dataloader)
    for idx, data in enumerate(pbar):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        context_mask = data['context_mask'].to(device)
        # get generated texts
        gen_outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=128,
            eos_token_id=terminators,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            return_dict_in_generate=True,
            output_scores=True,
        )
        gen_tokens = gen_outputs.sequences
        pad_length = gen_tokens.size(1) - input_ids.size(1)
        # get the attention mask for the generated tokens, and also mask the padding tokens
        gen_attention_mask = F.pad(attention_mask, (0, pad_length), mode='constant', value=1)
        # (gen_tokens != pad_token_id).long() is the tokens mask, 1 for real tokens and 0 for padding tokens
        unpaded_token_mask = (gen_tokens != pad_token_id).long()
        unpaded_token_mask[:, :-pad_length] = 1
        gen_attention_mask = gen_attention_mask * unpaded_token_mask
        # get the response mask, which is the mask for the generated tokens (the user inputs are masked with 0)
        response_mask = gen_attention_mask.clone()
        response_mask[:, :-pad_length] = 0 # TODO: 有问题. 有问题吗？

        # Get the last hidden state for the prompt sequence
        with torch.no_grad():
            prompt_outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
            last_hidden_state = prompt_outputs.hidden_states[-1]
            last_hidden_state = last_hidden_state.float()
        
        mask_logits = mask_gen_model(last_hidden_state)

        loss, reward_loss, mask_loss, mask_mean = mask_gen_model.loss_func(model, gen_tokens, gen_attention_mask, context_mask, mask_logits, response_mask, num_samples=5)
        
        log = (f"Epoch {epoch+1}, Step {idx+1}: Loss = {loss.item():.4f}, " 
                             f"Reward Loss = {reward_loss.item():.4f}, "

                             f"Mask = {mask_loss.item():.4f} "
                             f"maskmean = {mask_mean.item():.4f}"
        )
        pbar.set_description(log)
        logging.debug(log)
        # Train the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #                      )
        # if idx % 10 == 0:
        #     print()
        if idx % 200 == 0 and idx != 0:
            torch.save(mask_gen_model.state_dict(), f'saved_model/mask_gen_model_{epoch}_{idx}.pth') 
            # break


        #     
