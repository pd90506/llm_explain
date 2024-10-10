#%%
# Configuring environment parameters
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


# Loading necessary packages
import transformers 
import torch

from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, LlamaForTokenClassification #, LlamaRotaryEmbedding
# from transformers import LlamaTokenizerFast
import torch.nn.functional as F

from llmexp.helper import DataHelper
from datasets import load_dataset
from torch.utils.data import DataLoader

# TODO 注意load正确的模型
from llmexp.next_token_model import MaskGeneratingModelForIMDB
from tqdm import tqdm


#%% 
# Load datasets
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token, padding_side='left')
tokenizer.pad_token = tokenizer.eos_token

ds = load_dataset("imdb")
# ds = load_dataset("rajpurkar/squad")
# ds = load_dataset("stanfordnlp/sst2")
train_ds = ds['train']
test_ds = ds['test']

llm_exp_helper = DataHelper(tokenizer)
collate_fn = llm_exp_helper.get_collate_fun('imdb')

# Define batch size here!
batch_size = 16
train_dataloader = DataLoader(train_ds, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
test_dataloader = DataLoader(train_ds, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

# %% 
# Configure and load model
accelerator = Accelerator()
device = accelerator.device
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_id = "meta-llama/Meta-Llama-3-8B"  # non-instruct version

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    # device_map=device,
    token=access_token,
)

config = model.config

#%%
# Configure mask model and  Training parameters
mask_gen_model = MaskGeneratingModelForIMDB()
mask_gen_model.to(device)

# Set pad_token_id if it is not set
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

optimizer = torch.optim.Adam(mask_gen_model.parameters(), lr=1e-3)


# %%
mask_gen_model.train()
for epoch in range(1):
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
        unpaded_token_mask = (gen_tokens != tokenizer.pad_token_id).long()
        unpaded_token_mask[:, :-pad_length] = 1
        gen_attention_mask = gen_attention_mask * unpaded_token_mask

        # get the response mask, which is the mask for the generated tokens (the user inputs are masked with 0)
        response_mask = gen_attention_mask.clone()
        response_mask[:, :-pad_length] = 0 # TODO: 有问题. 有问题吗？

        context_mask = F.pad(context_mask, (0, pad_length), mode='constant', value=0)

        loss_dict = mask_gen_model.train_one_batch(model, gen_tokens, gen_attention_mask, context_mask, response_mask, optimizer,
                                                   num_steps=5, mini_batch_size=16, ppo_epochs=2)


        log = f"Epoch {epoch+1}, Step {idx+1}: Loss = {loss_dict['loss']:.4f}, " \
               f"Actor Loss = {loss_dict['actor_loss']:.4f}, " \
               f"Critic Loss = {loss_dict['critic_loss']:.4f}, " \
               f"Entropy = {loss_dict['entropy']:.4f}, " \
               f"Returns = {loss_dict['returns']:.4f}, " \
               f"Value = {loss_dict['value']:.4f}, " \
                f"mask_loss = {loss_dict['mask_loss']:.4f}" \
                f"std_loss = {loss_dict['std_loss']:.4f}" \
            #    f"Cont_loss = {loss_dict['contrast_loss']:.4f}, "  \
               
        pbar.set_description(log)

        # if idx % 1 == 0:
        #     print()
        if idx % 10 == 0 and idx != 0:
            torch.save(mask_gen_model.state_dict(), f'saved_model/imdb_mask_gen_model_{epoch}_{idx}.pth') 
        #     print()
        #     # break