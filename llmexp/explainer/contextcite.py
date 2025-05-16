import torch
import os 
import json
json_path = 'env_config.json'
with open(json_path, 'r') as file:
    env_config = json.load(file)

hf_home = env_config['HF_HOME']
# Set the HF_HOME environment variable
os.environ['HF_HOME'] = hf_home
# Set the access token to huggingface hub
access_token = env_config['access_token']
os.environ['HUGGINGFACE_HUB_TOKEN'] = access_token

from context_cite import ContextCiter

class ContextCiteWrapper():
    def __init__(self, model, tokenizer, device):
        super().__init__()
        self.model = model.model
        self.tokenizer = tokenizer
        self.device = device
        

    def attribute(self, context, query, source_type='sentence', num_ablations=20):
        cc = ContextCiter(self.model, self.tokenizer, context=context, query=query, source_type=source_type, num_ablations=num_ablations)
        
        sources = cc.sources
        attributions = cc.get_attributions()
        
        return sources, attributions




