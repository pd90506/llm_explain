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
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from typing import List
import numpy as np
from llmexp.llm.smollm import LLMWrapper, Template
from accelerate import Accelerator
from llmexp.data.hotpot import HotpotDataset
from llmexp.data.sst2 import SST2Dataset
from llmexp.data.tydi import TydiDataset
import random 
from llmexp.utils.evaluation import calculate_avg_log_prob_diff, calculate_bertscore
from llmexp.explainer.shap import SHAPExplainer
import pandas as pd
import torch

def get_perturbed_sentences(sentences: List[str], theta: np.ndarray, k: int = 3):
    # remove the top k sentences
    non_top_k_indices = np.argsort(theta)[:-k]
    non_top_k_sentences = [sentences[i] for i in non_top_k_indices]

    return non_top_k_sentences

def get_results_for_k(llm: LLMWrapper, sentences: List[str], question: str, response: str, theta: np.ndarray, log_likelihood: torch.Tensor, k: int = 3):
    perturbed_sentences = get_perturbed_sentences(sentences, theta, k=k)
    perturbed_response = llm.get_response(perturbed_sentences, question)
    perturbed_log_likelihood = llm.get_log_likelihood(perturbed_sentences, question, response)
    log_prob_drop = calculate_avg_log_prob_diff(perturbed_log_likelihood, log_likelihood)
    bertscore = calculate_bertscore(perturbed_response, response)
    return log_prob_drop, bertscore

if __name__ == "__main__":
    random.seed(42)
    
    # define parameters
    K = 100
    num_iter = 60
    model_name = "smollm"
    dataset_name = "hotpot_qa"

    accelerator = Accelerator()
    device = accelerator.device

    # load the model
    if model_name == "llama":
        checkpoint = "meta-llama/Meta-Llama-3-8B-Instruct"
    elif model_name == "smollm":
        checkpoint = "HuggingFaceTB/SmolLM-1.7B-Instruct"
    llm = LLMWrapper(checkpoint, device=device, access_token=access_token)
    tokenizer = llm.tokenizer
    template = Template(tokenizer, task='qa')
    
    # load the data
    if dataset_name == "hotpot_qa":
        dataset = HotpotDataset(split='test')
    elif dataset_name == "sst2":
        dataset = SST2Dataset(split='test')
    elif dataset_name == "tydi":
        dataset = TydiDataset(split='test')
    # get the first example
    total_samples = len(dataset)
    sampled_indices = random.sample(range(total_samples), K)
    test_data = [dataset[i] for i in sampled_indices]
    
    # load explainer 
    explainer = SHAPExplainer(llm, tokenizer, device)
    
    log_prob_drop_k1_list = []
    bertscore_k1_list = []
    log_prob_drop_k3_list = []
    bertscore_k3_list = []
    log_prob_drop_k5_list = []
    bertscore_k5_list = []
    
    
    for data in test_data:
        if dataset_name == "hotpot_qa":
            sentences = data['sentences']
            question = data['question']
            response = llm.get_response(sentences, question)
        elif dataset_name == "sst2":
            sentences = data['words']
            question = "Is the sentiment of this sentence positive or negative?"
            response = llm.get_response(sentences, question)
        elif dataset_name == "tydi":
            sentences = data['sentences']
            question = data['question']
            response = llm.get_response(sentences, question)

        sentences, theta = explainer.attribute(sentences, question, response, num_ablations=num_iter)
        
        log_likelihood = llm.get_log_likelihood(sentences, question, response)
        
        log_prob_drop_k1, bertscore_k1 = get_results_for_k(llm, sentences, question, response, theta, log_likelihood, k=1)
        log_prob_drop_k3, bertscore_k3 = get_results_for_k(llm, sentences, question, response, theta, log_likelihood, k=3)
        log_prob_drop_k5, bertscore_k5 = get_results_for_k(llm, sentences, question, response, theta, log_likelihood, k=5)
        
        # save the results as a csv file 
        log_prob_drop_k1_list.append(log_prob_drop_k1)
        bertscore_k1_list.append(bertscore_k1)
        log_prob_drop_k3_list.append(log_prob_drop_k3)
        bertscore_k3_list.append(bertscore_k3)
        log_prob_drop_k5_list.append(log_prob_drop_k5)
        bertscore_k5_list.append(bertscore_k5)
        
        results = {
            'log_prob_drop_k1': log_prob_drop_k1_list,
            'log_prob_drop_k3': log_prob_drop_k3_list,
            'log_prob_drop_k5': log_prob_drop_k5_list,
            'bertscore_k1': bertscore_k1_list,
            'bertscore_k3': bertscore_k3_list,
            'bertscore_k5': bertscore_k5_list
        }
        
        # save the results as a csv file 
        results_df = pd.DataFrame(results)
        results_df.to_csv(f'results/{model_name}_{dataset_name}_shap_niter_{num_iter}.csv', index=False)
