from typing import List
import numpy as np
from llmexp.llm.smollm import LLMWrapper    
import torch
import torch.nn.functional as F



class MABExplainer:
    def __init__(self, model: LLMWrapper, tokenizer, template):
        self.model = model
        self.tokenizer = tokenizer
        self.template = template
        # self.bert_scorer = BERTScorer(lang="en", model_type="bert-base-uncased")
    
    def attribute(self, sentences: List[str], sentence_mask: np.ndarray):
        pass
    
    def sample_sentences(self, sentences: List[str], theta: np.ndarray, dismiss=True):
        """
        Sample sentences based on the combinatorial multi-armed bandit scores.
        """
        mask = self.oracle(theta)
        if dismiss:
            mask = 1 - mask
        sampled_indices = np.where(mask == 1)[0]
        # get the sentences
        sampled_sentences = [sentences[i] for i in sampled_indices]
        return sampled_sentences
    
    def oracle(self, theta: np.ndarray, top_p=0.2, top_k=None):
        """
        Oracle that returns a binary mask for sentence selection.
        Args:
            theta: probability distribution array
            top_p: proportion of sentences to select (used if top_k is None)
            top_k: exact number of top sentences to select
        Returns:
            Binary mask array where 1 indicates selected sentences
        """
        # Get indices of top sentences
        num_to_select = top_k if top_k is not None else int(len(theta) * top_p)
        top_indices = np.argsort(theta)[::-1][:num_to_select]
        
        # Create and return binary mask
        mask = np.zeros(len(theta))
        mask[top_indices] = 1
        return mask.astype(int)
        
    @torch.no_grad()
    def thompson_sampling(self, sentences: List[str], question: str, response: str, n_iter=100,
                        prior_mean=0.0, prior_variance=1.0, noise_variance=1.0):
        n_arms = len(sentences)
        means = np.ones(n_arms) * prior_mean
        variances = np.ones(n_arms) * prior_variance

        # Get log-likelihood of full context and empty context for normalization
        full_sentence_rewards, empty_sentence_rewards = self.get_baseline_rewards(sentences, question, response)


        for t in range(1, n_iter + 1):
            # Step 1: Thompson sample θ̃_j ∼ N(μ_j, σ²_j)
            theta = np.random.normal(means, np.sqrt(variances))

            # Step 2: Select top-k segments as super-arm
            super_arm = self.oracle(theta, top_p=0.2)
            super_arm = super_arm.astype(bool)

            # Step 3: Query the LLM and observe reward V(S)
            reward = self.pull(sentences, super_arm, question, response,
                            full_sentence_rewards, empty_sentence_rewards)  # scalar

            # Step 4: Update posterior for selected arms using CTS Gaussian rule
            for i in np.where(super_arm)[0]:
                old_var = variances[i]
                old_mean = means[i]
                new_var = 1.0 / (1.0 / old_var + 1.0 / noise_variance)
                new_mean = new_var * (old_mean / old_var + reward / noise_variance)
                variances[i] = new_var
                means[i] = new_mean

        return means  # estimated relevance scores for each segment
    
    # @torch.no_grad()
    # def thompson_sampling(self, sentences: List[str], question: str, response: str, n_iter=100):
    #     # initialize the parameters
    #     alpha = np.ones(len(sentences))
    #     beta = np.ones(len(sentences))
    #     reward_type = "log_probability"
        
    #     full_sentence_rewards, empty_sentence_rewards = self.get_baseline_rewards(sentences, question, response, reward_type)

        
    #     for t in range(1, n_iter + 1):
    #         # 1) Thompson‑sample a success‑probability for every arm
    #         theta = np.random.beta(alpha, beta)

    #         # 2) Oracle: pick the super‑arm = top‑k sampled arms
    #         super_arm = self.oracle(theta, top_p=0.2)     # 1‑line exact oracle
    #         super_arm = super_arm.astype(bool)

    #         # 3) Play the super‑arm and observe semi‑bandit feedback
    #         rewards = self.pull(sentences, super_arm, question, response, full_sentence_rewards, empty_sentence_rewards, reward_type)  

    #         # 4) Bayesian update for selected arms
    #         alpha[super_arm] += rewards.item()           # successes
    #         beta[super_arm]  += 1 - rewards.item()        # failures
        
    #     theta0 = alpha / (alpha + beta)
    #     return theta0

    
    # @torch.no_grad()
    # def get_baseline_rewards(self, sentences: List[str], question: str, response: str):
    #     full_sentence_rewards = self.model.get_log_likelihood(sentences, question, response)
    #     empty_sentence_rewards = self.model.get_log_likelihood([" "], question, response)
    #     return full_sentence_rewards, empty_sentence_rewards

    @torch.no_grad()
    def get_baseline_rewards(self, sentences: List[str], question: str, response: str, reward_type: str = "log_probability"):
        full_sentence_logits = self.get_response_logits(sentences, question, response)
        empty_sentence_logits = self.get_response_logits([""], question, response)
        full_sentence_rewards = self.get_reward(full_sentence_logits, response, reward_type)
        empty_sentence_rewards = self.get_reward(empty_sentence_logits, response, reward_type)
        return full_sentence_rewards, empty_sentence_rewards

    @torch.no_grad()
    def pull(self, sentences: List[str], super_arm: np.ndarray, question: str, response: str, full_sentence_rewards: torch.Tensor, empty_sentence_rewards: torch.Tensor, reward_type: str = "log_probability"):
        """
        Pull the super arm and observe the reward.
        """
        # get the sentences
        sampled_sentences = self.sample_sentences(sentences, super_arm, dismiss=False)
        # rewards = self.model.get_log_likelihood(sampled_sentences, question, response)
        sampled_sentence_logits = self.get_response_logits(sampled_sentences, question, response)
        rewards = self.get_reward(sampled_sentence_logits, response, reward_type)
        # return rewards
        
        if reward_type == "log_probability":
            reward_diff = (rewards.exp() - empty_sentence_rewards.exp()).sum(dim=1) / (full_sentence_rewards.exp() - empty_sentence_rewards.exp()).sum(dim=1)
            reward_diff = torch.clip(reward_diff, -1, 1)
            # reward_diff = (rewards - torch.log((1-rewards.exp()))).mean(dim=1)

            similarity = reward_diff.mean()

            return similarity
        else:
            raise ValueError(f"Reward type {reward_type} not supported")
        
    
    def get_response(self, sentences: List[str], question: str):
        response_tokens = self.get_response_tokens(sentences, question)
        response = self.tokenizer.decode(response_tokens[0])
        return response
        

    def get_response_tokens(self, sentences: List[str], question: str):
        messages = self.template(sentences, question)
        inputs = self.tokenizer(messages, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        response_output = self.model.generate(**inputs, max_new_tokens=256)
        # get the response tokens
        response_tokens = response_output['input_ids']
        response_tokens = response_tokens[:, inputs['input_ids'].shape[1]:]
        return response_tokens
    
    
    @torch.no_grad()
    def get_response_logits(self, sentences: List[str], question: str, response: str):
        messages = self.template(sentences, question)
        
        input_tokenized = self.tokenizer(messages, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        og_input_ids = input_tokenized.input_ids
        input_attention_mask = input_tokenized.attention_mask

        response_tokenized = self.tokenizer(response, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        response_ids = response_tokenized.input_ids
        response_attention_mask = response_tokenized.attention_mask
        
        # note that the batch size is 1, so they can be concatenated
        input_ids = torch.cat([og_input_ids, response_ids], dim=1)
        attention_mask = torch.cat([input_attention_mask, response_attention_mask], dim=1)

        outputs = self.model.get_logits(input_ids, attention_mask)
        
        # Extract the logits for the response tokens
        response_logits = outputs[:, og_input_ids.shape[1]-1:-1, :]
        
        return response_logits

    # def get_bert_score(self, response1: str, response2: str):
    #     P, R, F1 = self.bert_scorer.score([response2], [response1])

    #     return F1
    
    @torch.no_grad()
    def get_log_likelihood(self, sentences: List[str], question: str, response: str):
        response_tokens = self.tokenizer(response, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        response_ids = response_tokens.input_ids
        sentence_logits = self.get_response_logits(sentences, question, response)
        log_probs = F.log_softmax(sentence_logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=response_ids.unsqueeze(-1)).squeeze(-1)
        log_likelihood = token_log_probs.mean()
        return log_likelihood
        
    
    @torch.no_grad()
    def get_reward(self, masked_logits: torch.Tensor, response: str,
                   baseline_logits=None, reward_type: str = "log_probability"):
        """
        masked_logits: [batch_size, seq_len, vocab_size] — logits
        baseline_tokens: [batch_size, seq_len] — token ids (targets)
        Returns: token-level negative log-likelihood (cross entropy) reward
        """
        response_tokens = self.tokenizer(response, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        response_ids = response_tokens.input_ids
        response_attention_mask = response_tokens.attention_mask
        
        
        if reward_type == "cross_entropy":
            # Shift logits and targets if needed (e.g., decoder-style models)
            logits = masked_logits  # [B, T, V]
            targets = response_ids  # [B, T]

            # Flatten for cross entropy
            logits_flat = logits.view(-1, logits.size(-1))  # [B*T, V]
            targets_flat = targets.view(-1)  # [B*T]

            # Cross entropy loss (negative log likelihood), no reduction
            loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')  # [B*T]

            # Reshape and take mean per example
            loss = loss.view(targets.size())  # [B, T]
            reward = -loss.mean(dim=1)  # reward = negative loss (higher is better)

            return reward  # shape: [B]
        if reward_type == 'log_probability':
            log_probs = F.log_softmax(masked_logits, dim=-1)
            targets = response_ids  # [B, T]
            token_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
            return token_log_probs
        
        if reward_type == 'kl':
            if baseline_logits is None:
                raise ValueError("baseline_logits is required for kl reward")
            kl_div = F.kl_div(
                F.log_softmax(masked_logits, dim=-1),
                F.softmax(baseline_logits, dim=-1),
                reduction='batchmean'
            )
            return -kl_div
        
        