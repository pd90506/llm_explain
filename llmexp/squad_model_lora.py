'''
This version uses self attention instead of direct MLP to generate the masks.
other worthnoting changes
1. this version prepend the response to the context, 
'''


import torch.nn as nn 
import torch.nn.functional as F
import torch
from transformers import BertConfig 
from transformers.models.bert.modeling_bert import BertEncoder, BertPredictionHeadTransform


from torch.utils.data import DataLoader, Dataset


class Encoder(BertEncoder):
    def __init__(self, hidden_size=768, pred_hidden_dim=768):
        config = BertConfig()
        config.num_hidden_layers = 6
        config.hidden_size = hidden_size
        config.intermediate_size = 3072
        config.num_attention_heads = 12
        super().__init__(config)
        self.pre_fc = nn.Linear(hidden_size, pred_hidden_dim)
        self.transform = BertPredictionHeadTransform(config)
        self.fc = nn.Linear(pred_hidden_dim, 1)
    
    def forward(self, hidden_states):
        hidden_states = self.pre_fc(hidden_states)
        last_hidden_state = super().forward(hidden_states)['last_hidden_state']
        output = self.transform(last_hidden_state)
        output = self.fc(output)
        return output


def similarity_measure(logits, labels, attention_mask):
    """ 
    args:
        logis: torch.Tensor, shape (batch_size, seq_len, vocab_size)
        labels: torch.Tensor, shape (batch_size, seq_len)
        attention_mask: torch.Tensor, shape (batch_size, seq_len) the original input text is masked with 0

    return:
        mean_log_probs: torch.Tensor, shape (batch_size,)
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    # correct_log_probs = log_probs[range(log_probs.shape[0]), labels]

    # Flatten the tensors to simplify indexing
    batch_size, seq_len, vocab_size = logits.size()
    labels = labels.view(-1)  # flatten to (batch_size * seq_len)
    log_probs_flat = log_probs.view(-1, vocab_size)

    # Extract the log probabilities for the correct labels
    correct_log_probs_flat = log_probs_flat[range(batch_size * seq_len), labels]

    # Reshape to (batch_size, seq_len)
    correct_log_probs = correct_log_probs_flat.view(batch_size, seq_len)

    # Mask out the original input texts, only keep the generated tokens
    masked_log_probs = correct_log_probs * attention_mask

    # Calculate the mean log probability for each sequence
    mean_log_probs = masked_log_probs.sum(dim=-1)/ attention_mask.sum(dim=-1)

    return mean_log_probs, correct_log_probs


class MaskGeneratingModel(nn.Module):
    def __init__(self, hidden_size=4096, mlp_hidden_dim=1024, lr=1e-4, epsilon=0.2):
        """ 
        hidden_size: int
            The hidden size of the output of the generative model, 4096 for llama3
        """
        super().__init__()

        self.hidden_size = hidden_size

        self.policy_net = Encoder(hidden_size=hidden_size, pred_hidden_dim=mlp_hidden_dim)
        self.value_net = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1)
        )
        self.lr = lr
        self.epsilon = epsilon
    
    def forward(self, pred_features):
        mask_logits, value = self.policy_net(pred_features), self.value_net(pred_features).squeeze(-1)
        return mask_logits, value

    def training_step(self, batch, batch_idx):
        # Unpack batch
        states, actions, old_log_probs, rewards = batch
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).to(self.device)
        
        logits = self(states)
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        ratio = torch.exp(new_log_probs - old_log_probs)
        advantages = returns - returns.mean()

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        loss = -torch.min(surr1, surr2).mean()

        self.log("train_loss", loss)
        return loss
    
    @torch.no_grad()
    def generate_mask(self, mask_logits, context_mask):
        """
        generate mask based on a Bernoulli distribution with the probabilities of the given logits 
        args:
            mask_logits: torch.Tensor, shape (batch_size, seq_len)
            context_mask: torch.Tensor, shape (batch_size, seq_len), the context texts 
                (for example, the chatbot instruction template, not including the user inputs) are masked with 0.
                This mask strategy is to ensure we only focus on the real user inputs.
        return:
            mask: torch.Tensor, shape (batch_size, seq_len), with contexts being all 1s and user inputs being randomly masked.
        """
        mask_probs = torch.sigmoid(mask_logits) # (batch_size, seq_len)

        # For each batch, sample a proportion k from uniform [0, 1)
        batch_size, seq_len = mask_logits.size()
        k_values = torch.rand(batch_size).to(mask_logits.device)  # shape (batch_size,)

        # Initialize masks with all ones (unmasked tokens)
        masks = torch.zeros_like(mask_logits)

        for i in range(batch_size):
            # Get the indices of context tokens
            context_indices = torch.nonzero(context_mask[i], as_tuple=False).squeeze() 
            # Get the number of tokens in the context 
            num_context_tokens = context_indices.size(0)

            # Calculate the number of tokens to unmask based on k
            k = k_values[i].item()  # scalar k for this batch

            num_unmasked = max(1, int(k * num_context_tokens))  # Ensure at least one token is unmasked

            # Get logits for context tokens and sort them by value
            context_logits = mask_logits[i, context_indices]
            sorted_indices = torch.argsort(context_logits, descending=True)  # Sort in descending order

            # Select top-k tokens to unmask
            top_k_indices = sorted_indices[:num_unmasked]

            # Set the corresponding mask values to 1 (unmask)
            unmasked_indices = context_indices[top_k_indices]
            masks[i, unmasked_indices] = 1  # Unmask the selected tokens

        # Combine with the context mask to ensure only context tokens are affected
        final_mask = (context_mask * masks) + (1. - context_mask)  # Ensure non-context tokens remain unmasked

        return final_mask # only randomly mask the context locations, otherwise 1.

    def ppo_loss(self, new_log_probs, old_log_probs, advantage):
        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        loss = -torch.min(ratio * advantage, clipped_ratio * advantage).mean()
        return loss
    
    @torch.no_grad()
    def sample_one_batch(self, input_ids, attention_mask, mask_logits, context_mask):
        """ 
        args:
            input_ids: torch.Tensor, shape (batch_size, seq_len)
            attention_mask: torch.Tensor, shape (batch_size, seq_len), the attention_mask generated automatically by tokenizer, usually all 1s
            mask_logits: torch.Tensor, shape (batch_size, seq_len), the logits for the mask generation
            context_mask: torch.Tensor, shape (batch_size, seq_len), the context texts 
                (for example, the chatbot instruction template, not including the user inputs) are masked with 0.
        """
        mask = self.generate_mask(mask_logits=mask_logits, context_mask=context_mask) # (batch_size, seq_len)

        # get the masked attention_mask
        masked_attention_mask = attention_mask * mask
        # print(masked_attention_mask[0])

        # get the padded input_ids
        masked_input_ids = self.tokenizer.pad_token_id * torch.ones_like(input_ids) * (1 - masked_attention_mask).long() + input_ids * masked_attention_mask.long()

        return masked_input_ids, masked_attention_mask, mask
    
    def sample_one_batch_multi_samples(self, input_ids, attention_mask, mask_logits, context_mask, num_samples=5):
        """ 
        sample multiple samples for each input
        """
        batch_size, seq_len = input_ids.size()
        samples = []
        for _ in range(num_samples):
            masked_input_ids, masked_attention_mask, user_input_mask = self.sample_one_batch(input_ids, attention_mask, mask_logits, context_mask)
            samples.append((masked_input_ids, masked_attention_mask, user_input_mask))
        return samples
    
    @torch.no_grad()
    def calculate_sim(self, model, input_ids, attention_mask, response_mask, labels):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = outputs.logits.float()
        # shift labels and response_mask to the left by one to match the next token prediction format
        labels = labels.clone()
        labels[:, :-1] = labels[:, 1:]
        labels[:, -1] = -100 # a random value that will be masked by the shift_response_mask
        shift_response_mask = response_mask.clone()
        shift_response_mask[:, :-1] = shift_response_mask[:, 1:]
        shift_response_mask[:, -1] = 0 # mast be zero.
        sim, correct_logprob = similarity_measure(logits, labels, shift_response_mask)
        # print('sim', sim.mean())
        return sim, correct_logprob
    
    @torch.no_grad()
    def get_reward(self, sim, sim_gt, user_input_mask, context_mask):
        reward = torch.exp(sim - sim_gt)
        # factor = context_mask.sum(-1) / user_input_mask.sum(-1)
        factor = user_input_mask.sum(-1) / context_mask.sum(-1)
        reward = reward * torch.log(factor + 1e-5)
        # reward = torch.exp(sim_gt - sim)
        # reward = torch.exp(sim_gt - sim) 
        # reward = torch.exp(sim)
        # reward = torch.exp(sim_gt) - torch.exp(sim)
        # reward = torch.clamp(reward, 0.1, 0.5)

        return reward
    
    def loss_func(self, 
                  model, 
                  gen_tokens,
                  gen_attention_mask, 
                  context_mask,
                  mask_logits,
                  response_mask,
                  num_samples=5):
        # obtain the mask_prob
        mask_prob = torch.sigmoid(mask_logits)
        mask_prob_1 = mask_prob * context_mask
        # mask_prob = (1 - mask_prob) * context_mask
        # print('mask_prob', mask_prob[0])

        # obtain the perturbed_samples
        perturbed_samples = self.sample_one_batch_multi_samples(gen_tokens, gen_attention_mask, mask_logits, context_mask, num_samples=num_samples)

        # obtain the groundtruth similarity (similarity of the generated tokens without perturbation)
        reward_loss = 0
        total_reward = 0
        total_effective_mask_prob = 0
        total_advantage = 0
        sim_gt, _ = self.calculate_sim(model, gen_tokens, gen_attention_mask, response_mask, labels=gen_tokens)
        for perturbed_input_ids, perturbed_attention_mask, user_input_mask in perturbed_samples:
            # obtain the similarity of the perturbed samples
            sim, _ = self.calculate_sim(model, perturbed_input_ids, perturbed_attention_mask, response_mask, labels=gen_tokens)
            reward = self.get_reward(sim, sim_gt, user_input_mask, context_mask) 
            advantage = reward - 0.2
            total_advantage += advantage
            # reward = torch.exp(sim)
            total_reward += reward
            # reward_loss += (torch.relu(reward - 0.5) * self.bce_loss(mask_prob, user_input_mask, context_mask)).mean(dim=-1)
            user_input_mask = user_input_mask * context_mask 
            reward_loss += (reward * self.bce_loss(mask_prob, user_input_mask, context_mask)).mean()
            # reward_loss += (advantage * self.bce_loss(mask_prob, user_input_mask, context_mask)).mean()

            # the effective mask is the mask sum of entries that are not masked 
            effective_mask_prob = (mask_prob_1 * user_input_mask * context_mask).sum(-1) / context_mask.sum(dim=-1)
            total_effective_mask_prob += effective_mask_prob

            
        
        reward_loss = reward_loss / num_samples
        # reward_loss = reward_loss.mean()
        effective_mask_prob = total_effective_mask_prob / num_samples
        # mask_loss = (mask_prob * context_mask).sum(-1) / context_mask.sum(dim=-1)

        mask_mean = (mask_prob_1.sum(-1) / context_mask.sum(-1)).mean()
        mean_reward = total_reward / num_samples

        # mask_loss = ((1.2 - mean_reward - mask_loss)**2).mean()
        # mask_loss = ((0.05 -  mask_loss)**2).mean()
        # effective_mask_loss = 
        # mask_loss = torch.relu(0.2 - effective_mask_prob).mean()
        # mask loss, enforce the mask to be sparse
        # advantage = mean_reward - 0.2
        advantage = total_advantage / num_samples
        mask_loss = (mask_prob_1.sum(-1) / context_mask.sum(-1) * advantage).mean()
        # mask_loss = mask_mean
        # mask_loss = ((0.2 -  effective_mask_prob)**2).mean()
        # mean_reward = mean_reward.mean()

        loss = reward_loss #+ 0.01 * mask_loss
        return {'loss': loss, 'reward_loss': reward_loss, 'mask_loss': mask_loss, 'mask_mean': mask_mean, 'mean_reward': mean_reward, 'advantage': advantage.mean()}
    
    def compute_similarity(self, logits, labels, attention_mask):
        """ 
        compute the mean log_probs for the generated tokens
        """
        mean_log_probs = similarity_measure(logits, labels, attention_mask)
        return mean_log_probs
    