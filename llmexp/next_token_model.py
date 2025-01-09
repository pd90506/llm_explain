'''
This version uses self attention instead of direct MLP to generate the masks.
other worthnoting changes
1. this version prepend the response to the context, 
'''


import torch.nn as nn 
import torch.nn.functional as F
import torch
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()


def similarity_measure(logits, labels):
    """
    Args:
        logits: torch.Tensor, shape (batch_size, vocab_size)
        labels: torch.Tensor, shape (batch_size,)
    """
    # Calculate log probabilities for the last token
    log_probs = torch.log_softmax(logits, dim=-1)  # [batch_size, vocab_size]
    log_prob_predictions = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # [batch_size]

    return log_prob_predictions, log_probs # [batch_size], [batch_size, vocab_size]
    

def mask_similarity_measure(logits, labels):
    """ 
    Args:
        logits: torch.Tensor, shape (batch_size, seq_len, vocab_size)
        labels: torch.Tensor, shape (batch_size,)

    Returns:
        log_prob_predictions_full: torch.Tensor, shape (batch_size, seq_len)
        all_log_probs_full: torch.Tensor, shape (batch_size, seq_len, vocab_size)
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Get logits for all previous tokens predicting the last token
    logits_previous_tokens = logits[:, :-1, :]  # [batch_size, seq_len - 1, vocab_size]
    labels_last_token = labels.unsqueeze(1).expand(-1, logits_previous_tokens.size(1))  # [batch_size, seq_len - 1]

    # Calculate log probabilities for all previous tokens
    all_log_probs = torch.log_softmax(logits_previous_tokens, dim=-1)  # [batch_size, seq_len - 1, vocab_size]
    log_prob_predictions = torch.gather(all_log_probs, dim=-1, index=labels_last_token.unsqueeze(-1)).squeeze(-1)  # [batch_size, seq_len - 1]

    # Expand log_prob_predictions to match seq_len, with last token masked to 0
    log_prob_predictions_full = torch.zeros(batch_size, seq_len, device=logits.device)
    log_prob_predictions_full[:, :-1] = log_prob_predictions

    # Expand all_log_probs to match seq_len, with last token masked to 0
    all_log_probs_full = torch.zeros(batch_size, seq_len, vocab_size, device=logits.device)
    all_log_probs_full[:, :-1, :] = all_log_probs

    return log_prob_predictions_full, all_log_probs_full  # [batch_size, seq_len], [batch_size, seq_len, vocab_size]


class Environment:
    def __init__(self, model):
        """ 
        model: object
            The Llama3 model used in the environment.
        """
        self.model = model
        self.mask_token_id = 128009
        self.state = None

    def set_state(self, state):
        self.state = state
    
    def action_on_state(self, action):
        """ 
        Applies the action (mask) on the current state.
        """
        input_ids, attention_mask, context_mask = self.state
        # get the mask for the last token
        mask = action

        # To be masked tokens are those in the context and also masked by the agent
        # note that here 1 means the token is not masked
        to_be_masked = (1 - mask) * context_mask

        masked_input_ids = input_ids.clone()
        # Mask with the pad(mask) tokens
        masked_input_ids[to_be_masked == 1] = self.mask_token_id
        new_state = masked_input_ids, attention_mask, context_mask
        return new_state

    @torch.no_grad()
    def calculate_sim(self, input_ids, attention_mask, labels):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = outputs.logits.float() # [batch_size, seq_len, vocab_size]
        # Take the logit of the correct token
        logits = logits[:, -2, :]  # [batch_size, vocab_size]
        labels = labels[:, -1]  # [batch_size]

        log_prob_predictions, log_probs = similarity_measure(logits, labels)

        return log_prob_predictions, log_probs

    @torch.no_grad()
    def get_reward(self, action, state):
        """
        Returns the reward for a given action. Since state does not change, reward depends only on the action.
        """
        self.set_state(state)
        new_state = self.action_on_state(action)
        masked_input_ids, attention_mask, context_mask = new_state
        log_prob_predictions, log_probs = self.calculate_sim(masked_input_ids, attention_mask, labels=masked_input_ids)
        reward = torch.exp(log_prob_predictions) * context_mask.sum(-1) / (1 + (action * context_mask).sum(-1))  # [batch_size]
        # reward = torch.exp(log_prob_predictions)  # [batch_size]
        # print("reward", reward.mean())
        return reward


class AutoregressivePredictor(nn.Module):
    def __init__(self, hidden_size, vocab_size, embedding_layer=None):
        super(AutoregressivePredictor, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.reduce_map = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_size),
        )
        self.linear = nn.Linear(hidden_size, hidden_size)
        
        # Linear layer to project hidden state to vocabulary logits (tied to embedding layer)
        if embedding_layer is not None:
            self.output_layer = nn.Linear(hidden_size, vocab_size, bias=False)
            self.output_layer.weight = embedding_layer.weight
            self.output_layer.weight.requires_grad = False
        else:
            self.output_layer = nn.Linear(hidden_size, vocab_size)
        
        # self.norm = nn.LayerNorm(hidden_size)
        self.norm = LlamaRMSNorm(hidden_size, eps=1e-6)

    def forward(self, hidden_state):
        """
        Args:
            hidden_state: torch.Tensor, shape [N, L, d]
                - N: batch size
                - L: sequence length
                - d: hidden size
        
        Returns:
            predictions: torch.Tensor, shape [N, L, vocab_size]
                - Represents the predictions for the L-th token using all previous tokens.
        """
        N, L, d = hidden_state.shape

        # Split hidden_state into previous tokens and the last token
        hidden_state = self.reduce_map(hidden_state)  # [N, L, d]
        previous_tokens = hidden_state[:, :-1, :]  # [N, L-1, d]
        last_token = hidden_state[:, -1, :]  # [N, d]

        # Compute the dot product (KQ similarity) between each of the previous tokens and the last token
        # similarity_scores = torch.bmm(previous_tokens, last_token.unsqueeze(-1)).squeeze(-1)  # [N, L-1]
        similarity_vector = torch.einsum("nld,nd->nld", previous_tokens, last_token)  # [N, L-1, d]
        # similarity_vector = previous_tokens
        # similarity_vector = self.linear(similarity_vector)  # [N, L-1, d]
        # TODO: 改成 self-attention
        similarity_vector = self.norm(similarity_vector)  # [N, L-1, d]
        # print("similarity_vector", similarity_vector[:1])

        # Project to vocabulary size for each previous token
        logits = self.output_layer(similarity_vector)  # [N, L-1, vocab_size]
        # # 注册钩子函数来清除 logits 的梯度
        # def zero_grad_hook(grad):
        #     return grad * 0  # 设置梯度为0，避免更新 output_layer 的权重

        # logits.register_hook(zero_grad_hook)

        # Expand to match expected output shape [N, L, vocab_size] with the last token masked
        final_predictions = torch.zeros(N, L, self.vocab_size, device=hidden_state.device)
        final_predictions[:, :-1, :] = logits
        final_predictions[:, -1, :] = torch.zeros_like(logits[:, 0, :])  # Mask the last token prediction
        
        return final_predictions  # [N, L, vocab_size]


class MaskGenModelForNextToken(nn.Module):
    def __init__(self, hidden_size=512, embedding_layer=None):
        """ 
        hidden_size: int
            The hidden size of the output of the generative model, 4096 for llama3
        """
        super(MaskGenModelForNextToken, self).__init__()

        self.hidden_size = hidden_size
        if embedding_layer is None:
            raise ValueError("The embedding weights must be provided to match the weights of the prediction head.")
        
        self.autoregressive_predictor = AutoregressivePredictor(hidden_size, embedding_layer.weight.shape[0], embedding_layer=embedding_layer)
        self.mask_token_id = 128009

        # Additional layer for value estimation
        self.value_map = nn.Linear(hidden_size, 1)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, context_mask: torch.Tensor):
        """ 
        hidden_states: torch.Tensor of shape [N, L, hidden_size]
        attention_mask: torch.Tensor of shape [N, L]
        context_mask: torch.Tensor of shape [N, L]
        """
        # Obtain value estimates
        value_hidden_states = self.get_pooled_feature(hidden_states)  # [N, hidden_size]
        value = self.value_map(value_hidden_states).squeeze(-1)  # [N]
        # print("value", value[:1])

        # Generate predictions using autoregressive predictor
        predictions = self.autoregressive_predictor(hidden_states)  # [N, L, vocab_size]

        # print("predictions", predictions[:1])

        return predictions, value  # [N, L, vocab_size], [N]

    def get_pooled_feature(self, hidden_states):
        """ 
        Obtain the last token of the sequence
        """
        batch_size = hidden_states.shape[0]
        device = hidden_states.device
        # Always select the last token
        pooled_feature = hidden_states[:, -1, :]  # [N, hidden_size]
        return pooled_feature

    def get_dist_critic(self, model, state):
        """ 
        model: The Llama3 model to generate the logits
        state: The input state, consisting of input_ids, attention_mask, context_mask
        """
        input_ids, attention_mask, context_mask = state
        last_token = input_ids[:, -1:]  # [N, 1]
        # Extract features from the model
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
            last_hidden_state = outputs.hidden_states[-1]  # [N, L, hidden_size]
        
        predictions, value = self.forward(last_hidden_state, attention_mask, context_mask)
        predictions = torch.softmax(predictions, -1)  # [N, L, vocab_size]
        # predictions = torch.gather(predictions, dim=-1, index=last_token.unsqueeze(-1)).squeeze(-1)  # [N, L]
        predictions = torch.gather(predictions, dim=2, index=last_token.unsqueeze(1).expand(-1, predictions.size(1), -1)).squeeze(-1)  # [N, L]
        inferred_state = last_hidden_state, attention_mask, context_mask, last_token
        
        dist = torch.distributions.Bernoulli(probs=predictions)  # [N, L]
        return dist, value, inferred_state # [N, L], [N]


    def bandit_update(self, optimizer, inferred_states, actions, old_log_probs, rewards, advantages, logits, alpha=0.1, clip_param=0.2, batch_size=16):
        """
        Perform bandit-style update with advantage estimation and PPO clipping using mini-batch updates.
        Args:
            optimizer: The optimizer for updating the model parameters.
            inferred_states: Inferred states consisting of input_ids, attention_mask, and context_mask.
            actions: Actions taken by the agent.
            old_log_probs: Log probabilities of actions from the previous policy.
            advantages: Advantages obtained for each action.
            alpha: Learning rate for the bandit update.
            clip_param: Clipping parameter for PPO update.
            batch_size: Size of the mini-batches.
        """
        # Split data into mini-batches
        num_samples = actions.size(0)

        total_loss = 0
        total_ratio = 0
        total_advantages = 0
        num_updates = 0
        total_entropy = 0
        total_kl_div = 0

        # Iterate through mini-batches
        for _ in range(num_samples // batch_size):
            # Randomly sample mini-batch indices
            batch_indices = torch.randint(0, num_samples, (batch_size,))
            
            # Select mini-batch data
            mini_hidden_states = inferred_states[0][batch_indices]
            mini_attention_mask = inferred_states[1][batch_indices]
            mini_context_mask = inferred_states[2][batch_indices]
            mini_last_token = inferred_states[3][batch_indices]
            mini_actions = actions[batch_indices]
            mini_old_log_probs = old_log_probs[batch_indices]
            mini_advantages = advantages[batch_indices]
            mini_logits = logits[batch_indices]
            mini_rewards = rewards[batch_indices]

            # Calculate log probabilities for the actions taken
            mu_logits, value = self.forward(mini_hidden_states, mini_attention_mask, mini_context_mask)
            mu_probs = torch.softmax(mu_logits, -1)

            dist_probs = torch.gather(mu_probs, dim=2, index=mini_last_token.unsqueeze(1).expand(-1, mu_probs.size(1), -1)).squeeze(-1)  # [batch_size, L]
            dist = torch.distributions.Bernoulli(probs=dist_probs)  # [batch_size, L]
            # print("last token", mini_last_token[:3])
            # print("dist_probs", dist_probs[:3])
            # print("mini_actions", mini_actions[:3])
            log_probs = dist.log_prob(mini_actions)

            # Entropy of the distribution
            entropy = dist.entropy().mean()

            # Calculate KL divergence between the mu_logits and the predicted logits
            logits_expanded = mini_logits.unsqueeze(1).expand_as(mu_logits) # [batch_size, L, vocab_size]
            mu_logits_softmax = torch.log_softmax(mu_logits, -1) # [batch_size, L, vocab_size]
            logit_expand_softmax = torch.softmax(logits_expanded, -1) # [batch_size, L, vocab_size]
            # kl_div = F.kl_div(mu_logits_softmax, logit_expand_softmax, reduction='batchmean') # [batch_size, L]
            kl_div = (kl_div.sum(-1) * mini_context_mask).sum(-1) / mini_context_mask.sum(-1) # [batch_size, L]
            kl_div = kl_div.mean()
            
            mask_loss = dist_probs.mean()
            
            # Calculate ratio for PPO clipping
            ratio = torch.exp(log_probs.sum(-1) - mini_old_log_probs.sum(-1))  # [batch_size]
            print("logprob", log_probs.mean())
            # print("old_log_probs", mini_old_log_probs.mean())
            # print("ratio", ratio.mean())

            # Calculate clipped loss
            surr1 = ratio * mini_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * mini_advantages
            actor_loss = -torch.mean(torch.min(surr1, surr2))

            # Claculate critic loss
            # critic_loss = F.mse_loss(mini_rewards, value)
            critic_loss = (mini_rewards - value).pow(2).mean()

            loss = 0.5 * critic_loss +  actor_loss - 0.01 * entropy + 1 * kl_div

            # Perform gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Aggregate losses and metrics
            total_loss += loss.item()
            total_ratio += ratio.mean().item()
            total_advantages += mini_advantages.mean().item()
            total_entropy += entropy.item()
            total_kl_div += kl_div.item()
            num_updates += 1

        # Return average metrics over all mini-batches
        return {
            "loss": total_loss / num_updates,
            "advantages": total_advantages / num_updates,
            "ratio": total_ratio / num_updates,
            "entropy": total_entropy / num_updates,
            "kl_div": total_kl_div / num_updates,
            "mask_loss": mask_loss.item()
        }



    def train_one_batch(self, model, input_ids, attention_mask, context_mask, optimizer, 
                        environment: Environment, num_steps=5, alpha=0.1):
        """
        Train the model for one batch using a multi-armed bandit approach with advantage estimation.
        Args:
            model: The Llama3 model to generate the logits.
            optimizer: The optimizer for updating the model parameters.
            environment: The environment to interact with.
            num_steps: Number of steps to interact with the environment.
            alpha: Learning rate for the bandit update.
        """
        states = []
        actions = []
        rewards = []
        log_probs = []
        inferred_states = []
        logits = []
        values = []
        advantages = []

        with torch.no_grad():
            # Set initial state
            state = input_ids, attention_mask, context_mask
            environment.set_state(state)
            dist, value, inferred_state = self.get_dist_critic(model, state)
            logit = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).logits[:, -1, :]  # [N, vocab_size]

            
            for _ in range(num_steps):
                action = dist.sample() # [N, L]
                
                log_prob = dist.log_prob(action) # [N, L]
                # State does not change during step
                reward = environment.get_reward(action, state) # [N]

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                inferred_states.append(inferred_state)
                logits.append(logit.clone())
                advantages.append(reward - value)
                values.append(value)
            
            # Get mean reward as value estimate 
            value = torch.stack(rewards).mean(dim=0) # [N]
            # advantage = [r - value for r in rewards] # num_steps * [N]
            # advantage = rewards

            # Convert lists to tensors
            advantages = torch.cat(advantages) # [N*num_steps,]
            actions = torch.cat(actions) # [N*num_steps, L]
            states = [torch.cat([s[i] for s in states]) for i in range(len(states[0]))]
            inferred_states = [torch.cat([s[i] for s in inferred_states]) for i in range(len(inferred_states[0]))]
            log_probs = torch.cat(log_probs) # [N*num_steps, L]
            logits = torch.cat(logits) # [N*num_steps, vocab_size]

            rewards = torch.cat(rewards)

            

        # Perform bandit-style update with advantage estimation and PPO clipping
        loss_dict = self.bandit_update(optimizer, inferred_states, actions, log_probs, rewards, advantages, logits, alpha)
        return loss_dict 

    