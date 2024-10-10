'''
This version uses self attention instead of direct MLP to generate the masks.
other worthnoting changes
1. this version prepend the response to the context, 
'''


import torch.nn as nn 
import torch.nn.functional as F
import torch


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
        reward = torch.exp(log_prob_predictions)
        return reward


class AutoregressivePredictor(nn.Module):
    def __init__(self, hidden_size, vocab_size, embedding_layer=None):
        super(AutoregressivePredictor, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # Linear layer to project hidden state to vocabulary logits (tied to embedding layer)
        if embedding_layer is not None:
            self.output_layer = nn.Linear(hidden_size, vocab_size)
            self.output_layer.weight = embedding_layer.weight.t()  # Tie weights
            self.output_layer.weight.requires_grad = False  # Freeze weights
        else:
            self.output_layer = nn.Linear(hidden_size, vocab_size)

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
        previous_tokens = hidden_state[:, :-1, :]  # [N, L-1, d]
        last_token = hidden_state[:, -1, :]  # [N, d]

        # Compute the dot product (KQ similarity) between each of the previous tokens and the last token
        similarity_scores = torch.bmm(previous_tokens, last_token.unsqueeze(-1)).squeeze(-1)  # [N, L-1]

        # Project to vocabulary size for each previous token
        logits = self.output_layer(previous_tokens)  # [N, L-1, vocab_size]
        
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

        # Generate predictions using autoregressive predictor
        predictions = self.autoregressive_predictor(hidden_states)  # [N, L, vocab_size]

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
        # Extract features from the model
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
            last_hidden_state = outputs.hidden_states[-1].float()  # [N, L, hidden_size]
        
        predictions, value = self.forward(last_hidden_state, attention_mask, context_mask)
        inferred_state = last_hidden_state, attention_mask, context_mask
        
        dist = torch.distributions.Bernoulli(logits=predictions)  # [N, L]
        return dist, value, inferred_state # [N, L], [N]


    def bandit_update(self, optimizer, inferred_states, actions, old_log_probs, advantages, alpha=0.1, clip_param=0.2):
        """
        Perform bandit-style update with advantage estimation and PPO clipping.
        Args:
            model: The Llama3 model to generate the logits.
            optimizer: The optimizer for updating the model parameters.
            states: Input states consisting of input_ids, attention_mask, and context_mask.
            actions: Actions taken by the agent.
            advantages: Advantages obtained for each action.
            alpha: Learning rate for the bandit update.
            clip_param: Clipping parameter for PPO update.
        """
        # Calculate log probabilities for the actions taken
        # hidden_states # [N*num_steps, L, hidden_size]
        # attention_mask # [N*num_steps, L]
        # context_mask # [N*num_steps, L]
        hidden_states, attention_mask, context_mask = inferred_states
        dist, _ = self.forward(hidden_states, attention_mask, context_mask)
        log_probs = dist.log_prob(actions)
        old_log_probs = old_log_probs.detach()

        # Calculate ratio for PPO clipping
        ratio = torch.exp(log_probs - old_log_probs)
        
        # Calculate clipped loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
        loss = -torch.mean(torch.min(surr1, surr2))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


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


        # Set initial state
        state = input_ids, attention_mask, context_mask
        environment.set_state(state)
        dist, _, inferred_state = self.get_dist_critic(model, state)
        
        
        for _ in range(num_steps):
            action = dist.sample() # [N, L]
            log_prob = dist.log_prob(actions) # [N, L]
            # State does not change during step
            reward = environment.get_reward(action, state) # [N]

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            inferred_states.append(inferred_state)
        
        # Get mean reward as value estimate 
        value = torch.stack(rewards).mean(dim=0) # [N]
        advantage = [r - value for r in rewards] # num_steps * [N]

        # Convert lists to tensors
        advantages = torch.stack(advantage) # [N*num_steps,]
        actions = torch.cat(actions) # [N*num_steps, L]
        states = [torch.cat([s[i] for s in states]) for i in range(len(states[0]))]
        inferred_states = [torch.cat([s[i] for s in inferred_states]) for i in range(len(inferred_states[0]))]
        log_probs = torch.cat(log_probs) # [N*num_steps, L]

        # Perform bandit-style update with advantage estimation and PPO clipping
        self.bandit_update(model, optimizer, inferred_states, actions, log_probs, advantages, alpha)


    # @torch.no_grad()
    # def compute_gae(self, next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    #     """
    #     Computes the Generalized Advantage Estimation (GAE) for the given rewards, masks, values, and next value.
    #     Parameters:
    #     - next_value (Tensor): The value of the next state.
    #     - rewards (Tensor): The rewards received.
    #     - masks (Tensor): The masks of the environment.
    #     - values (Tensor): The value estimates.
    #     - gamma (float, optional): The discount factor. Defaults to 0.99.
    #     - tau (float, optional): The GAE parameter. Defaults to 0.95.
    #     Returns:
    #     - list: List of GAE-estimated returns for each step.
    #     """
    #     values = values + [next_value]
    #     gae = 0
    #     sum_gae=0
    #     returns = []
    #     mean_reward = sum(rewards) / len(rewards)
    #     # print("mean_reward:", mean_reward.shape)
    #     # mean_reward = rewards.mean(dim=-1)
    #     # print("reward in gae:", rewards[0].shape)
    #     for idx, step in enumerate(reversed(range(len(rewards)))):  
    #         # gae = rewards[step] - values[step]
    #         # gae = rewards[step] - 0.1* mean_reward - 0.9 * values[step]
    #         # delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
    #         # delta = rewards[step] - values[step]
    #         # delta = rewards[step] - 0.5 * mean_reward - 0.5 * values[step]
    #         # gae = rewards[step] - gamma * mean_reward - (1 - gamma) * values[step]
    #         gae = rewards[step] - values[step]
    #         # gae = rewards[step] - mean_reward
    #         # gae = delta + gamma * tau * masks[step] * gae
    #         # gae = delta
    #         # print("gae:", gae.mean())

    #         # returns.insert(0, gae + values[step])
    #         returns.insert(0, gae + values[step])
        
    #     return returns


    # @torch.no_grad()
    # def get_action_reward(self, model, state, action, correct_logprob_upper, correct_logprob_lower, new_response_mask):
    #     """ 
    #     action : mask
    #     state : pixel_values
    #     """
    #     mask = action
    #     input_ids, attention_mask, context_mask, response_mask = state

    #     # mask = (context_mask * mask + (1. - context_mask)) # do not mask the context tokens for prompting
    #     # masked_attention_mask = attention_mask * mask
    #     to_be_masked = (1 - mask) * context_mask

    #     masked_input_ids = input_ids.clone()
    #     masked_input_ids[to_be_masked == 1] = self.mask_token_id

    #     # get reward
    #     sim, correct_logprob, _, _ = self.calculate_sim(model, masked_input_ids, attention_mask, response_mask, labels=input_ids)
    #     reward = self.get_reward(correct_logprob, correct_logprob_upper, correct_logprob_lower, action, context_mask, attention_mask, new_response_mask)
    #     # print("reward:", reward.mean())

    #     return state, reward

    # @torch.no_grad()
    # def calculate_sim(self, model, input_ids, attention_mask, response_mask, labels):
    #     outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    #     logits = outputs.logits.float()
    #     # shift labels and response_mask to the left by one to match the next token prediction format
    #     labels = labels.clone()
    #     labels[:, :-1] = labels[:, 1:]
    #     labels[:, -1] = -100 # a random value that will be masked by the shift_response_mask
    #     shift_response_mask = response_mask.clone()
    #     shift_response_mask[:, :-1] = shift_response_mask[:, 1:]
    #     shift_response_mask[:, -1] = 0 # mast be zero.

    #     sim, correct_logprob, new_response_mask = similarity_measure(logits, labels, shift_response_mask)
    #     # print('sim_shape_before', sim.shape)
    #     # sim = sim * response_mask
    #     # print('sim_shape_after', sim.shape)

    #     return sim, correct_logprob, shift_response_mask, new_response_mask
    
    # @torch.no_grad()
    # def get_reward(self, correct_logprob, correct_logprob_upper, correct_logprob_lower, user_input_mask, context_mask, attention_mask, new_response_mask):
    #     # reward_raw = torch.exp(sim - sim_upper) - torch.exp(sim_lower - sim_upper)
    #     # reward_raw = torch.exp(sim)
    #     # print("reward_raw:", reward_raw)
    #     # reward_raw = torch.exp(sim) - torch.exp(sim_lower)
    #     # reward_scale = torch.exp(sim_upper - sim_lower)
    #     # print("reward_raw:", reward_raw.mean())
    #     factor = ((user_input_mask * context_mask).sum(-1)) / context_mask.sum(-1)
    #     num_attent_non_context_tokens = attention_mask.sum(-1) - context_mask.sum(-1) - new_response_mask.sum(-1)
    #     # factor = ((user_input_mask * context_mask).sum(-1) + num_attent_non_context_tokens) / (context_mask.sum(-1) + num_attent_non_context_tokens)
    #     # inverse_factor = torch.where(factor <= 0.2, torch.ones_like(factor), 1 / factor)
    #     # inverse_factor = torch.where(reward_raw < 0.1, torch.zeros_like(factor), 1 / factor)
    #     inverse_factor = 1 / factor
    #     comp_factor = 1 - factor
    #     # reward_raw = torch.exp(sim / ((user_input_mask * context_mask).sum(-1) + num_attent_non_context_tokens))

    #     distance = ((correct_logprob.exp() ) * new_response_mask).sum(-1) / new_response_mask.sum(-1)
    #     reward_raw = distance #* inverse_factor
    #     reward = distance + comp_factor
    #     # distance = (distance + 0.5*comp_factor.unsqueeze(-1)) * response_mask
    #     # reward_raw = distance.sum(-1) / response_mask.sum(-1)
    #     # reward = distance.sum(-1) / response_mask.sum(-1)
    #     # distance = torch.where(distance < 0.5, torch.zeros_like(distance), distance * inverse_factor.unsqueeze(-1))

    #     # print("distance:", distance[0])

    #     # print("reward:", reward.shape)
    #     # print("sim:", sim.mean(), "sim_upper:", sim_upper.mean(), "sim_lower:", sim_lower.mean())
    #     print("reward_raw:", reward_raw.mean(), "reward:", reward.mean(), "factor:", factor.mean())

    #     return reward

    # def train_one_batch(self, model, input_ids, attention_mask, context_mask, response_mask,
    #                     optimizer: torch.optim.Optimizer, num_steps=20, mini_batch_size=32, ppo_epochs=10):
    #     self.train() 
    #     log_probs = []
    #     values    = []
    #     states    =  {"input_ids":[], "attention_mask":[], "context_mask":[], "response_mask":[]}
    #     actions   = []
    #     rewards   = []
    #     masks     = []
    #     entropy = 0
    #     # labels = [] 


    #     state = input_ids, attention_mask, context_mask, response_mask
    #     sim_upper, correct_logprob_upper, _, _ = self.calculate_sim(model, input_ids, attention_mask, response_mask, labels=input_ids)

    #     to_be_masked = context_mask
    #     masked_input_ids = input_ids.clone()
    #     masked_input_ids[to_be_masked == 1] = self.mask_token_id
    #     sim_lower, correct_logprob_lower, shift_response_mask, new_response_mask  = self.calculate_sim(model, masked_input_ids, attention_mask, response_mask, labels=input_ids)
    
    #     with torch.no_grad():
    #         for step in range(num_steps):
    #             dist, value = self.get_dist_critic(model, state)
    #             action = dist.sample()
    #             action = action * context_mask
    #             next_state, reward = self.get_action_reward(model, state, action, correct_logprob_upper, correct_logprob_lower, new_response_mask)

    #             # log_prob = dist.log_prob(action).sum(-1, keepdim=True)
    #             log_prob = (dist.log_prob(action) * context_mask).sum(-1) #/ context_mask.sum(-1) # (N,)
    #             # log_prob = dist.log_prob(action) # (N, L)
    #             entropy += (dist.entropy()* context_mask).sum(-1) / context_mask.sum(-1)

    #             log_probs.append(log_prob.unsqueeze(-1))
    #             values.append(value.unsqueeze(-1))
    #             rewards.append(reward.unsqueeze(-1))
    #             if step == num_steps - 1:
    #                 masks.append(torch.zeros_like(reward).unsqueeze(-1))
    #             else:
    #                 masks.append(torch.ones_like(reward).unsqueeze(-1))
                
    #             states["input_ids"].append(input_ids)
    #             states["attention_mask"].append(attention_mask)
    #             states["context_mask"].append(context_mask)
    #             states["response_mask"].append(response_mask)
    #             actions.append(action.clone())
    #             # labels.append(input_ids.clone())


    #             state = next_state 
            
    #         _, next_value = self.get_dist_critic(model, state)
            
    #         returns = self.compute_gae(next_value, rewards, masks, values)
    #         # returns = self.compute_gae_static_state(next_value, rewards, masks, values)
    #         returns = torch.cat(returns)
    #         # returns = returns[0].repeat(len(rewards), 1)
    #         log_probs = torch.cat(log_probs)
    #         values    = torch.cat(values)
    #         actions   = torch.cat(actions)
    #         labels    = values

    #         states["input_ids"] = torch.cat(states["input_ids"])
    #         states["attention_mask"] = torch.cat(states["attention_mask"])
    #         states["context_mask"] = torch.cat(states["context_mask"])
    #         states["response_mask"] = torch.cat(states["response_mask"])
    #         # returns = returns[0]
    #         # log_probs = log_probs[0]
    #         # values    = values[0]
    #         # states    = states[0]
    #         # actions   = actions[0]
    #         # print("returns.shape:", returns.shape)
    #         # print("values.shape:", values.shape)
    #         advantages = returns - values

    #     loss_dict = self.ppo_update(model, optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, labels)
    #     return loss_dict

    
