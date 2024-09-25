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
from torch.distributions import Bernoulli
import numpy as np
import torch.utils.checkpoint as checkpoint


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_blocks=5, bottleneck_dim=64):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        self.attention_layers = nn.ModuleList()
        self.layers = nn.ModuleList()
        for _ in range(num_blocks):
            shortcut_layers = []
            shortcut_layers.append(nn.LayerNorm(hidden_dim))
            shortcut_layers.append(nn.PReLU())
            shortcut_layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            shortcut_layers.append(nn.LayerNorm(hidden_dim))
            self.attention_layers.append(nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True, bias=False))
            self.layers.append(nn.Sequential(*shortcut_layers))

        self.output_layer= nn.Linear(hidden_dim, output_dim, bias=False)
        # self.output_w = nn.Parameter(torch.randn(hidden_dim, output_dim))
        
    def forward(self, x):
        x = self.input_layer(x)
        for idx, layer in enumerate(self.layers):
            x = x + layer(self.attention_layers[idx](x, x, x)[0]) # shortcut
        # x = F.normalize(x, p=2, dim=-1)
        # w = F.normalize(self.output_w, p=2, dim=0)
        # x = x @ w
        return self.output_layer(x)


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
    # print('log_probs.shape', log_probs.shape)

    label_safe = torch.clamp(labels, min=0)
    selected_log_probs = torch.gather(log_probs, dim=-1, index=label_safe.unsqueeze(-1)).squeeze(-1)  # (N, L)
    selected_log_probs[labels == -100] = 0
    # print('selected_log_probs.shape', selected_log_probs.shape)

    mean_log_probs = (selected_log_probs * attention_mask).sum(-1) / attention_mask.sum(-1)
    # print('mean_log_probs.shape', mean_log_probs.shape)
    # sum_log_probs = (selected_log_probs * attention_mask).sum(-1)
    # print("mean_log_probs", mean_log_probs[:1])


    return mean_log_probs, selected_log_probs


def get_causal_mask(inputs, attention_mask=None):
    """ 
    inputs: torch.Tensor, shape (batch_size, seq_len, ...)
    attention_mask: torch.Tensor, shape (batch_size, seq_len)
    """
    seq_len = inputs.size(1)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=0).to(inputs.device).unsqueeze(0) # [1, seq_len, seq_len]
    causal_mask = causal_mask.repeat(inputs.size(0), 1, 1) # [batch_size, seq_len, seq_len]
    if attention_mask is not None:
        # print("attention_mask", attention_mask.shape)
        # print("causal_mask", causal_mask.shape)
        # Ensure causal_mask matches the batch size and seq_len dimensions of attention_mask
        causal_mask = causal_mask * attention_mask.unsqueeze(1)  # [batch_size, seq_len, seq_len]
        causal_mask = causal_mask * attention_mask.unsqueeze(2)  # [batch_size, seq_len, seq_len]
    
    return causal_mask # [batch_size, seq_len, seq_len]


class MaskGeneratingModelForIMDB(nn.Module):
    def __init__(self, hidden_size=512):
        """ 
        hidden_size: int
            The hidden size of the output of the generative model, 4096 for llama3
        """
        super().__init__()

        self.hidden_size = hidden_size
        # if emb_weights is None:
        #     raise ValueError("The embedding weights must be provided to match the weights of the prediction head.")
        
        # self.lm_head = nn.Linear(hidden_size, emb_weights.size(0), bias=False)
        # self.lm_head.weight = torch.nn.Parameter(emb_weights)
        # # freeze lm_head.weight 
        # self.lm_head.weight.requires_grad = False
        # self.reduce_map = nn.Sequential(nn.Linear(4096, 2048),
        #                                 nn.ReLU(),
        #                                 nn.Linear(2048, 1024),
        #                                 nn.ReLU(),
        #                                 nn.Linear(1024, hidden_size))
        self.reduce_map = MLP(input_dim=4096, 
                               hidden_dim=512, 
                               output_dim=hidden_size, 
                               num_blocks=2, 
                               bottleneck_dim=64) 
        
        self.reduce_map_value = MLP(input_dim=4096, 
                               hidden_dim=512, 
                               output_dim=hidden_size, 
                               num_blocks=2, 
                               bottleneck_dim=64) 
        # self.reduce_map = nn.Linear(4096, hidden_size)
       
        self.policy_states_map = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                        nn.ReLU(),
                                        nn.Linear(hidden_size, hidden_size))
        # self.layer_norm_policy = nn.LayerNorm(hidden_size)

        self.policy_map = nn.Linear(hidden_size, 1)
        self.value_map = nn.Linear(4096, 1)
        
        # self.logit_scale = nn.Parameter(torch.tensor(1.0))

        # self.value_layer_norm = nn.LayerNorm(hidden_size)

        self.mask_token_id = 128009

    
    def forward(self, hidden_states: torch.Tensor, context_mask: torch.Tensor, response_mask: torch.Tensor):
        """ 
        hidden_states: torch.Tensor of shape [N, L, hidden_size]
        context_mask: torch.Tensor of shape [N, L]
        response_mask: torch.Tensor of shape [N, L]
        labels: torch.Tensor of shape [N, L]
        """
        # convert to low dimensional space
        reduced_hidden_states = self.reduce_map(hidden_states) # [N, L, hidden_size]

        policy_hidden_states = self.policy_states_map(reduced_hidden_states) # [N, L, hidden_size]
        # policy_hidden_states = self.layer_norm_policy(policy_hidden_states)
        # policy_states = pairwise_token_attention(policy_hidden_states, context_mask, response_mask, self.logit_scale) # [N, N, L]
        policy_logits = self.policy_map(policy_hidden_states).squeeze(-1) # [N, L]

        # combined_mask = torch.clamp(context_mask + response_mask, 0, 1) # [N, L]
        # value_hidden_states = policy_hidden_states * combined_mask.unsqueeze(-1) # [N, L, hidden_size]
        # value_hidden_states = (value_hidden_states * combined_mask.unsqueeze(-1)).sum(-2) / (combined_mask.unsqueeze(-1).sum(-2) + 1e-5) # [N, hidden_size]
        def get_pooled_feature(hidden_states, response_mask):
            """ 
            Obtain the last non-padded token of the sequence
            """
            # reverse
            batch_size = hidden_states.shape[0]
            device = hidden_states.device
            reverse_response_mask = torch.flip(response_mask, dims=[-1])
            first_non_pad_from_right = reverse_response_mask.argmax(-1)
            # reverse back
            first_non_pad_position = response_mask.shape[-1] - first_non_pad_from_right - 1
            first_non_pad_position = first_non_pad_position.to(device)
            pooled_feature = hidden_states[torch.arange(batch_size, device=device), first_non_pad_position]
            return pooled_feature

        value_hidden_states = get_pooled_feature(hidden_states, response_mask)
        value = self.value_map(value_hidden_states).squeeze(-1) # [N,]


        return policy_logits, value # [N, L], [N,]

    def get_dist_critic(self, model, state):
        """ 
        model: The Llama3 model to generate the logits
        state: The input state, consisting of input_ids, attention_mask, context_mask, response_mask
        """
        input_ids, attention_mask, context_mask, response_mask = state
        # Extract features from the model
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
            last_hidden_state = outputs.hidden_states[-1].float() # [N, L, hidden_size]
        
        policy_logits, value = self.forward(last_hidden_state, context_mask, response_mask=response_mask)
        # print("policy_logits_origin", policy_logits[0][:5][:10])
        # policy_logits = torch.diagonal(policy_logits, offset=0, dim1=0, dim2=1).permute(1,0) # [N, L]
        # print("policy_logits", policy_logits[0][:100])
        # policy_logits = policy_logits * self.logit_scale.exp()

        dist = Bernoulli(logits=policy_logits) # [N, L]

        return dist, value

    # def get_dist_critic_for_ppo(self, model, state):
    #     """ 
    #     model: The Llama3 model to generate the logits
    #     state: The input state, consisting of input_ids, attention_mask, context_mask, response_mask
    #     """
    #     input_ids, attention_mask, context_mask, response_mask = state
    #     # Extract features from the model
    #     with torch.no_grad():
    #         outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
    #         last_hidden_state = outputs.hidden_states[-1].float() # [N, L, hidden_size]
        
    #     policy_logits, value = self.forward(last_hidden_state, context_mask, response_mask=response_mask)
    #     diagonal_logits = torch.diagonal(policy_logits, offset=0, dim1=0, dim2=1).permute(1,0) # [N, L]
    #     # diagonal_logits = diagonal_logits * self.logit_scale.exp()

    #     dist = Bernoulli(probs=diagonal_logits) # [N, L]

    #     return dist, value, policy_logits
    
    def ppo_iter(self, mini_batch_size, states, actions, log_probs, returns, advantage, labels):
        """
        Generates mini-batches of data for Proximal Policy Optimization (PPO) algorithm.
        Parameters:
            mini_batch_size (int): The size of each mini-batch.
            states (List of Lists): The states of the environment.
            actions (Tensor): The actions taken in the environment.
            log_probs (Tensor): The logarithm of the probabilities of the actions taken.
            returns (Tensor): The expected returns for each state-action pair.
            advantage (Tensor): The advantage estimates for each state-action pair.
        Yields:
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: A mini-batch of states, actions, log probabilities,
            expected returns, and advantage estimates.
        """
        
        batch_size = actions.size(0)
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, mini_batch_size)
            state = states["input_ids"][rand_ids, :], states["attention_mask"][rand_ids, :], states["context_mask"][rand_ids, :], states["response_mask"][rand_ids, :]

            yield state, actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :], labels[rand_ids, :]

    def ppo_update(self, model, optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, labels, clip_param=0.2):
        """
        Perform PPO (Proximal Policy Optimization) update for the given number of epochs.
        Parameters:
        - optimizer (torch.optim.Optimizer): The optimizer used for updating the model.
        - ppo_epochs (int): Number of PPO update epochs.
        - mini_batch_size (int): Size of mini-batches for PPO update.
        - states (Tensor): Input states.
        - actions (Tensor): Actions taken.
        - log_probs (Tensor): Log probabilities of actions.
        - returns (Tensor): Expected returns.
        - advantages (Tensor): Advantage estimates.
        - clip_param (float, optional): Clipping parameter for PPO update. Defaults to 0.2.
        """

        for _ in range(ppo_epochs):
            for state, action, old_log_probs, return_, advantage, label in self.ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages, labels):
                input_ids, attention_mask, context_mask, response_mask = state
                mask = action

                # dist, value, policy_logits = self.get_dist_critic_for_ppo(model, state) # [N, L], [N], [N, N, L]
                dist, value = self.get_dist_critic(model, state) # [N, L], [N]
                entropy = ((dist.entropy() * context_mask).sum(-1) / context_mask.sum(-1)).mean()

                new_log_probs = (dist.log_prob(mask) * context_mask).sum(-1) / context_mask.sum(-1) # (N,)
                new_log_probs = new_log_probs.unsqueeze(-1) # (N, 1)

                # ratio = ((new_log_probs - old_log_probs) * response_mask.sum(-1, keepdim=True)).exp()
                ratio = (new_log_probs - old_log_probs).exp()

                # print("ratio", ratio.squeeze(-1).mean())
                surr1 = ratio * advantage
                # print("advantage", advantage.squeeze(-1).data)
                # print("new_log_probs", new_log_probs.squeeze(-1).data)
                # print("old_log_probs", old_log_probs.squeeze(-1).data)
                # print('ratio', ratio.squeeze(-1).data)
                # clip = clip_param / torch.sqrt(context_mask.sum(-1, keepdim=True))
                clip = clip_param * 0.1
                surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantage
                # PPO step
                actor_loss  = - torch.min(surr1, surr2).mean()
                # learn the value function based on the estimated return
                critic_loss = (return_ - value).pow(2).mean()

                mask_mean = (torch.sigmoid(dist.logits) * context_mask).sum(-1) / context_mask.sum(-1) # (N,)
                mask_loss = mask_mean.mean()
                std_mean = ((torch.sigmoid(dist.logits) - mask_mean.unsqueeze(-1)).pow(2) * context_mask).sum(-1) / context_mask.sum(-1) # (N,)
                std_loss = torch.sqrt(std_mean).mean()

                # contrast_loss = calc_contrast_loss(policy_logits, context_mask, self.logit_scale.exp())

                loss = 0.5 * critic_loss + 1 *actor_loss - 0.001 * entropy #+ 0.001 * mask_loss


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Print the losses after one ppo update
        return  {"loss": loss.item(), 
                 "actor_loss": actor_loss.item(), 
                 "critic_loss": critic_loss.item(),
                 "returns": return_.mean().item(),
                 'entropy': entropy.item(),
                 "value": value.mean().item(),
                #  "contrast_loss": contrast_loss.item(),
                 "mask_loss": mask_loss.item(),
                 "std_loss": std_loss.item()}


    @torch.no_grad()
    def compute_gae(self, next_value, rewards, masks, values, gamma=0.99, tau=0.95):
        """
        Computes the Generalized Advantage Estimation (GAE) for the given rewards, masks, values, and next value.
        Parameters:
        - next_value (Tensor): The value of the next state.
        - rewards (Tensor): The rewards received.
        - masks (Tensor): The masks of the environment.
        - values (Tensor): The value estimates.
        - gamma (float, optional): The discount factor. Defaults to 0.99.
        - tau (float, optional): The GAE parameter. Defaults to 0.95.
        Returns:
        - list: List of GAE-estimated returns for each step.
        """
        values = values + [next_value]
        gae = 0
        sum_gae=0
        returns = []
        mean_reward = sum(rewards) / len(rewards)
        # print("reward in gae:", rewards[0].shape)
        for idx, step in enumerate(reversed(range(len(rewards)))):  
            gae = rewards[step] - values[step]
            # print("gae:", gae.mean())

            returns.insert(0, gae + values[step])
        
        return returns


    @torch.no_grad()
    def get_action_reward(self, model, state, action, sim_upper, sim_lower):
        """ 
        action : mask
        state : pixel_values
        """
        mask = action
        input_ids, attention_mask, context_mask, response_mask = state

        # mask = (context_mask * mask + (1. - context_mask)) # do not mask the context tokens for prompting
        # masked_attention_mask = attention_mask * mask
        to_be_masked = (1 - mask) * context_mask

        masked_input_ids = input_ids.clone()
        masked_input_ids[to_be_masked == 1] = self.mask_token_id

        # get reward
        sim = self.calculate_sim(model, masked_input_ids, attention_mask, response_mask, labels=input_ids)
        reward = self.get_reward(sim, sim_upper, sim_lower, action, context_mask, attention_mask*(1-response_mask))
        # print("reward:", reward.mean())

        return state, reward

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
        # print('sim_shape_before', sim.shape)
        # sim = sim * response_mask
        # print('sim_shape_after', sim.shape)

        return sim
    
    @torch.no_grad()
    def get_reward(self, sim, sim_upper, sim_lower, user_input_mask, context_mask, attention_mask):
        # print('sim', sim[:1])
        # print('sim_upper', sim_upper[:1])
        # print('sim_lower', sim_lower[:1])
        # reward_raw = torch.relu(torch.exp(sim - sim_lower) - 1)
        # reward_range = torch.relu(torch.exp(sim_upper - sim_lower))
        reward_raw = torch.exp(sim - sim_upper) - torch.exp(sim_lower - sim_upper)
        # reward_range = torch.exp(sim_upper - sim_lower)
        # reward = reward / (reward2 + 1e-5)
        # reward = reward_raw #/ (reward_range + 1e-5)
        # reward = torch.exp(sim - sim_upper)
        # reward = torch.exp(sim - sim_lower) / torch.exp(sim_upper - sim_lower)
        # reward = torch.exp(sim - sim_upper)
        # reward = torch.exp(sim_gt - sim)
        factor = ((user_input_mask * context_mask).sum(-1)) / context_mask.sum(-1)
        inverse_factor = torch.where(factor == 0, torch.zeros_like(factor), 1 / factor)
        # attention_sum = attention_mask.sum(-1)
        # context_sum = context_mask.sum(-1)
        # factor = ((user_input_mask * context_mask).sum(-1) + attention_sum - context_sum) / attention_sum
        # factor = ((1 - user_input_mask) * context_mask).sum(-1) / attention_sum
        # print("reward_original", reward.mean())
        # reward = reward * (1 / (factor+1e-5))
        # reward = reward_raw / (reward_range + 1)
        reward = reward_raw
        reward = reward * inverse_factor
        # reward = reward - factor
        # reward = reward * (1 / (factor**2 + 1e-5))
        # print("reward:", reward.shape)
        print("reward:", reward.mean(), "factor:", factor.mean())

        return reward

    def train_one_batch(self, model, input_ids, attention_mask, context_mask, response_mask,
                        optimizer: torch.optim.Optimizer, num_steps=20, mini_batch_size=32, ppo_epochs=10):
        self.train() 
        log_probs = []
        values    = []
        states    =  {"input_ids":[], "attention_mask":[], "context_mask":[], "response_mask":[]}
        actions   = []
        rewards   = []
        masks     = []
        entropy = 0
        # labels = [] 


        state = input_ids, attention_mask, context_mask, response_mask
        sim_upper = self.calculate_sim(model, input_ids, attention_mask, response_mask, labels=input_ids)

        to_be_masked = context_mask
        masked_input_ids = input_ids.clone()
        masked_input_ids[to_be_masked == 1] = self.mask_token_id
        sim_lower = self.calculate_sim(model, masked_input_ids, attention_mask, response_mask, labels=input_ids)
    
        with torch.no_grad():
            for step in range(num_steps):
                dist, value = self.get_dist_critic(model, state)
                action = dist.sample()
                action = action * context_mask
                next_state, reward = self.get_action_reward(model, state, action, sim_upper, sim_lower)

                # log_prob = dist.log_prob(action).sum(-1, keepdim=True)
                log_prob = (dist.log_prob(action) * context_mask).sum(-1) / context_mask.sum(-1) # (N,)
                entropy += (dist.entropy()* context_mask).sum(-1) / context_mask.sum(-1)

                log_probs.append(log_prob.unsqueeze(-1))
                values.append(value.unsqueeze(-1))
                rewards.append(reward.unsqueeze(-1))
                if step == num_steps - 1:
                    masks.append(torch.zeros_like(reward).unsqueeze(-1))
                else:
                    masks.append(torch.ones_like(reward).unsqueeze(-1))
                
                states["input_ids"].append(input_ids)
                states["attention_mask"].append(attention_mask)
                states["context_mask"].append(context_mask)
                states["response_mask"].append(response_mask)
                actions.append(action.clone())
                # labels.append(input_ids.clone())


                state = next_state 
            
            _, next_value = self.get_dist_critic(model, state)
            
            returns = self.compute_gae(next_value, rewards, masks, values)
            # returns = self.compute_gae_static_state(next_value, rewards, masks, values)
            returns = torch.cat(returns)
            # returns = returns[0].repeat(len(rewards), 1)
            log_probs = torch.cat(log_probs)
            values    = torch.cat(values)
            actions   = torch.cat(actions)
            labels    = values

            states["input_ids"] = torch.cat(states["input_ids"])
            states["attention_mask"] = torch.cat(states["attention_mask"])
            states["context_mask"] = torch.cat(states["context_mask"])
            states["response_mask"] = torch.cat(states["response_mask"])
            # returns = returns[0]
            # log_probs = log_probs[0]
            # values    = values[0]
            # states    = states[0]
            # actions   = actions[0]
            # print("returns.shape:", returns.shape)
            # print("values.shape:", values.shape)
            advantages = returns - values
        
        # print("states", states.shape)
        # print("actions", actions.shape)
        # print("log_probs", log_probs.shape)
        # print("returns", returns.shape)
        # print("advantages", advantages.shape)
        # print("labels", labels.shape)


        loss_dict = self.ppo_update(model, optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, labels)
        return loss_dict


    
    def compute_similarity(self, logits, labels, attention_mask):
        """ 
        compute the mean log_probs for the generated tokens
        """
        mean_log_probs = similarity_measure(logits, labels, attention_mask)
        return mean_log_probs
    