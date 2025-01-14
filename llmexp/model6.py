'''
This version uses self attention instead of direct MLP to generate the masks.
other worthnoting changes
1. A layernorm and PReLU layer is added after the self-attention layer
2. Masking strategy: 
    a) Select the top k% as candidates 
    b) random sampling 60% of the candidates as the mask
'''


import torch.nn as nn 
import torch.nn.functional as F
import torch

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
    def __init__(self, hidden_size=4096, mlp_hidden_dim=1024, mlp_bottleneck_dim=768, mlp_num_blocks=2):
        """ 
        hidden_size: int
            The hidden size of the output of the generative model, 4096 for llama3
        """
        super().__init__()

        self.hidden_size = hidden_size

        self.explain_map = MLP(input_dim=hidden_size, 
                               hidden_dim=mlp_hidden_dim, 
                               output_dim=1, 
                               num_blocks=mlp_num_blocks, 
                               bottleneck_dim=mlp_bottleneck_dim) # takes [N, L, hidden_size] outputs [N, L, 1]
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

        def bce_loss(input, target, context_mask):
            loss = F.binary_cross_entropy(input, target, reduction='none')
            loss = (loss * context_mask).sum(-1) / context_mask.sum(dim=-1)

            return loss

        self.bce_loss = bce_loss
    
    def forward(self, pred_features):
        """ 
        pred_features: torch.Tensor of shape [N, L, hidden_size]
        """
        mask_logits = self.explain_map(pred_features).squeeze(-1) # [N, L]
        # mask_logits = F.normalize(mask_logits, p=2, dim=-1) # normalize the logits
        mask_logits = mask_logits * self.logit_scale.exp()
        # print("mask_logits_before", mask_logits[0])
        new_mask_logits = mask_logits.clone()

        return new_mask_logits # [batch_size, seq_len]
    
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
        # labels[:, :-1] = labels[:, 1:]
        # labels[:, -1] = pad_token_id
        mask_probs = torch.sigmoid(mask_logits) # (batch_size, seq_len)
        # mask_probs = torch.clamp(mask_probs, 0.2, 1) # remove the negative values
        # mask = torch.bernoulli(mask_probs) # (batch_size, seq_len)
        # Mask the top K value of each example in the batch
        mask_probs = mask_probs * context_mask
        # k_values = (mask_probs * context_mask).sum(-1).int()
        k_ratio = 0.5
        k_values = (k_ratio * context_mask.sum(-1)).int()
        # print(k_values)
        mask = torch.zeros_like(mask_logits)
        for i in range(mask_logits.size(0)):
            top_k_values, top_k_indices = torch.topk(mask_probs[i], k=max(1, k_values[i]), dim=-1)
            mask[i, top_k_indices] = 1
        # generate a mask of shape mask_probs with probability 0.6
        bernoulli_mask = torch.bernoulli(torch.ones_like(mask_probs) * 0.6)
        # Top k% of candidate mask then randomly sample 60% of the candidates as the mask
        mask = mask * bernoulli_mask 
        mask_ratio = mask.sum(-1) / context_mask.sum(-1)

        # print("mask_ratio", mask_ratio)

        # print(mask[0])
        # mask = mask.clone()
        # mask[:, 1:] = mask[:, :-1] # shift the mask to the left
        # mask[:, 0] = 0 # the last token should be masked
        mask = (context_mask * mask + (1. - context_mask)) # do not mask the context tokens for prompting
        # print(mask[0])
        # print(mask[0])
        return mask # this could be used as the attention mask for the generative model
    
    @torch.no_grad()
    def sample_one_batch(self, input_ids, attention_mask, mask_logits, context_mask):
        """ 
        args:
            input_ids: torch.Tensor, shape (batch_size, seq_len)
            attention_mask: torch.Tensor, shape (batch_size, seq_len), the attention_mask generated automatically by tokenizer, usually all 1s
            mask_logits: torch.Tensor, shape (batch_size, prompt_len), the logits for the mask generation
            context_mask: torch.Tensor, shape (batch_size, prompt_len), the context texts 
                (for example, the chatbot instruction template, not including the user inputs) are masked with 0.
        """
        #  get the mask of interest, i.e., for the user inputs, then pad to match the shape of other masks (of the full length)
        seq_len = input_ids.size(1)
        prompt_len = mask_logits.size(1)
        mask = self.generate_mask(mask_logits=mask_logits, context_mask=context_mask) # (batch_size, prompt_len)
        pad_length = seq_len - prompt_len
        padded_mask = F.pad(mask, (0, pad_length), mode='constant', value=1)

        # get the masked attention_mask
        masked_attention_mask = attention_mask * padded_mask

        # get the padded input_ids
        # masked_input_ids = self.tokenizer.pad_token_id * torch.ones_like(input_ids) * (1 - masked_attention_mask).long() + input_ids * masked_attention_mask.long()

        return input_ids, masked_attention_mask, mask
    
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
    def get_reward(self, sim, sim_gt):
        reward = torch.exp(sim - sim_gt)
        # print('reward', reward.mean())
        # print('sim', sim)
        # print('sim_gt', sim_gt)
        # print(reward[0])
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
        # print('mask_prob', mask_prob[0])

        # obtain the perturbed_samples
        perturbed_samples = self.sample_one_batch_multi_samples(gen_tokens, gen_attention_mask, mask_logits, context_mask, num_samples=num_samples)

        # obtain the groundtruth similarity (similarity of the generated tokens without perturbation)
        reward_loss = 0
        total_reward = 0
        sim_gt, _ = self.calculate_sim(model, gen_tokens, gen_attention_mask, response_mask, labels=gen_tokens)
        for perturbed_input_ids, perturbed_attention_mask, user_input_mask in perturbed_samples:
            # obtain the similarity of the perturbed samples
            sim, _ = self.calculate_sim(model, perturbed_input_ids, perturbed_attention_mask, response_mask, labels=gen_tokens)
            reward = self.get_reward(sim, sim_gt)
            # reward = torch.exp(sim)
            total_reward += reward
            # reward_loss += (torch.relu(reward - 0.5) * self.bce_loss(mask_prob, user_input_mask, context_mask)).mean(dim=-1)
            reward_loss += (reward * self.bce_loss(mask_prob, user_input_mask, context_mask)).mean(dim=-1)
        
        reward_loss = reward_loss / num_samples
        reward_loss = reward_loss.mean()
        mask_loss = (mask_prob * context_mask).sum(-1) / context_mask.sum(dim=-1)
        # TODO: mask loss 可能有问题，需要进一步检查， 因为测试样例中 mean_mask极小
        # mask_loss = mask_prob.mean(-1)
        mask_mean = mask_loss.mean()
        mean_reward = total_reward / num_samples
        # mask_loss = ((1.2 - mask_loss - mean_reward)**2).mean()
        # when mean_reward > 0.7, then mask_loss = 0, else mask_loss = mask_loss
        # print('mean_reward',mean_reward[0])
        # mask_loss = torch.where(mean_reward > 0.5,  mask_loss, mask_loss.detach()).mean()
        # mask_loss = torch.relu(mask_loss - mean_reward - 0.7).mean()
        mask_loss = ((0.2 - mask_loss)**2).mean()
        mean_reward = mean_reward.mean()

        loss = reward_loss #+ 1 * mask_loss
        # return {loss: loss, reward_loss: reward_loss, mask_loss: mask_loss, mask_mean: mask_mean}
        return {'loss': loss, 'reward_loss': reward_loss, 'mask_loss': mask_loss, 'mask_mean': mask_mean, 'mean_reward': mean_reward}
    
    def compute_similarity(self, logits, labels, attention_mask):
        """ 
        compute the mean log_probs for the generated tokens
        """
        mean_log_probs = similarity_measure(logits, labels, attention_mask)
        return mean_log_probs
    