import torch.nn as nn 
# from models import MLP
import torch.nn.functional as F
import torch

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_blocks=5, bottleneck_dim=64):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(num_blocks):
            shortcut_layers = []
            shortcut_layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            shortcut_layers.append(nn.Dropout())
            shortcut_layers.append(nn.ReLU())  # Using ReLU for simplicity; you can choose other activations as needed
            shortcut_layers.append(nn.Linear(bottleneck_dim, bottleneck_dim))
            shortcut_layers.append(nn.Dropout())
            shortcut_layers.append(nn.ReLU())
            shortcut_layers.append(nn.Linear(bottleneck_dim, hidden_dim))
            shortcut_layers.append(nn.Dropout())
            self.layers.append(nn.Sequential(*shortcut_layers))

        self.output_layer= nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.input_layer(x)
        x = self.self_attention(x, x, x)[0] # self-attention
        x = self.layernorm(x)
        for layer in self.layers:
            x = x + layer(x) # shortcut
        x = F.normalize(x, p=2, dim=-1)
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
            # input = input * context_mask
            # target = (target * context_mask).long()
            # print('target', target[0])
            # print('input', input[0])
            # loss = - target * torch.log(input)  - (1 - target) * torch.log(1 - input)
            # print('loss', loss)
            loss = F.binary_cross_entropy(input, target, reduction='none')
            # print('loss', loss)
            loss = (loss * context_mask).sum(-1) / context_mask.sum(dim=-1)
            # print('bce_loss', loss)

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
        # new_mask_logits[:, :-1] = mask_logits[:, 1:] # shift the mask to the left
        # new_mask_logits[:, -1] = -1e9 # the last token should be masked
        # print("mask_logits_after", mask_logits[0])
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
        mask = torch.bernoulli(mask_probs) # (batch_size, seq_len)
        # mask = mask.clone()
        # mask[:, 1:] = mask[:, :-1] # shift the mask to the left
        # mask[:, 0] = 0 # the last token should be masked
        mask = context_mask * mask + (1. - context_mask) # (batch_size, seq_len)
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
            total_reward += reward
            reward_loss += (reward * self.bce_loss(mask_prob, user_input_mask, context_mask)).mean(dim=-1)
        reward_loss = reward_loss / num_samples
        reward_loss = reward_loss.mean()
        mask_loss = (mask_prob * context_mask).sum(-1) / context_mask.sum(dim=-1)
        # mask_loss = mask_prob.mean(-1)
        mask_mean = mask_loss.mean()
        mean_reward = (total_reward / num_samples).mean()
        # mask_loss = ((0.5 - mask_loss)**2).mean()
        mask_loss = torch.relu(mask_loss - 0.5).mean()

        loss = reward_loss + 0.1 * mask_loss
        return {loss: loss, reward_loss: reward_loss, mask_loss: mask_loss, mask_mean: mask_mean, mean_reward: mean_reward}
    
    def compute_similarity(self, logits, labels, attention_mask):
        """ 
        compute the mean log_probs for the generated tokens
        """
        mean_log_probs = similarity_measure(logits, labels, attention_mask)
        return mean_log_probs
    

