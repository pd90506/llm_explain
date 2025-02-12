import torch
import torch.nn as nn
import torch.nn.functional as F
from llmexp.explainer.mab_model import MABModel
from typing import Dict, Any
import transformers
from llmexp.utils import get_mab_masked_inputs
import wandb
from tqdm import tqdm
from llmexp.utils.model_utils import GumbelKHotDistribution


class MABTrainer:
    def __init__(
        self,
        mab_model: MABModel,
        target_model: torch.nn.Module,
        tokenizer: transformers.PreTrainedTokenizer,
        config: Dict[str, Any],
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        # Initialize the MAB model
        self.mab_model = mab_model.to(device)
        # Initialize the target model
        self.target_model = target_model.to(device)

        self.tokenizer = tokenizer

        self.config = config
        self.device = device
        self.optimizer = torch.optim.Adam(self.mab_model.parameters(), lr=config['lr'])

        self.num_pulls = config['num_pulls']
        self.minibatch_size = config['minibatch_size']
        self.clip_epsilon = config['clip_epsilon']

        self.lambda_entropy = config['lambda_entropy']

        self.topk = config['topk']
        self.temperature = 1.0

        self.loss_fn = nn.MSELoss()



    def train(self, dataloader: torch.utils.data.DataLoader, gamma=0.9):
        """
        dataloader: DataLoader object
        """

        for batch_idx, batch in tqdm(enumerate(dataloader), desc="Training MAB"):
            # Generate the response from the target model
            batch = batch.to(self.device)
            generated_outputs = self.target_model.generate(batch['input_ids'], attention_mask=batch['attention_mask'])
            # Randomly cut and pad to align sequences such that the last token is the MAB label
            processed = randomly_cut_and_pad_generations(batch, generated_outputs, self.tokenizer)

            input_ids = processed['input_ids'].to(self.device)
            attention_mask = processed['attention_mask'].to(self.device)
            context_mask = processed['context_mask'].to(self.device)

            # get the probability of the correct label
            label_prob = self.get_pull_prob(input_ids, attention_mask, input_ids[:, -1])

            logits, value = self.mab_model.get_logits_value(input_ids, attention_mask)
            dist = GumbelKHotDistribution(logits=logits, context_mask=context_mask[:, :-1], k=self.topk, temperature=self.temperature)

            pulls = self.collect_pulls(input_ids, attention_mask, context_mask, label_prob, dist, value, num_pulls=self.num_pulls, gamma=gamma)

            for _ in range(self.config['ppo_epochs']):
                for minibatch in self.ppo_iter(self.minibatch_size, pulls):
                    mab_info_dict = self.train_mab_minibatch(minibatch)

            # log 

            wandb.log(mab_info_dict)

            # save the model
            if batch_idx % self.config['save_interval'] == 0:
                torch.save(self.mab_model.state_dict(), f"checkpoints/mab_model_{batch_idx}.pth")


    @torch.no_grad()
    def collect_pulls(self, 
                      input_ids: torch.Tensor, 
                      attention_mask: torch.Tensor,
                      context_mask: torch.Tensor,
                      label_prob: torch.Tensor,
                      dist: torch.distributions.Distribution,
                      value: torch.Tensor,
                      num_pulls: int = 3,
                      gamma: float = 0.99,
                      lam: float = 0.95,
                      ) -> dict:
        """Collect pull results for MAB training.
        
        """
        # Initialize lists to gather pull outputs
        pulls = [self._get_single_pull(input_ids, attention_mask, context_mask, label_prob, dist, value)
                 for _ in range(num_pulls)]
        # Stack the individual pull outputs along a new axis and flatten out the batch dimension
        rewards = torch.cat([p['reward'] for p in pulls], dim=0)
        pull_masks = torch.cat([p['pull_mask'] for p in pulls], dim=0)

        returns_list = self.compute_returns([p['reward'] for p in pulls],
                                             [p['value'] for p in pulls], gamma, lam)
        returns = torch.cat(returns_list, dim=0)
        logits = dist.logits
        batch_rep = lambda t: t.repeat(num_pulls, 1)
        return {
            'pull_masks': pull_masks,
            'input_ids': batch_rep(input_ids),
            'attention_masks': batch_rep(attention_mask),
            'context_masks': batch_rep(context_mask),
            'rewards': rewards,
            'returns': returns,
            'label_probs': batch_rep(label_prob),
            'old_logits': batch_rep(logits),
        }

    def _get_single_pull(self, input_ids, attention_mask, context_mask, label_prob, dist, value):
        # Sample a pull mask and compute related reward and log probability
        pull_mask = dist.sample()

        reward = self.get_pull_reward_prob(input_ids, attention_mask, context_mask, label_prob, pull_mask)
        return {
            'reward': reward.clone(),
            'pull_mask': pull_mask.clone(),
            'value': value.clone(),
        }

    def compute_returns(self, rewards: list[torch.Tensor], 
                        value: list[torch.Tensor],
                        gamma: float = 0.99,
                        lam: float = 0.95) -> list:
        """
        Compute the returns for the rewards
        """
        returns = []
        last_advantage = 0
        # Iterate backwards over the rewards
        for r, v in zip(reversed(rewards), reversed(value)):
            delta = r + gamma * v - v
            last_advantage = delta + gamma * lam * last_advantage
            returns.insert(0, (last_advantage + v))
        return returns

    def ppo_iter(self, minibatch_size: int, pulls: Dict):
        """
        minibatch_size: int
        pulls: Dict
        """
        keys = list(pulls.keys())
        batch_size = pulls[keys[0]].shape[0]

        for _ in range(batch_size // minibatch_size):
            rand_ids = torch.randint(0, batch_size, (minibatch_size,), device=self.device)
            yield {key: value[rand_ids] for key, value in pulls.items()}
    
    def train_mab_minibatch(self, minibatch: Dict):
        """
        minibatch: Dict
        """
        logits, value = self.mab_model.get_logits_value(minibatch['input_ids'], minibatch['attention_masks'])
        dist = GumbelKHotDistribution(logits=logits, context_mask=minibatch['context_masks'][:, :-1], k=self.topk, temperature=self.temperature)

        pull_masks = minibatch['pull_masks'] # [batch_size, sequence_length-1]
        context_masks = minibatch['context_masks'] # [batch_size, sequence_length]
        _context_mask = context_masks[:, :-1] # [batch_size, sequence_length-1]

        # log_probs = minibatch['log_probs'] # [batch_size, 1]
        # returns = minibatch['returns'] # [batch_size, 1]
        reward = minibatch['rewards'] # [batch_size, 1]
        old_logits = minibatch['old_logits'] # [batch_size, sequence_length-1]

        actor_loss = self.get_mab_actor_loss(dist, old_logits, value, context_masks, pull_masks, reward)
        # critic_loss = (returns - value).pow(2).mean()

        # entropy = (dist.entropy() * _context_mask.float()).sum(dim=-1, keepdim=True) / _context_mask.sum(dim=-1, keepdim=True) # [batch_size, 1]
        entropy = dist.entropy()
        entropy = entropy.mean()

        mask_ratio = (pull_masks * _context_mask).sum(-1).float() / _context_mask.sum(-1).float() # [batch_size]
        mask_ratio = mask_ratio.mean()

        # # limit dist prob to close to 0.5
        # dist_probs = (dist.probs * _context_mask.float()).sum(-1, keepdim=True) / _context_mask.sum(-1, keepdim=True) # [batch_size, 1]
        # prob_loss = (dist_probs - 0.5).pow(2).mean()

        loss = actor_loss - self.lambda_entropy * entropy #+ 0.1 * prob_loss

        # update the mab model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



        return {
            'loss': loss.item(),
            'actor_loss': actor_loss.item(),
            # 'critic_loss': critic_loss.item(),
            'entropy': entropy.item(),
            'mask_ratio': mask_ratio.item(),
            # 'returns': returns.mean().item(),
            'reward': reward.mean().item(),
            'entropy': entropy.item(),
            'value': value.mean().item(),
        }


    def get_mab_actor_loss(self, dist: torch.distributions.Distribution, 
                           old_logits: torch.Tensor,
                           value: torch.Tensor, 
                           context_masks: torch.Tensor,
                           pull_masks: torch.Tensor, 
                           reward: torch.Tensor):
        """ 
        dist: torch.distributions.Distribution
        value: [batch_size, 1]
        context_masks: [batch_size, sequence_length]
        pull_masks: [batch_size, sequence_length-1]
        reward: [batch_size, 1]
        """

        # Calculate the advantage
        # advantage = returns - value.detach() 
        context_masks = context_masks[:, :-1]
        pull_masks = pull_masks.detach() * context_masks

        # get new logits 
        new_logits = dist.logits 
        new_logits_masked = new_logits * context_masks
        
        # get old logits
        delta = 100 * reward * pull_masks
        # Where delta is 0, use old_logits. Where delta is non-zero, use delta
        updated_old_logits = torch.where(delta == 0, old_logits, delta)
        # Optional: Apply exponential moving average
        updated_old_logits = 0.8 * old_logits + 0.2 * updated_old_logits

        actor_loss = self.loss_fn(new_logits_masked * context_masks, updated_old_logits.detach() * context_masks)

        return actor_loss



    def get_second_last_token_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Get the logits for the last token.
        """
        logits_all = self.target_model(input_ids, attention_mask).logits
        logits = logits_all[:, -2, :] # [batch_size, vocab_size] the second last output is the logits for predicting the last token
        return logits

    
    def is_pull_correct(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, label: torch.Tensor):
        """
        input_ids: [batch_size, sequence_length]
        attention_mask: [batch_size, sequence_length]
        label: [batch_size]
        """
        pred = self.get_second_last_token_logits(input_ids, attention_mask) # [batch_size, vocab_size]
        pred = torch.argmax(pred, -1) # [batch_size]
        correct = (pred == label).float() # [batch_size]
        return correct
    
    @torch.no_grad()
    def get_pull_reward_acc(self, 
                          input_ids: torch.Tensor, 
                          attention_mask: torch.Tensor,
                          context_mask: torch.Tensor,
                          pull_mask: torch.Tensor):
        """
        input_ids: [batch_size, sequence_length]
        attention_mask: [batch_size, sequence_length]
        context_mask: [batch_size, sequence_length]
        pull_mask: [batch_size, sequence_length-1]
        """
        # We heed the pull mask to be as small as possible 
        pull_mask = pull_mask * context_mask[:, :-1] # [batch_size, sequence_length-1]
        pull_ratio = pull_mask.sum(-1).float() / context_mask[:, :-1].sum(-1).float() # [batch_size] 

        # Calculate the token prediction accuracy
        masked_inputs = get_mab_masked_inputs(input_ids, attention_mask, context_mask, pull_mask, self.tokenizer)
        masked_input_ids = masked_inputs['input_ids']
        masked_attention_mask = masked_inputs['attention_mask']

        # correct = 1 or 0
        correct = self.is_pull_correct(masked_input_ids, masked_attention_mask, label=input_ids[:, -1]) # [batch_size]

        # reward =  (1-correct) * pull_ratio # [batch_size]
        reward = correct / (pull_ratio+1) # [batch_size]

        return reward
    
    @torch.no_grad()
    def get_pull_prob(self, 
                      input_ids: torch.Tensor, 
                      attention_mask: torch.Tensor,
                      label: torch.Tensor):
        """
        input_ids: [batch_size, sequence_length]
        attention_mask: [batch_size, sequence_length]
        label: [batch_size]
        """
        logits = self.get_second_last_token_logits(input_ids, attention_mask) # [batch_size, vocab_size]
        prob = F.softmax(logits, -1) # [batch_size, vocab_size]
        # get the probability of the correct label 
        prob = torch.gather(prob, -1, label.unsqueeze(-1)) # [batch_size, 1]
        return prob # [batch_size, 1]
    
    @torch.no_grad()
    def get_pull_reward_prob(self, 
                            input_ids: torch.Tensor, 
                            attention_mask: torch.Tensor,
                            context_mask: torch.Tensor,
                            label_prob: torch.Tensor,
                            pull_mask: torch.Tensor):
        """
        input_ids: [batch_size, sequence_length]
        attention_mask: [batch_size, sequence_length]
        context_mask: [batch_size, sequence_length]
        label_prob: [batch_size, 1]
        pull_mask: [batch_size, sequence_length-1]
        """
        # We heed the pull mask to be as small as possible 
        pull_mask = pull_mask * context_mask[:, :-1] # [batch_size, sequence_length-1]
        # pull_ratio = (pull_mask.sum(-1, keepdim=True).float()) / (context_mask[:, :-1].sum(-1, keepdim=True).float()) # [batch_size, 1] 

        # Calculate the token prediction probability
        masked_inputs = get_mab_masked_inputs(input_ids, attention_mask, context_mask, pull_mask, self.tokenizer)
        masked_input_ids = masked_inputs['input_ids']
        masked_attention_mask = masked_inputs['attention_mask']

        #  probability of the correct label
        prob = self.get_pull_prob(masked_input_ids, masked_attention_mask, label=input_ids[:, -1])# [batch_size, 1]

        reward = label_prob - prob  # the degradation after masking
        return reward


    def epsilon_sampling(self, dist: torch.distributions.Distribution, epsilon: float = 0.2):
        """
        dist: torch.distributions.Distribution
        epsilon: float
        """
        if torch.rand(1) < epsilon:
            return torch.bernoulli(torch.full_like(dist.probs, 0.9, device=dist.probs.device))
        else:
            return dist.sample()
    



def randomly_cut_and_pad_generations(inputs, generated_outputs, tokenizer):
    """
    Randomly truncates generated sequences and pads them to a uniform length.

    Args:
        inputs (dict): Original input sequences containing:
            - input_ids (torch.Tensor): Input token IDs [batch_size, input_length]
            - attention_mask (torch.Tensor): Input attention masks [batch_size, input_length]
        generated_outputs (dict): Model-generated sequences containing:
            - input_ids (torch.Tensor): Generated token IDs [batch_size, total_length]
            - attention_mask (torch.Tensor): Generated attention masks [batch_size, total_length]
        tokenizer: Tokenizer object with pad_token_id attribute

    Returns:
        dict: Processed sequences containing:
            - input_ids (torch.Tensor): Padded token IDs [batch_size, max_length]
            - attention_mask (torch.Tensor): Padded attention masks [batch_size, max_length]

    Note:
        For each sequence in the batch:
        1. Takes the original input and a random portion of the generated text
        2. Concatenates them together
        3. Left-pads all sequences in the batch to the same length
    """
    
    input_length = inputs['input_ids'].shape[1]
    batch_size = inputs['input_ids'].shape[0]
    device = inputs['input_ids'].device
    
    # Get the generated portions
    generated_portions = generated_outputs['input_ids'][:, input_length:]
    generated_masks = generated_outputs['attention_mask'][:, input_length:]
    generated_context_masks = generated_masks.clone()
    generated_context_masks[:, :2] = 0 # the first two tokens are \n
    
    cut_sequences = []
    cut_masks = []
    cut_context_masks = []
    max_length = 0
    
    for i in range(batch_size):
        # Use attention mask to determine valid generated tokens
        gen_length = generated_masks[i].sum().item()
        if gen_length > 0:
            # Randomly choose cut point
            # Note there are two \n start the generated sequence, so we need to cut after the second \n
            cut_point = torch.randint(2, gen_length+1, (1,)).item()
            # Combine input sequence with cut generated sequence
            full_seq = torch.cat([
                inputs['input_ids'][i],
                generated_portions[i][:cut_point]
            ])
            full_mask = torch.cat([
                inputs['attention_mask'][i],
                generated_masks[i][:cut_point]
            ])
            full_context_mask = torch.cat([         
                inputs['context_mask'][i],
                generated_context_masks[i][:cut_point]
            ])
        else:
            # If no generation, just use input sequence
            full_seq = inputs['input_ids'][i]
            full_mask = inputs['attention_mask'][i]
            full_context_mask = inputs['context_mask'][i]
            
        cut_sequences.append(full_seq)
        cut_masks.append(full_mask)
        cut_context_masks.append(full_context_mask)
        max_length = max(max_length, len(full_seq))
    
    # Left pad all sequences to max_length
    padded_sequences = torch.full((batch_size, max_length), 
                                tokenizer.pad_token_id,
                                dtype=inputs['input_ids'].dtype,
                                device=device)
    attention_masks = torch.zeros((batch_size, max_length),
                                dtype=inputs['attention_mask'].dtype,
                                device=device)
    
    new_context_masks = torch.zeros_like(attention_masks).to(device)
    # Fill in the sequences from the right
    for i, (seq, mask, context_mask) in enumerate(zip(cut_sequences, cut_masks, cut_context_masks)):
        seq_len = len(seq)
        start_idx = max_length - seq_len
        padded_sequences[i, start_idx:] = seq
        attention_masks[i, start_idx:] = mask  # Use the original attention mask values
        new_context_masks[i, start_idx:] = context_mask
    
    return {
        'input_ids': padded_sequences,
        'attention_mask': attention_masks,
        'context_mask': new_context_masks
    }

