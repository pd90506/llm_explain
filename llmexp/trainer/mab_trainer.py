import torch
import torch.nn as nn
import torch.nn.functional as F
from llmexp.explainer.mab_model import MABModel
from typing import Dict, Any
import transformers
from llmexp.utils import get_mab_masked_inputs
import wandb
from tqdm import tqdm


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

        self.loss_fn = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def train(self, dataloader: torch.utils.data.DataLoader, gamma=0.9):
        """
        dataloader: DataLoader object
        """

        num_pulls = self.num_pulls
        for batch_idx, batch in tqdm(enumerate(dataloader), desc="Training MAB"):
            # Generate the response from the target model
            batch = batch.to(self.device)
            generated_outputs = self.target_model.generate(batch['input_ids'], attention_mask=batch['attention_mask'])
            # Randomly cut and pad to align sequences such that the last token is the MAB label
            random_cut_generated_outputs = randomly_cut_and_pad_generations(batch, generated_outputs, self.tokenizer)

            input_ids = random_cut_generated_outputs['input_ids'].to(self.device)
            attention_mask = random_cut_generated_outputs['attention_mask'].to(self.device)

            dist, value = self.mab_model.get_dist_value(input_ids, attention_mask)

            pulls = self.collect_pulls(input_ids, attention_mask, dist, value, num_pulls)

            ppo_iterator = self.ppo_iter(self.minibatch_size, pulls)
            for minibatch in ppo_iterator:
                self.train_ppo_minibatch(minibatch)
            
            # loss = mab_value_loss - 0.00001 *entropy
            loss = surr_loss + 0.5 * mse_loss - 0.0001 * entropy

            # Update the MAB model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # log 
            profits = torch.stack(profits_list, dim=1) # [batch_size, num_pulls, 1]
            pull_masks = torch.stack(pull_masks_list, dim=1) # [batch_size, num_pulls, sequence_length-1]
            mask_ratio = pull_masks.mean().item()
            
            wandb.log({
                'surr_loss': surr_loss.item(),
                'mse_loss': mse_loss.item(),
                'entropy': entropy.item(),
                'loss': loss.item(),
                'profits': profits.mean().item(),
                'value': value.mean().item(),
                'mask_ratio': mask_ratio,
            })

            # save the model
            if batch_idx % 100 == 0:
                torch.save(self.mab_model.state_dict(), f"checkpoints/mab_model_{batch_idx}.pth")


    @torch.no_grad()
    def collect_pulls(self, 
                      input_ids: torch.Tensor, 
                      attention_mask: torch.Tensor,
                      dist: torch.distributions.Distribution,
                      value: torch.Tensor,
                      num_pulls: int = 3,
                      ) -> dict:
        """Collect pull results for MAB training.
        
        """
        # Initialize lists for PPO
        log_probs, pull_masks, rewards = [], [], []
                            
        # Collect pulls
        for _ in range(num_pulls):
            pull_mask = dist.sample() # [batch_size, sequence_length-1]
            rewards = self.get_pull_reward_acc(input_ids, attention_mask, pull_mask).unsqueeze(-1) # [batch_size, 1]
            log_prob = dist.log_prob(pull_mask).sum(-1, keepdim=True) # [batch_size, 1]
            returns = self.get_returns(rewards, value, num_pulls)            
            log_probs.append(log_prob.clone())
            rewards.append(rewards.clone())
            pull_masks.append(pull_mask.clone())
        
        log_probs = torch.cat(log_probs, dim=0) # [batch_size * num_pulls, 1]
        rewards = torch.cat(rewards, dim=0) # [batch_size * num_pulls, 1]
        pull_masks = torch.cat(pull_masks, dim=0) # [batch_size * num_pulls, sequence_length-1]
        input_ids = input_ids.repeat(num_pulls, 1) # [batch_size * num_pulls, sequence_length]
        attention_masks = attention_mask.repeat(num_pulls, 1) # [batch_size * num_pulls, sequence_length]
        return {
            'log_probs': log_probs,
            'pull_masks': pull_masks,
            'input_ids': input_ids,
            'attention_masks': attention_masks,
            'rewards': rewards,
        }
    
    def get_returns(self, rewards: torch.Tensor, value: torch.Tensor, num_pulls: int):
        """
        rewards: [batch_size, 1]
        value: [batch_size, 1]
        num_pulls: int
        """
        returns = rewards * (1/num_pulls) + value * (1-1/num_pulls)
        return returns

    def ppo_iter(self, minibatch_size: int, pulls: Dict):
        """
        minibatch_size: int
        pulls: Dict
        """
        keys = pulls.keys() 
        batch_size = pulls[keys[0]].shape[0]

        for _ in range(batch_size // minibatch_size):
            rand_ids = torch.randint(0, batch_size, (minibatch_size,), device=self.device)
            yield {key: value[rand_ids] for key, value in pulls.items()}
    
    def train_ppo_minibatch(self, minibatch: Dict):
        """
        minibatch: Dict
        """
        dist, value = self.mab_model.get_dist_value(minibatch['input_ids'], minibatch['attention_masks'])

        input_ids = minibatch['input_ids'] # [batch_size, sequence_length]
        attention_masks = minibatch['attention_masks'] # [batch_size, sequence_length]
        pull_masks = minibatch['pull_masks'] # [batch_size, sequence_length-1]
        rewards = minibatch['rewards'] # [batch_size, 1]
        log_probs = minibatch['log_probs'] # [batch_size, 1]

        surr_loss = self.get_mab_actor_loss(dist, value, pull_masks, rewards, log_probs)
        critic_loss = (returns - value).pow(2).mean()


    def get_mab_actor_loss(self, dist: torch.distributions.Distribution, 
                            value: torch.Tensor, 
                            pull_masks: torch.Tensor, 
                            rewards: torch.Tensor, 
                            log_probs: torch.Tensor):
        """ 
        dist: torch.distributions.Distribution
        value: [batch_size, 1]
        pull_masks: [batch_size, sequence_length-1]
        rewards: [batch_size, 1]
        log_probs: [batch_size, 1]
        """

        # Calculate the advantage
        advantage = rewards - value.clone().detach() 

        # Calculate the ppo loss 
        new_log_probs = dist.log_prob(pull_masks).sum(-1, keepdim=True) # [batch_size, 1]
        old_log_probs = log_probs.clone().detach() # [batch_size, 1]
        ratio = (new_log_probs - old_log_probs).exp()

        surr1 = advantage * ratio
        surr2 = advantage * torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon)
        actor_loss = - torch.min(surr1, surr2).mean()

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
    
    def get_pull_reward_acc(self, 
                          input_ids: torch.Tensor, 
                          attention_mask: torch.Tensor,
                          pull_mask: torch.Tensor):
        """
        input_ids: [batch_size, sequence_length]
        attention_mask: [batch_size, sequence_length]
        pull_mask: [batch_size, sequence_length-1]
        """
        # We heed the pull mask to be as small as possible 
        pull_mask = pull_mask * attention_mask[:, :-1] # [batch_size, sequence_length-1]
        pull_ratio = pull_mask.sum(-1).float() / attention_mask[:, :-1].sum(-1).float() # [batch_size] 

        # Calculate the token prediction accuracy
        masked_inputs = get_mab_masked_inputs(input_ids, attention_mask, pull_mask, self.tokenizer)
        masked_input_ids = masked_inputs['input_ids']
        masked_attention_mask = masked_inputs['attention_mask']

        # correct = 1 or 0
        correct = self.is_pull_correct(masked_input_ids, masked_attention_mask, label=input_ids[:, -1]) # [batch_size]

        reward = correct / (pull_ratio + 1e-5) + 0.1 * (1-correct) * pull_ratio # [batch_size]

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
    
    cut_sequences = []
    cut_masks = []
    max_length = 0
    
    for i in range(batch_size):
        # Use attention mask to determine valid generated tokens
        gen_length = generated_masks[i].sum().item()
        if gen_length > 0:
            # Randomly choose cut point
            cut_point = torch.randint(1, gen_length+1, (1,)).item()
            # Combine input sequence with cut generated sequence
            full_seq = torch.cat([
                inputs['input_ids'][i],
                generated_portions[i][:cut_point]
            ])
            full_mask = torch.cat([
                inputs['attention_mask'][i],
                generated_masks[i][:cut_point]
            ])
        else:
            # If no generation, just use input sequence
            full_seq = inputs['input_ids'][i]
            full_mask = inputs['attention_mask'][i]
            
        cut_sequences.append(full_seq)
        cut_masks.append(full_mask)
        max_length = max(max_length, len(full_seq))
    
    # Left pad all sequences to max_length
    padded_sequences = torch.full((batch_size, max_length), 
                                tokenizer.pad_token_id,
                                dtype=inputs['input_ids'].dtype,
                                device=device)
    attention_masks = torch.zeros((batch_size, max_length),
                                dtype=inputs['attention_mask'].dtype,
                                device=device)
    
    # Fill in the sequences from the right
    for i, (seq, mask) in enumerate(zip(cut_sequences, cut_masks)):
        seq_len = len(seq)
        start_idx = max_length - seq_len
        padded_sequences[i, start_idx:] = seq
        attention_masks[i, start_idx:] = mask  # Use the original attention mask values
    
    return {
        'input_ids': padded_sequences,
        'attention_mask': attention_masks
    }

