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

        self.loss_fn = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def get_second_last_token_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Get the logits for the last token.
        """
        logits_all = self.target_model(input_ids, attention_mask).logits
        logits = logits_all[:, -2, :] # [batch_size, vocab_size] the second last output is the logits for predicting the last token
        return logits

    def get_pull_profit(self, 
                          input_ids: torch.Tensor, 
                          attention_mask: torch.Tensor,
                          pull_mask: torch.Tensor,
                          cost_value: torch.Tensor):
        """
        input_ids: [batch_size, sequence_length]
        attention_mask: [batch_size, sequence_length]
        pull_mask: [batch_size, sequence_length-1]
        cost_value: [batch_size]
        """
        label = input_ids[:, -1] # [batch_size] the last token is the label
        # Get the masked input_ids and attention_mask
        masked_inputs = get_mab_masked_inputs(input_ids, attention_mask, pull_mask, self.tokenizer)
        masked_input_ids = masked_inputs['input_ids']
        masked_attention_mask = masked_inputs['attention_mask']

        # Get the prediction of the target model
        logits = self.get_second_last_token_logits(masked_input_ids, masked_attention_mask) # [batch_size, vocab_size]

        masked_pred_prob = torch.softmax(logits, -1) # [batch_size, vocab_size]
        masked_true_prob = masked_pred_prob.gather(1, label.unsqueeze(-1)).squeeze(-1) # [batch_size]

        # Get the reward
        pull_mask = pull_mask * attention_mask[:, :-1] # [batch_size, sequence_length-1]
        mask_size = pull_mask.sum(-1).float() # [batch_size]
        profit = masked_true_prob - cost_value * mask_size # [batch_size] 

        return profit
    
    def _get_cost_value(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        input_ids: [batch_size, sequence_length]
        attention_mask: [batch_size, sequence_length]
        """
        logits = self.get_second_last_token_logits(input_ids, attention_mask) # [batch_size, vocab_size]
        probs = torch.softmax(logits, -1) # [batch_size, vocab_size]
        label = input_ids[:, -1] # [batch_size] the last token is the label
        cost_value = probs.gather(1, label.unsqueeze(-1)).squeeze(-1) # [batch_size]
        # ignore the last token
        attention_mask = attention_mask[:, :-1] # [batch_size, sequence_length-1]
        cost_value = cost_value / attention_mask.sum(-1).float() # [batch_size]
        return cost_value
    
    @torch.no_grad()
    def collect_pulls(self, 
                      input_ids: torch.Tensor, 
                      attention_mask: torch.Tensor,
                      dist: torch.distributions.Distribution,
                      num_pulls: int = 3,
                      ) -> dict:
        """Collect pull results for MAB training.
        
        """
        # Initialize lists for PPO
        log_probs, pull_masks, profits = [], [], []
        # Get the cost value from the target model
        cost_value = self._get_cost_value(input_ids, attention_mask) # [batch_size]
                            
        # Collect pulls
        for _ in range(num_pulls):
            pull_mask = dist.sample() # [batch_size, sequence_length-1]
            profit = self.get_pull_profit(input_ids, attention_mask, pull_mask, cost_value).unsqueeze(-1) # [batch_size, 1]
            log_prob = dist.log_prob(pull_mask).sum(-1, keepdim=True) # [batch_size, 1]
            
            log_probs.append(log_prob.clone())
            profits.append(profit.clone())
            pull_masks.append(pull_mask.clone())
            
        return {
            'log_probs_list': log_probs,
            'pull_masks_list': pull_masks,
            'profits_list': profits,
        }
    
    @torch.no_grad()
    def get_mab_empirical_profit(self, mab_values: torch.Tensor, profit_value: torch.Tensor, pull_masks_list: list[torch.Tensor], profits_list: list[torch.Tensor], gamma=0.9):
        """ 
        mab_values: [batch_size, sequence_length-1]
        profit_value: [batch_size, 1]
        pull_masks_list: list of [batch_size, sequence_length-1]
        profits_list: list of [batch_size, 1]
        """
        pull_masks = torch.stack(pull_masks_list, dim=1) # [batch_size, num_pulls, sequence_length-1]
        profits = torch.stack(profits_list, dim=1) # [batch_size, num_pulls, 1]

        profit_advantage = (profits - profit_value.unsqueeze(1)) # [batch_size, num_pulls, 1]

        nominator = (pull_masks * profit_advantage).sum(dim=1) # [batch_size, sequence_length-1]
        # replace the 0s with values in mab_values
        # nominator = torch.where(nominator == 0, mab_values, nominator)

        denominator = pull_masks.sum(dim=1) # [batch_size, sequence_length-1]
        # replace the 0s with 1s
        denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)

        mab_value_estimation = nominator / denominator # [batch_size, sequence_length-1]

        running_mab_value = mab_values * gamma + mab_value_estimation * (1 - gamma)  # [batch_size, sequence_length-1]
        # running_mab_value = mab_value_estimation + mab_values
        # running_mab_value = mab_value_estimation
        profit_value_estimation = profits.mean(dim=1) # [batch_size, 1]

        return running_mab_value, profit_value_estimation, profit_advantage # [batch_size, sequence_length-1], [batch_size, 1]



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

            dist, mab_values, profit_value = self.mab_model.get_dist_value(input_ids, attention_mask)

            pulls = self.collect_pulls(input_ids, attention_mask, dist, num_pulls)
            pull_masks_list = pulls['pull_masks_list']
            profits_list = pulls['profits_list']
            profit_value = profit_value.unsqueeze(-1) # [batch_size, 1]
            running_mab_value, profit_value_estimation, profit_advantage = self.get_mab_empirical_profit(mab_values=mab_values.clone().detach(), 
                                                                  profit_value=profit_value.clone().detach(),
                                                                  pull_masks_list=pull_masks_list, 
                                                                  profits_list=profits_list, 
                                                                  gamma=gamma) # [batch_size, sequence_length-1], [batch_size, 1]

            mab_value_loss = self.loss_fn(mab_values, running_mab_value.clone().detach())
            # mab_value_loss = self.bce_loss(mab_values, torch.sigmoid(running_mab_value.clone().detach()))
            profit_value_loss = self.loss_fn(profit_value, profit_value_estimation.clone().detach())
            entropy = dist.entropy().mean()
            
            loss = mab_value_loss + 0.5 * profit_value_loss - 0.001 *entropy

            # Update the MAB model
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # log 
            profits = torch.stack(profits_list, dim=1) # [batch_size, num_pulls, 1]
            pull_masks = torch.stack(pull_masks_list, dim=1) # [batch_size, num_pulls, sequence_length-1]
            mask_ratio = pull_masks.mean().item()
            
            wandb.log({
                'mab_value_loss': mab_value_loss.item(),
                'entropy': entropy.item(),
                'loss': loss.item(),
                "profit_value_loss": profit_value_loss.item(),
                'running_mab_value': running_mab_value.mean().item(),
                'mab_values': mab_values.mean().item(),
                'profits': profits.mean().item(),
                'profit_advantage': profit_advantage.mean().item(),
                'mask_ratio': mask_ratio,
                'scale': self.mab_model.log_scale.exp().item(),
                'profit_value': profit_value.mean().item(),
            })

            # save the model
            if batch_idx % 100 == 0:
                torch.save(self.mab_model.state_dict(), f"checkpoints/mab_model_{batch_idx}.pth")



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

