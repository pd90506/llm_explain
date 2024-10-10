    def token_wise_prediction_head(self, hidden_states, causal_mask=None):
        """ 
        hidden_states: torch.Tensor of shape [N, L, hidden_size]
        causal_mask: torch.Tensor of shape [N, L, L]
        """
        if causal_mask is None:
            causal_mask = get_causal_mask(hidden_states) # [N, L, L]

        # hidden_states = self.policy_map(hidden_states) # [N, L, hidden_size]
    

        N, L, hidden_size = hidden_states.size()
        # element_wise_hidden_states = hidden_states.unsqueeze(2) * hidden_states.unsqueeze(1) # [N, L, L, hidden_size]
        # # print("element_wise_hidden_states", element_wise_hidden_states.shape)
        # element_wise_policy_logits = self.policy_map(element_wise_hidden_states).squeeze(-1) # [N, L, L]
        # element_wise_policy_logits = element_wise_policy_logits * causal_mask # [N, L, L]
        hidden_states = self.policy_map(hidden_states) # [N, L, hidden_size]
        hidden_states = F.normalize(hidden_states, p=2, dim=-1) # [N, L, hidden_size]
        element_wise_policy_logits = torch.matmul(hidden_states, hidden_states.transpose(1, 2)) # [N, L, L]
        element_wise_policy_logits = element_wise_policy_logits * causal_mask # [N, L, L]
        element_wise_policy_logits = element_wise_policy_logits * self.logit_scale.exp() # [N, L, L]

        
        return element_wise_policy_logits # [N, L, L]
    
    def calculate_contrast_loss(self, hidden_states, attention_mask, context_mask, response_mask):
        """ 
        hidden_states: torch.Tensor of shape [N, L, hidden_size]
        attention_mask: torch.Tensor of shape [N, L]
        response_mask: torch.Tensor of shape [N, L]
        """
        N, L, hidden_size = hidden_states.size()
        causal_mask = get_causal_mask(hidden_states, attention_mask)
        hidden_states = self.policy_map(hidden_states) # [N, L, hidden_size]
        hidden_states = F.normalize(hidden_states, p=2, dim=-1) # [N, L, hidden_size]
        # hidden_states = hidden_states * self.logit_scale.exp() # [N, L, hidden_size]

        # response_mask = response_mask.unsqueeze(-1) # [N, L, 1]
        # mean_state = (hidden_states * response_mask).sum(1) / response_mask.sum(1) # [N, hidden_size]

        # hidden_states_1 = hidden_states.unsqueeze(2) * context_mask.unsqueeze(-1).unsqueeze(-1)  # [N, L, 1, hidden_size]
        # hidden_states_2 = hidden_states.unsqueeze(1) * response_mask.unsqueeze(1).unsqueeze(-1) # [N, 1, L, hidden_size]

        # cross_inner_product = (hidden_states_1 * hidden_states_2).sum(1) / context_mask.unsqueeze(-1).unsqueeze(-1).sum(1) 
        # cross_inner_product = cross_inner_product.sum(1) / response_mask.unsqueeze(-1).sum(1) # [N, hidden_size]

        # cross_inner_product = cross_inner_product * self.logit_scale.exp() # [N, hidden_size]
        # cross_inner_product = (cross_inner_product * response_mask.unsqueeze(1)).sum(-1) / (response_mask.unsqueeze(1).sum(-1) + 1) # [N, L]
        hidden_state_context = hidden_states * context_mask.unsqueeze(-1)
        flat_hidden_state_context = hidden_state_context.view(N * L, hidden_size)
        hidden_state_response = hidden_states * response_mask.unsqueeze(-1)
        flat_hidden_state_response = hidden_state_response.view(N * L, hidden_size)

        cross_inner_product = torch.matmul(flat_hidden_state_context, flat_hidden_state_response.transpose(0, 1)) # [N*L, N*L]
        cross_inner_product = cross_inner_product.view(N, L, N, L) # [N, L, N, L]
        cross_inner_product = (cross_inner_product * context_mask.unsqueeze(-1).unsqueeze(-1)).sum(1) / (context_mask.unsqueeze(-1).unsqueeze(-1).sum(1) + 1) # [N, N, L]
        cross_inner_product = (cross_inner_product * response_mask.unsqueeze(0)).sum(-1) / (response_mask.unsqueeze(0).sum(-1) + 1) # [N, N]
        

        # Calculate similarity matrix
        # similarity_matrix = torch.matmul(cross_inner_product, cross_inner_product.transpose(0, 1)) # [N, N]
        similarity_matrix = cross_inner_product

        # Apply temperature scaling
        temperature = 1
        similarity_matrix = similarity_matrix / temperature

        # Create labels for contrastive learning
        labels = torch.arange(similarity_matrix.size(0)).to(similarity_matrix.device)

        # Compute contrastive loss using cross-entropy
        contrast_loss = F.cross_entropy(similarity_matrix, labels)

        return contrast_loss



def pairwise_token_attention(h, context_mask, response_mask):
    """
    h: torch.Tensor, shape (batch_size, seq_len, hidden_size)
    context_mask: torch.Tensor, shape (batch_size, seq_len)
    response_mask: torch.Tensor, shape (batch_size, seq_len)
    """
    N, L, d = h.shape

    # get h1 and h2 for pairwise attention
    h1_expanded = (h ).unsqueeze(1).expand(N, N, L, d)
    h2_expanded = (h ).unsqueeze(0).expand(N, N, L, d)
    torch.set_printoptions(threshold=torch.inf)
    # print("h1_expanded", h1_expanded[:2, :2, :, 0])

    # calculate pairwise attention scores，(N, N, L, L)
    scores = torch.matmul(h1_expanded, h2_expanded.transpose(-1, -2)) / torch.sqrt(torch.tensor(d, dtype=torch.float32)) # (N, N, L, L)
    # print("scores", scores[0, 0, 0, :])
    mask = response_mask.unsqueeze(0).unsqueeze(2).expand(N, N, L, L) # (N, N, L, L)
    # print("mask", mask.sum([-1, -2]))
    # mask = response_mask.unsqueeze(0).expand(N, N, L, L) # (N, N, L, L)
    scores = scores.masked_fill(~mask.bool(), -1e9) # mask out the padding tokens

    weights = F.softmax(scores, dim=-1) # (N, N, L, L)
    # print("mask", mask[0, 0, 0, :])
    # print("weights", weights[0, 0, 0, :])
    # print("h2_expanded", h2_expanded[0, 0, :, 0])

    # calculate the output of the pairwise attention
    output = torch.matmul(weights, h2_expanded) # (N, N, L, d)
    # output = F.normalize(output, p=2, dim=-1) # (N, N, L, d)
    # print("output", output[0, 0, :, 0])

    return output