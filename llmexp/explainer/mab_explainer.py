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


class MABExplainer:
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
