"""
vlm
"""

import torch
import torch.nn as nn


from learning.models.vit import TorchVisionTransformer
from learning.models.llm import CausalTransformerLM

class VisionLanguageModel(nn.Module):
    """
    llava style VLM        
    """
    def __init__(self):
        super().__init__()
        self.vit = TorchVisionTransformer()
        self.llm = CausalTransformerLM()
