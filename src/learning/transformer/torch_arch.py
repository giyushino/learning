"""
implementation of transformer with
multihead attention
"""

import math
import torch
import torch.nn as nn
import torch.functional as F

class MultiHeadAttenion(nn.Module):
    def __init__(self, num_heads: int, emb_dim: int):
        super().__init__()
        # for multi head attention, we split
        # the embeddings across heads
        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.emb_dim = emb_dim
        self.head_dim = self.emb_dim // self.num_heads

        self.q_proj = nn.Linear(emb_dim, emb_dim)
        self.k_proj = nn.Linear(emb_dim, emb_dim)
        self.v_proj = nn.Linear(emb_dim, emb_dim)
        self.o_proj = nn.Linear(emb_dim, emb_dim)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # reshape input tensor for mha attention
        # x is (batch_size, seq_length, emb_dim)
        B, S, _ = x.shape
    
        # this is so that each head sees a slice of the embeddings
        # new shape is (batch_size, num_heads, seq_length, head_dim)
        return x.reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        # combine outputs from each head back together
        B, _, S, _ = x.shape
        # simply reverse the transformation we
        # applied when splitting the input tensor
        return x.transpose(1, 2).reshape(B, S, self.emb_dim)
        
    def scaled_self_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        attention_mask: torch.Tensor | None
    ):
        # Q matrix contains what each token is looking for
        # K matrix contains how each token should be matched
        # V matrix contains the information each token can contribute

        # Q and K are (batch_size, num_heads, seq_length, head_dim)
        # for matrix mult, we'll take the transpose of K, where we
        # can simply swap head_dim and seq_length
        # divide by sqrt of head dim to reduce variance  
        # matmul automatically broadcasts over the first two dims
        # shape is now (batch_size, num_heads, seq_length, seq_length)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask_value = torch.finfo(attn_scores.dtype).min

        S = Q.size
        


        
if __name__ == "__main__":
    mha_head = MultiHeadAttenion(10, 100) 
