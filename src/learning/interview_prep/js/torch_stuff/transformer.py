import math

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, n_heads*self.head_dim)
        self.k_proj = nn.Linear(d_model, n_kv_heads*self.head_dim)
        self.v_proj = nn.Linear(d_model, n_kv_heads*self.head_dim)
        self.o_proj = nn.Linear(n_heads*self.head_dim, d_model)

    def scaledDotProductAttention(
            self, 
            q: torch.Tensor, 
            k: torch.Tensor, 
            v: torch.Tensor,
            is_causal: bool,
            attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # we should have already split the heads for q, k, v
        # we scale down attn_scores by the square root of the head dim
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.size(-1))
        mask_value = torch.finfo(attn_scores.dtype).min

        if is_causal:
            S = q.size(2)
            causal_mask = torch.triu(
                torch.ones(S, S, device=q.device, dtype=torch.bool),
                diagonal=1
            )
            attn_scores = attn_scores.masked_fill(causal_mask, mask_value)

        if attn_mask is not None:
            attn_mask_values = ~attn_mask.bool()
            attn_scores = attn_scores.masked_fill(attn_mask_values, mask_value)


        attn_probs = torch.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_probs, v)

    def split_q_heads(self, x):
        B, S, V = x.shape
        return x.reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)

    def split_kv_heads(self,x):
        B, S, V = x.shape
        return x.reshape(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

    def combine_heads(self, x):
        B, H, S, D = x.shape
        return x.transpose(1, 2).reshape(B, S, H * D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.split_q_heads(self.q_proj(x))
        k = self.split_kv_heads(self.k_proj(x))
        v = self.split_kv_heads(self.v_proj(x))

        k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
        v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)

        attn_probs = self.scaledDotProductAttention(q, k, v, True, None)
        
        return self.o_proj(self.combine_heads(attn_probs))
    



if __name__ == "__main__":
    block = TransformerBlock(512, 16, 8)
    test_tensor = torch.rand((4, 128, 512))
    output = block(test_tensor)
    print(output)
    
