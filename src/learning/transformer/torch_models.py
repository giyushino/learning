"""
normal transformers and
vision models
"""

import torch
import torch.nn as nn

from learning.transformer.torch_arch import TransformerBlock


class CausalTransformerLM(nn.Module):
    """
    Decoder-only causal language model.
    """ 
    def __init__(self, num_layers: int, num_heads: int, emb_dim: int, ffn_mult: int, vocab_size: int):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, emb_dim)

        self.blocks = nn.ModuleList(
            TransformerBlock(num_heads, emb_dim, ffn_mult)
            for _ in range(num_layers)
        )

        self.norm = nn.LayerNorm(emb_dim)
        self.output = nn.Linear(emb_dim, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_hidden_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        B, S = x.shape

        # Keep left-padded batches compatible with RoPE by numbering only the
        # non-pad tokens.
        if attention_mask is None:
            position_ids = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)
        else:
            attention_mask = attention_mask.bool()
            position_ids = attention_mask.long().cumsum(dim=1) - 1
            position_ids = position_ids.masked_fill(~attention_mask, 0)

        x = self.token_emb(x)
        hidden_states: list[torch.Tensor] = [x] if return_hidden_states else []

        for block in self.blocks:
            x = block(x, attention_mask, position_ids)
            if return_hidden_states:
                hidden_states.append(x)
        
        # (batch_size, seq_lenth, vocab_size)
        logits = self.output(self.norm(x))
        if return_hidden_states:
            return logits, hidden_states

        return logits


class VisionTransfomer(nn.Module):
    # todo
    pass


if __name__ == "__main__":
    addition_config = {
        "num_layers": 10,
        "num_heads": 4,
        "emb_dim": 728,
        "ffn_mult": 4,
        "vocab_size": 14,
    }

    vocab = {
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
        "+": 10,
        "=": 11,
        "<eos>": 12,
        "<pad>": 13,
    }

    tokenizer_config = {
        "vocab": vocab,
        "eos_id": 12,
        "padding_id": 13
    }

    model = CausalTransformerLM(**addition_config)
    rand_tensor = torch.randint(0, 13, (2, 10)) 
    output = model(rand_tensor)
    print(output)
 
