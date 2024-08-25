from torch import nn
import torch
from typing import Callable
from torch.nn import LayerNorm

class CLIPMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.activation_fn = nn.GELU()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class AttentionalLayer(nn.Module):
    def __init__(self, d_model: int,
                        context_dim: int,
                        n_head: int = 8,
                        intermediate_size:int = 3072,
                        norm_layer: Callable = LayerNorm) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, kdim=context_dim, vdim=context_dim)
        self.layer_norm1 = norm_layer(d_model)
        self.mlp = CLIPMLP(d_model, intermediate_size)
        self.layer_norm2 = norm_layer(d_model)
        self.out_norm = norm_layer(d_model)

    def forward(self, hidden_states: torch.Tensor):
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.attn(
            hidden_states.permute(1, 0, 2),
            hidden_states.permute(1, 0, 2),
            hidden_states.permute(1, 0, 2),
            need_weights=False
        )[0].permute(1, 0, 2)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.out_norm(hidden_states)
        return hidden_states


class AttentionalPooler(nn.Module):
    def __init__(
            self,
            d_model: int,
            context_dim: int,
            n_head: int = 8,
            n_queries: int = 256,
            norm_layer: Callable = LayerNorm
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_head, kdim=context_dim, vdim=context_dim)
        self.ln_q = norm_layer(d_model)
        self.ln_k = norm_layer(context_dim)

        # self.out_attention = AttentionalLayer(d_model, d_model, n_head, norm_layer = norm_layer)

    def forward(self, x: torch.Tensor):
        x = self.ln_k(x).permute(1, 0, 2)  # NLD -> LND
        N = x.shape[1]
        q = self.ln_q(self.query)
        out = self.attn(q.unsqueeze(1).expand(-1, N, -1), x, x, need_weights=False)[0]

        # out = self.out_attention(out.permute(1, 0, 2))  # LND -> NLD
        out = out.permute(1, 0, 2)  # LND -> NLD
        return out