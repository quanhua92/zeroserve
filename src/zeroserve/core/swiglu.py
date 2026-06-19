import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # gate projection
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # up projection
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # down projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
