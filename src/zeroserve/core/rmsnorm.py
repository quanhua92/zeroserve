import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype

        # upcast to float32 for better numerical stability
        x = x.float()

        mean_square = x.pow(2).mean(-1, keepdim=True)
        x = x * (mean_square + self.eps).rsqrt()

        # downcast back to original dtype and apply weight
        x = x.to(original_dtype) * self.weight.to(original_dtype)

        return x
