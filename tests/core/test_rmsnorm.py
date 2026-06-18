import pytest
import torch

from zeroserve.core.rmsnorm import RMSNorm


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rmsnorm_correctness(dtype):
    """Verify our RMSNorm matches torch.nn.RMSNorm as an independent reference."""
    torch.manual_seed(42)

    B, L, D = 2, 4, 64
    eps = 1e-6

    x = torch.randn(B, L, D).to(dtype)

    custom_norm = RMSNorm(dim=D, eps=eps)

    # Independent reference: PyTorch's built-in RMSNorm with synced weights.
    ref_norm = torch.nn.RMSNorm(D, eps=eps).to(dtype)
    ref_norm.weight.data.copy_(custom_norm.weight.data.to(dtype))

    output_custom = custom_norm(x)
    output_ref = ref_norm(x)

    # Tighter tolerance for float32; looser for bfloat16 quantization noise.
    atol = 1e-6 if dtype == torch.float32 else 1e-3

    assert output_custom.shape == x.shape, "Output tensor shape mismatched!"
    assert output_custom.dtype == dtype, f"Output dtype was not preserved as {dtype}!"
    assert torch.allclose(output_custom, output_ref, atol=atol), (
        f"Numerical parity vs torch.nn.RMSNorm failed for {dtype}!"
    )


def test_rmsnorm_underflow_safety():
    """Ensure small micro-values do not cause NaN explosions due to underflow."""
    D = 8

    # Tiny numbers whose squares approach bf16 subnormal boundary (~1.2e-38),
    # so squaring without the float32 upcast would underflow to zero.
    x = torch.tensor([[1e-20] * D], dtype=torch.bfloat16)

    custom_norm = RMSNorm(dim=D, eps=1e-6)
    output = custom_norm(x)

    assert not torch.isnan(output).any(), "NaN values exploded inside the layer!"
    assert not torch.isinf(output).any(), "Infinity values exploded inside the layer!"
