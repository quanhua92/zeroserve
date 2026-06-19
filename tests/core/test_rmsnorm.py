import pytest
import torch

from zeroserve.core.rmsnorm import RMSNorm


@pytest.mark.skipif(not hasattr(torch.nn, "RMSNorm"), reason="requires torch>=2.4")
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
    """Tiny inputs must stay finite and numerically faithful to the float32 reference."""
    D = 8
    # Squares (~1e-40) sit near the bfloat16 subnormal floor, exercising the
    # degenerate-input path. The upcast itself is guarded numerically by
    # test_rmsnorm_correctness; here we assert finiteness + reference parity.
    x_bf16 = torch.tensor([[1e-20] * D], dtype=torch.bfloat16)

    custom_norm = RMSNorm(dim=D, eps=1e-6)
    output = custom_norm(x_bf16)

    # Reference: the same math computed entirely at float32.
    x_f32 = x_bf16.float()
    ms = x_f32.pow(2).mean(-1, keepdim=True)
    ref = (x_f32 * (ms + custom_norm.eps).rsqrt() * custom_norm.weight).to(
        torch.bfloat16
    )

    assert not torch.isnan(output).any(), "NaN values exploded inside the layer!"
    assert not torch.isinf(output).any(), "Infinity values exploded inside the layer!"
    assert torch.allclose(output, ref, atol=1e-2), (
        "Output for tiny inputs did not match the float32 reference!"
    )
