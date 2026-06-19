import pytest
import torch
import torch.nn.functional as F

from zeroserve.core.swiglu import SwiGLU


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_swiglu_forward(dtype):
    batch_size = 4
    seq_len = 10
    dim = 16
    hidden_dim = 32

    swiglu = SwiGLU(dim, hidden_dim).to(dtype)
    input_tensor = torch.randn(batch_size, seq_len, dim, dtype=dtype)

    # Reference
    gate = F.silu(F.linear(input_tensor, swiglu.w1.weight))
    up = F.linear(input_tensor, swiglu.w2.weight)
    mixed = gate * up
    expected = F.linear(mixed, swiglu.w3.weight)

    output = swiglu(input_tensor)

    assert output.shape == (batch_size, seq_len, dim), "Output shape mismatch"
    assert output.dtype == dtype, "Output dtype mismatch"

    # Check correctness against the reference implementation
    atol = 1e-6 if dtype == torch.float32 else 1e-3
    assert torch.allclose(output, expected, atol=atol)


def test_swiglu_gate_path_used():
    swiglu = SwiGLU(16, 32)
    swiglu.w1.weight.data.zero_()

    output = swiglu(torch.randn(2, 1, 16))
    expected = torch.zeros_like(output)

    assert torch.allclose(output, expected), (
        "Gate projection zeroed, output should be all zeros"
    )


def test_swiglu_up_path_used():
    swiglu = SwiGLU(16, 32)
    swiglu.w2.weight.data.zero_()

    output = swiglu(torch.randn(2, 1, 16))
    expected = torch.zeros_like(output)

    assert torch.allclose(output, expected), (
        "Up projection zeroed, output should be all zeros"
    )


def test_swiglu_down_path_used():
    swiglu = SwiGLU(16, 32)
    swiglu.w3.weight.data.zero_()

    output = swiglu(torch.randn(2, 1, 16))
    expected = torch.zeros_like(output)

    assert torch.allclose(output, expected), (
        "Down projection zeroed, output should be all zeros"
    )
