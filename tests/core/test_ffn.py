import pytest
import torch
import torch.nn.functional as F

from zeroserve.core.ffn import FFN


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_ffn_forward(dtype):
    batch_size = 4
    seq_len = 10
    dim = 16
    hidden_dim = 32

    ffn = FFN(dim, hidden_dim).to(dtype)
    input_tensor = torch.randn(batch_size, seq_len, dim, dtype=dtype)

    # Reference
    ref = F.gelu(F.linear(input_tensor, ffn.w1.weight, ffn.w1.bias))
    expected = F.linear(ref, ffn.w2.weight, ffn.w2.bias)

    output = ffn(input_tensor)

    assert output.shape == (batch_size, seq_len, dim), "Output shape mismatch"
    assert output.dtype == dtype, "Output dtype mismatch"

    # Check correctness against the reference implementation
    atol = 1e-6 if dtype == torch.float32 else 1e-3
    assert torch.allclose(output, expected, atol=atol)


def test_ffn_first_projection_used():
    ffn = FFN(16, 32)
    ffn.w1.weight.data.zero_()
    ffn.w1.bias.data.zero_()

    output = ffn(torch.randn(2, 1, 16))
    expected = ffn.w2.bias.broadcast_to(output.shape)

    assert torch.allclose(output, expected), (
        "First projection should be zero, output should equal second bias"
    )


def test_ffn_second_projection_used():
    ffn = FFN(16, 32)
    ffn.w2.weight.data.zero_()
    ffn.w2.bias.data.zero_()

    output = ffn(torch.randn(2, 1, 16))
    expected = torch.zeros_like(output)

    assert torch.allclose(output, expected), (
        "Second projection zeroed, output should be all zeros"
    )
