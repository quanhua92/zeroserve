# ZeroServe

A journey into building a high-performance LLM serving engine from scratch.

## 🚀 The Mission
To understand the "magic" behind LLM inference by building every layer manually using **PyTorch** and **Triton**.

## 🗺️ The Phases
1. **Logic**: Implementing the math (Attention, RoPE, Norms).
2. **Speed**: Writing GPU kernels (Triton / Flash Attention).
3. **Scale**: Managing memory like an OS (Paged Attention & Batching).
4. **Polish**: Support for 4-bit weights and multiple models.

## 📈 Status
Currently in **Phase 1**: Setting up the core Transformer logic.

Check [ROADMAP.md](./ROADMAP.md) for the full checklist.

## 🛠 Getting Started

### Prerequisites
- **Python** ≥ 3.10
- **[uv](https://docs.astral.sh/uv/)** — fast Python package manager

### Install
```bash
git clone git@github.com:quanhua92/zeroserve.git zeroserve && cd zeroserve
uv sync                      # resolves + installs the correct PyTorch wheels for your platform
```

`uv sync` automatically picks the right PyTorch build:
| Platform | Wheels |
|---|---|
| Linux x86_64 / Windows | CUDA 12.4 (`download.pytorch.org/whl/cu124`) |
| macOS (Intel & Apple Silicon) | PyPI (native MPS support) |
| Linux aarch64 / other | CPU-only (`download.pytorch.org/whl/cpu`) |

### Configure Git Hooks (one-time, per clone)
A `pre-commit` hook in [`githooks/`](./githooks) runs `ruff check --fix` and `ruff format` on staged Python files before each commit. Point Git at it once after cloning:

```bash
git config core.hooksPath githooks
```

To skip the hook for a single commit: `git commit --no-verify`.

## 📂 Project Structure

```text
zeroserve/
├── pyproject.toml         # Modern build system config
├── src/
│   └── zeroserve/
│       ├── core/          # Foundational math (RMSNorm, RoPE, MLP)
│       ├── models/        # Model architectures & weight loaders
│       ├── kernels/       # Triton custom kernels (FlashAttention)
│       ├── engine/        # Serving infrastructure (Scheduler, PagedMemory)
│       └── utils/         # Helper functions
├── tests/                 # Unit tests & verification
└── scripts/               # Benchmarks & utility scripts
```

---
*Learning journey by quanhua92*
