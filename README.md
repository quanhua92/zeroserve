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
