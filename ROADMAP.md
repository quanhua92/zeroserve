# The ZeroServe Journey: Roadmap & Mastery

This document tracks my evolution from understanding basic tensor math to architecting a production-grade heterogeneous serving engine. The learning path prioritizes mastering local implementations first via **tiny-llm** (MLX) and **nano-vllm** (PyTorch), culminating in the **llmsystem2026** end goals (Disaggregation, MoE, Trillion-Parameter Scale).

## 📍 Current Progress
**Phase 1: Foundations** - [▓░░░░░░░░░] 10%

---

## 🏗️ Phase 1: Foundations (Logic & Math Pipe)
*Mastering local dense transformer math via `tiny-llm` (Week 1) and `nanoGPT`.*

- [x] Setup `zeroserve/src/zeroserve/` directory structure.
- [ ] **Tokenization**: BPE vocabulary merging, pre-tokenization regex, and NFD normalization.
- [ ] **RMSNorm**: Implement standard and upcast normalization logic.
- [ ] **RoPE**: Manual implementation of Rotary Positional Embeddings (traditional and complex).
- [ ] **SwiGLU**: Build the gated MLP block.
- [ ] **GQA**: Grouped Query Attention broadcasting logic.
- [ ] **Transformer Block**: Combine Norms, Attention, and MLP into a cohesive layer.
- [ ] **Generation Loop**: Basic autoregressive decode with Top-p/Top-k/Temperature sampling.

---

## ⚡ Phase 2: Acceleration (Kernels & Quantization)
*Moving from Python math to hardware kernels via `tiny-llm` (Week 2).*

- [ ] **Quantization**: W4A16 asymmetric group-wise quantization (scales, biases, packing).
- [ ] **Custom CPU Kernels**: Write `quantized_matmul` using C++ loop unrolling.
- [ ] **Custom GPU Kernels**: Write `quantized_matmul` in Metal/Triton.
- [ ] **Dense KV-Cache**: Implement an offset-based pre-allocated dense cache.
- [ ] **FlashAttention**: Implement tiled online-softmax attention kernels to bypass the HBM memory wall.

---

## 🌊 Phase 3: Core Engine & Serving
*Building a vLLM-like continuous batching engine via `tiny-llm` (Week 3) and `nano-vllm`.*


- [ ] **Paged KV-Cache**: Allocate fixed-size physical memory pages in a global pool.
- [ ] **Block Manager**: Logical-to-physical block table mapping.
- [ ] **Paged Attention Kernel**: Update FlashAttention to walk non-contiguous page blocks.
- [ ] **The Scheduler**: Continuous batching loop (WAITING, RUNNING, FINISHED).
- [ ] **Chunked Prefill**: Interleave prefill chunks with decode steps to prevent latency spikes.
- [ ] **Radix Attention (Prefix Caching)**: Reuse KV cache across requests sharing identical prompts.
- [ ] **PyTorch Dynamo Compilation**: Compile eager modules with `torch.compile` using custom guards and graph verification.
- [ ] **PEFT & Multi-Adapter Serving**: Fused dynamic adapter weight loading and batch Grouped GEMM execution (Punica/S-LoRA style).
- [ ] **LMCache**: Multi-host global cache lookup and hierarchical memory offloading (GPU ➔ CPU ➔ SSD ➔ S3).
- [ ] **CUDA Graphs**: Capture static decode execution graphs to eliminate Python overhead.

---

## 🌍 Phase 4: Distributed Training & Scale
*Scaling to datacenters via `llmsystem2026`.*

- [ ] **DDP & NCCL**: Multi-GPU gradient synchronization.
- [ ] **ZeRO Optimization**: Partition Optimizer states, Gradients, and Parameters to break the memory limits.
- [ ] **Tensor Parallelism (TP)**: Shard Linear layers across GPUs via All-Reduce.
- [ ] **Pipeline Parallelism (PP)**: Micro-batch scheduling (1F1B) across nodes.

---

## 🚀 Phase 5: The End Goal (Next-Gen Serving)
*SOTA multi-trillion parameter architecture via `llmsystem2026`.*

- [ ] **Mixture of Experts (MoE)**: Router logic, expert-parallelism, and grouped GEMM kernels.
- [ ] **Speculative Decoding**: Draft model token generation and target model parallel verification.
- [ ] **Disaggregated Prefill & Decode**: Separate prefill (Compute-bound) and decode (Memory-bound) onto isolated GPU clusters (DistServe/Mooncake).
- [ ] **JAX/TPU Compiler**: Develop custom TPU exact attention kernels in JAX Pallas (Splash Attention) compiled using XLA.
- [ ] **Heterogeneous Hardware (KTransformers)**: Offload MoE expert weights to cheap CPU DRAM (AMX instructions) while retaining attention on GPU.
- [ ] **ZeroServe CLI**: A clean terminal interface wrapping the ultimate inference engine.

