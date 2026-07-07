# T2-1-2 服务能力优化 — 思路与方案

## 一、任务分析

### 1.1 目标

> 多并发、长文本服务场景中，输出吞吐和总吞吐量提升；小规模性能无明显下降，有提升可加分。

核心是**高并发 + 长序列场景下的吞吐优化**，同时保证小规模不退化。

### 1.2 测试矩阵

| 模型 | 并发数 | 输入长度 (tokens) | 输出长度 (tokens) |
|------|--------|-------------------|-------------------|
| 8B | 1, 4 | 32, 256, 4096 | 256, 1024, 4096 |
| 8B | 16, 64 | 32, 256 | 256, 1024, 4096 |
| 70B | 1, 4, 16 | 32, 256 | 256, 1024, 2048 |

### 1.3 瓶颈分析

| 场景 | 主要瓶颈 | 优化方向 |
|------|----------|----------|
| 低并发 + 短序列 | Kernel launch 延迟 | CUDA Graph |
| 高并发 + 短序列 | Batch 计算吞吐 | 连续批处理调度、GEMM 利用率 |
| 低并发 + 长序列 | Attention 计算 + 显存 | Flash Attention、KV Cache 显存管理 |
| 高并发 + 长序列 | KV Cache 显存容量 | PagedAttention 调优、Chunked Prefill |

### 1.4 InfiniLM 架构关键发现

通过深入代码分析，以下事实影响优化方案选择：

- **调度器在 Python 层**（`python/infinilm/llm/scheduler.py`），不是 C++。维护 waiting queue（prefill）和 running queue（decode），用 `max_batch_size` 和 `INFINILM_MAX_NUM_BATCHED_TOKENS` 控制每步 token 预算。
- **前缀缓存（Prefix Caching）已内置**（`cache_manager.py`）：基于 xxHash64 的 block 哈希匹配 + 引用计数写时复制，分页和静态缓存模式都自动启用。
- **INT8 KV Cache 仅支持 Static Attention 后端**（`static_attn.cpp`），**不支持 Paged Attention 和 Flash Attention**。这是最重要的限制。
- **权重量化（AWQ/GPTQ/Compressed-Tensors）是自动检测的**，从模型 `config.json` 中的 `quantization_config` 读取，不需要手动传参。
- **CUDA Graph 预编译**了 batch size 1-512 范围内的多个图（`[1..63], 64, 80, 96, 112, 128, 160, ..., 448, 512`），decode 时自动匹配。

---

## 二、推荐模型

Llama 架构在 InfiniLM 中支持最成熟，flash-attn / paged-attn / CUDA Graph 等优化路径都已在 Llama 上充分验证：

| 规格 | 模型 | 下载命令 | 显存 (FP16) |
|------|------|----------|-------------|
| 8B | `LLM-Research/Meta-Llama-3.1-8B-Instruct` | `modelscope download --model LLM-Research/Meta-Llama-3.1-8B-Instruct --local_dir /models/Llama-3.1-8B-Instruct` | ~16 GB |
| 70B | `LLM-Research/Meta-Llama-3.1-70B-Instruct` | `modelscope download --model LLM-Research/Meta-Llama-3.1-70B-Instruct --local_dir /models/Llama-3.1-70B-Instruct` | ~140 GB |

备选：`Qwen/Qwen2.5-7B-Instruct` / `Qwen/Qwen2.5-72B-Instruct`

---

## 三、配置层优化方案

### 3.1 参考 Baseline

```bash
python python/infinilm/server/inference_server.py \
  --device nvidia \
  --model=/models/Llama-3.1-8B-Instruct \
  --temperature 1.0 --top-p 0.8 --top-k 1 \
  --port 8102 --tp 1 \
  --max-new-tokens 4096 \
  --num-blocks 6144 \
  --max-batch-size <max_con> \
  --enable-graph \
  --enable-paged-attn \
  --attn flash-attn \
  --ignore-eos
```

### 3.2 方案 A：Paged Attention + Flash Attention（推荐主线）

这是参考命令的路线。适用高并发 + 长序列场景。

#### 3.2.1 Flash Attention（`--attn flash-attn`）

长文本（4096 tokens）的刚需。Flash Attention 将 attention 显存从 O(n²) 降为 O(n)。

- 使用 `mha_varlen`（prefill）和 `mha_kvcache`（decode）操作
- KV Cache 布局为 BHSD（`[2, num_blocks, block_size, num_heads, dim]`），专为 FlashAttention 优化
- **局限：不支持 INT8 KV Cache**（FlashAttn 后端没有接入 KV 量化代码路径）

> ⚠️ 需要 Flash Attention 子模块：
> ```bash
> cd /root/difanrui/InfiniCore
> git submodule update --init third_party/flash-attention
> python3 scripts/install.py --nv-gpu=y --cuda=/usr/local/cuda --flash-attn=<path>
> ```

#### 3.2.2 CUDA Graph（`--enable-graph`）

Decode 阶段关键优化。预编译 batch size 1-512 的图，消除 kernel launch 开销。

- 低并发（1, 4）收益最大（decode 延迟降 20-40%）
- 高并发下 batch 变大后 compute bound 比例上升，但仍有收益
- 首次启动时有额外编译时间

#### 3.2.3 KV Cache 块数精确计算

Paged Attention 下 KV Cache 容量由 `num-blocks × block-size` 决定。

```
KV Cache 显存 ≈ num_blocks × block_size × num_layers × 2 × num_kv_heads × head_dim × 2
```

以 Llama-3.1-8B（32 layers, 8 kv_heads, 128 head_dim）为例：
- 每 token FP16 KV ≈ 128 KB
- 6144 blocks × 256 tokens/block = 1,572,864 tokens 总容量
- 总 KV 显存 ≈ 6144 × 256 × 128KB ≈ **192 GB** → 但这是理论最大值，实际分到各层各 rank

实际上每层的 KV Cache 是 `[2, num_blocks, num_kv_heads, block_size, head_dim]`：
- 每层：2 × 6144 × 8 × 256 × 128 × 2 bytes = 2 × 6144 × 8 × 256 × 128 × 2 ≈ **6 GB**
- 32 层总计 ≈ 192 GB → **远超单卡 24GB！** 这说明参考命令里的 6144 是给多卡或更大显存的。
- 对于 24GB RTX 4090 单卡运行 8B：权重 16GB + KV 约 6-8GB → **num-blocks 应设在 2048-4096 之间**

**调优策略：**

| 场景 | block_size | num_blocks（单卡 RTX 4090） |
|------|-----------|---------------------------|
| 8B, 并发 4, 长序列 (4096) | 256 | 2048-3072 |
| 8B, 并发 64, 短序列 (32-256) | 64 | 4096-8192 |
| 8B, 并发 16, 长序列 | 128 | 3072-4096 |

#### 3.2.4 批次 Token 预算（`INFINILM_MAX_NUM_BATCHED_TOKENS`）

控制每个调度步骤能批处理的最大 token 数（默认 = `max_position_embeddings`）。对于长序列 + 高并发混合场景，可以显式限制：

```bash
export INFINILM_MAX_NUM_BATCHED_TOKENS=8192
```

- 更大的值 → 更激进批处理 → 更高吞吐但 TTFT 可能变差
- 更小的值 → prefill 切得更细 → 更好的延迟但 GPU 利用率可能不足

### 3.3 方案 B：Static Attention + INT8 KV Cache（长序列备选）

当 KV Cache 显存是瓶颈时，可以牺牲 Paged Attention 换取 INT8 KV Cache。INT8 让 KV Cache 从 FP16 减半为 INT8 → 同样的显存，块数可翻倍。

```bash
python python/infinilm/server/inference_server.py \
  --device nvidia \
  --model=/models/Llama-3.1-8B-Instruct \
  --temperature 1.0 --top-p 0.8 --top-k 1 \
  --port 8102 --tp 1 \
  --max-new-tokens 4096 \
  --max-cache-len 8192 \
  --kv-cache-dtype int8 \
  --enable-graph \
  --ignore-eos
```

> ⚠️ **关键限制：不能同时使用 `--enable-paged-attn`、`--attn flash-attn` 与 `--kv-cache-dtype int8`。** INT8 KV 量化仅在 Static Attention（`static_attn.cpp`）后端中实现。

**方案 A vs B 选择：**

| 条件 | 推荐 |
|------|------|
| 高并发（16+）+ 短序列 | 方案 A（Paged + Flash Attn） |
| 低并发 + 超长序列（4096+）| 方案 B（Static + INT8 KV） |
| 高并发 + 长序列 | 方案 A 优先，OOM 则切方案 B |

### 3.4 70B 模型专属优化

#### 3.4.1 Tensor Parallelism（`--tp`）

70B FP16 ≈ 140GB，必须多卡：

```bash
# 4 × 4090
--tp 4

# 8 × 4090（推荐，KV Cache 余量充足）
--tp 8
```

TP 越大，单卡显存越小但通信开销越大。

#### 3.4.2 权重量化（自动检测）

如果模型权重已经做了 AWQ/GPTQ 量化（`config.json` 中有 `quantization_config` 字段），InfiniLM 会自动检测并加载。支持：

| 量化方式 | 权重精度 | 显存 vs FP16 | 实现文件 |
|----------|----------|-------------|----------|
| AWQ | W4A16 | ~25% | `awq.cpp`, `awq_marlin.cpp` |
| GPTQ | W4A16 | ~25% | `gptq.cpp`, `gptq_marlin.cpp` |
| Compressed-Tensors | W8A8 | ~50% | `compressed_tensors.cpp` |

70B 用 4-bit 量化后 ~35GB，单卡 RTX 4090 也能跑（不含 KV Cache 的话）。

#### 3.4.3 MoE Expert Parallelism（`--ep`）

仅对 MoE 架构模型（Qwen3-MoE、DeepSeek-V2）有效。Llama 是 Dense 模型，不适用。

### 3.5 其他已有优化（内部自动启用）

| 优化 | 说明 |
|------|------|
| **前缀缓存** | `scheduler.py` 中 block 级 xxHash64 hash 匹配 + 引用计数共享，已自动启用 |
| **异步权重加载** | `--weight-load async`（默认），TP 场景下各 rank 并行加载 |
| **融合 MoE** | `--skip-legacy-moe`，Qwen3 MoE 模型使用融合内核 |

---

## 四、代码层优化方向

纯配置调优不够时，以下方向可以直接改代码。

### 4.1 Chunked Prefill（高优先级）

**位置：** `python/infinilm/llm/scheduler.py`（`schedule()` 方法）

**问题：** 当 4096 token 的 prefill 请求到来时，调度器将其整个送入一次 forward → decode 请求被阻塞，TTFT 和 ITL 抖动严重。

**方案：** 在 `schedule()` 中拆分长 prefill 为多个 chunk：

```python
# scheduler.py schedule() 中
MAX_PREFILL_TOKENS_PER_STEP = 2048  # 可配

# 对 waiting queue 中的请求，如果 prefill tokens > 阈值，拆分为 chunk
# chunk 之间让 decode 请求插队执行
```

**效果预期：** 长 prefill + 高并发混合场景，decode ITL 稳定性提升 30-50%。

### 4.2 调度策略优化（中优先级）

**位置：** `python/infinilm/llm/scheduler.py`

当前策略是先排空 waiting queue 再做 decode。可以改进：

- **Prefill/Decode 交替调度**：每步交替执行一批 prefill 和一批 decode，避免单边饥饿
- **Decode 优先 + Prefill 兜底**：decode 有请求时优先处理（因为对延迟敏感），无 decode 时才做 prefill

### 4.3 动态 Block Size（中优先级）

**位置：** `python/infinilm/llm/cache_manager.py`

当前 `block_size` 是全局固定的。短序列用小 block 减少浪费，长序列用大 block 减少查表开销。可以实现：

- 按请求长度自适应选择 block_size
- 或者用多级 block pool（小 block pool + 大 block pool）

### 4.4 GEMM 与 Kernel 调优（低优先级）

**位置：** `csrc/layers/linear/`

- cuBLAS workspace 调优
- 确保 batch 对齐到 128 的整数倍以最大化 Tensor Core 利用率
- RMS Norm + Residual Add 融合为一个 kernel

### 4.5 Prefill-Decode 分离（需要额外 GPU）

**位置：** `--kv-transfer-config` + `python/infinilm/llm/scheduler.py`

InfiniLM 已支持 PD 分离（通过 Mooncake KV Transfer）。把 prefill 和 decode 分配到不同 GPU 组：
- Prefill 节点处理长 prompt，完成后 KV Cache 通过 RDMA/TCP 传给 decode 节点
- 完全消除 prefill 对 decode 延迟的影响
- 适合 70B + 多卡 + 高并发场景

```bash
# Prefill 节点
--kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer"}'

# Decode 节点
--kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer"}'
```

### 4.6 编译期优化

```bash
# InfiniLM 编译时启用 KV Caching 算子
cd /root/difanrui/InfiniLM
xmake f --use-kv-caching=true
xmake build
```

`--use-kv-caching` 定义 `ENABLE_KV_CACHING` 预处理器宏，在 NVIDIA/天数等平台启用 KV 缓存专用算子。

---

## 五、优化路线图

```
Phase 1: Baseline
  └── 参考命令原样跑，记录所有指标

Phase 2: 配置调优（方案 A）
  ├── num_blocks / block_size 精确计算（匹配 24GB 显存）
  ├── INFINILM_MAX_NUM_BATCHED_TOKENS 调优
  └── 不同并发下的 max_batch_size 调优

Phase 3: 方案 B 对比（长序列场景）
  ├── Static + INT8 KV Cache
  └── 与方案 A 的吞吐/延迟对比

Phase 4: 代码级优化
  ├── Chunked Prefill（scheduler.py）
  ├── 调度策略调整
  └── block size 优化

Phase 5: 70B 模型
  ├── TP 多卡配置
  ├── AWQ/GPTQ 量化
  └── PD 分离（如果 GPU 够多）

Phase 6: 编译期优化
  └── xmake --use-kv-caching=true
```

---

## 六、关键指标

| 指标 | 含义 | 目标 |
|------|------|------|
| **Output Throughput** (tok/s) | decode 阶段每秒 token 数 | 提升 |
| **Total Throughput** (tok/s) | prefill + decode 总吞吐 | 提升 |
| **TTFT** (ms) | 首 token 延迟 | 不退化 |
| **ITL** (ms) | decode 每 token 延迟 | 不退化 |
| **GPU 显存利用率** | `nvidia-smi` used / total | 尽量用满不 OOM |

---

## 七、关键源文件索引

| 文件 | 作用 |
|------|------|
| `python/infinilm/base_config.py` | 所有 CLI 参数定义 |
| `python/infinilm/server/inference_server.py` | 服务入口 |
| `python/infinilm/llm/scheduler.py` | **连续批处理调度器**（核心） |
| `python/infinilm/llm/cache_manager.py` | **Block Manager + 前缀缓存** |
| `python/infinilm/llm/model_runner/model_runner.py` | ModelRunner |
| `python/infinilm/infer_engine.py` | Python InferEngine |
| `csrc/engine/infer_engine.cpp` | C++ 引擎入口 |
| `csrc/engine/rank_worker.cpp` | RankWorker（forward + graph replay） |
| `csrc/engine/compiler/paged_compiler.cpp` | Paged CUDA Graph 编译 |
| `csrc/layers/attention/backends/flash_attn.cpp` | FlashAttention 后端 |
| `csrc/layers/attention/backends/paged_attn.cpp` | PagedAttention 后端 |
| `csrc/layers/attention/backends/static_attn.cpp` | **Static Attention + INT8 KV** |
| `csrc/layers/quantization/` | AWQ/GPTQ/KV 量化实现 |
| `csrc/cache/kv_cache.cpp` | KV Cache 分配逻辑 |

---

## 八、环境准备清单

- [x] Python 虚拟环境 (`.venv`)
- [x] InfiniCore SDK 编译安装（CUDA 12.8 + NVIDIA GPU 后端）
- [x] InfiniLM 编译安装
- [ ] 8B 模型下载
- [ ] 70B 模型下载（需额外 GPU 卡）
- [ ] 服务器依赖（`pip install uvicorn fastapi`）
- [ ] Flash Attention 子模块 + 编译
- [ ] 压测工具（`pip install vllm`）
