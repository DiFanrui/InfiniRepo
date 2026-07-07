# InfiniLM 优化项目

T2-1-2 服务能力优化 — InfiniLM 推理引擎性能优化。

## 目录结构

```
InfiniLM/     # InfiniTensor/InfiniLM 推理引擎
InfiniCore/   # InfiniTensor/InfiniCore C++ 核心库
```

## 新机器快速搭建

详细步骤见 [InfiniLM/docs_extern/环境搭建指南.md](InfiniLM/docs_extern/环境搭建指南.md)

### 30 秒速览

```bash
# 1. 确认 CUDA 版本匹配
nvidia-smi | grep "CUDA Version"   # 看驱动支持的最高版本
nvcc --version                      # 看 Toolkit 版本
# 两者必须匹配，不匹配则降级 Toolkit

# 2. 初始化第三方库
cd InfiniCore
git submodule update --init third_party/spdlog third_party/nlohmann_json
git clone --depth 1 https://github.com/NVIDIA/cutlass.git third_party/cutlass

# 3. 编译 InfiniCore（GPU 后端）
python3 scripts/install.py --nv-gpu=y --cuda=/usr/local/cuda

# 4. 安装 InfiniLM
cd ../InfiniLM
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# 5. 安装 InfiniCore Python 包 + 编译 Python 绑定
cd ../InfiniCore
pip install -e .
XMAKE_ROOT=y xmake build -y _infinicore
mkdir -p python/infinicore/lib/ && touch python/infinicore/lib/__init__.py
cp build/linux/x86_64/release/_infinicore.cpython-*.so python/infinicore/lib/

# 6. 安装 Python 依赖
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install numpy Pillow safetensors transformers modelscope uvicorn fastapi vllm

# 7. 下载模型
modelscope download --model LLM-Research/Meta-Llama-3.1-8B-Instruct \
  --local_dir InfiniLM/models/Llama-3.1-8B-Instruct

# 8. 启动服务 + 验证
cd InfiniLM
python python/infinilm/server/inference_server.py \
  --device nvidia --model=models/Llama-3.1-8B-Instruct \
  --port 8102 --tp 1 --max-new-tokens 4096 \
  --num-blocks 64 --max-batch-size 4 --enable-graph --enable-paged-attn

# 另开终端
curl http://localhost:8102/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hi"}],"max_tokens":10}'
```

## 在当前机器 Push

```bash
# 先在 GitHub 创建新 repo，然后：
cd /root/difanrui
git remote add origin https://github.com/<your-username>/<your-repo>.git
git add -A
git commit -m "Initial commit: InfiniLM + InfiniCore with setup fixes"
git push -u origin main
```

## 代码改动

相对于原版 `InfiniTensor/InfiniLM` 和 `InfiniTensor/InfiniCore`，本仓库做了以下修改：

| 文件 | 改动 |
|------|------|
| `InfiniLM/setup.py` | `build_cpp_module()` 加 `XMAKE_ROOT=y` 环境变量 |
| `InfiniCore/scripts/install.py` | `run_cmd()` 加 `XMAKE_ROOT=y`；`install()` build 命令加 `-j4` |
| `InfiniCore/setup.py` | `run_xmake_build()` 加 `XMAKE_ROOT=y` |

## 相关文档

- [环境搭建指南](InfiniLM/docs_extern/环境搭建指南.md) — 完整的新机器环境搭建步骤
- [优化思路](InfiniLM/docs_extern/claude思路.md) — T2-1-2 性能优化方案分析
- [题目](InfiniLM/docs_extern/题目.md) — 原始赛题要求
