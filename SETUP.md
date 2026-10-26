# 环境配置与运行说明

## 当前环境

- Conda 环境：`adit`
- Python：3.10.22
- PyTorch：2.3.1 CPU 版
- PyG：torch_geometric 2.6.1
- torch_scatter：2.1.2+pt23cpu
- GPU：Intel Arc 130T；当前 PyTorch 环境不支持 CUDA

该 CPU 环境适合数据加载、冒烟测试（smoke test）、采样和小规模推理。完整 VAE/DiT
训练建议在支持 CUDA 的 NVIDIA GPU 机器上运行。

## 激活环境

```powershell
conda activate adit
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
```

在 Windows 上设置 UTF-8 很重要。部分化学依赖会用进程默认编码读取数据
文件；Carbon-24 的纯碳有效性检查已不依赖 SMACT 数据表，但保留的多元
HiLo 实验仍会使用 SMACT。

运行实验前，建议先确认实际使用的解释器：

```powershell
python --version
python -c "import torch, lightning, torch_geometric; print(torch.__version__, torch.cuda.is_available(), lightning.__version__, torch_geometric.__version__)"
```

预期能看到 Python 3.10、PyTorch 2.3.1、Lightning 2.4.0 和 PyG 2.6.1。
本机 `torch.cuda.is_available()` 应为 `False`。

## 数据

Carbon-24 数据已位于：

```text
data/carbon_24/
  train.csv
  val.csv
  test.csv
  train_ori.pt
  val_ori.pt
  processed/test.pt    首次加载测试集时自动生成
```

当前不需要额外下载数据或运行预处理脚本。测试集缓存由 `test.csv` 解析得到；
除非源 CSV 发生变化，否则不需要删除 `processed/test.pt`。

## 冒烟测试（Smoke Test）

验证固定每 10 个 epoch 执行一次。以下命令每个 epoch 运行一个训练 batch，
共执行 10 次训练更新，并在第 10 个 epoch 运行一个验证 batch：

```powershell
python src/train_csp.py `
  --stage vae `
  --epochs 10 `
  --limit-batches 1 `
  --batch-size 2 `
  --output-dir runs/smoke_vae
```

使用刚生成的 VAE 检查点（checkpoint）运行一个 DiT 训练 batch 和一个
验证 batch：

```powershell
python src/train_csp.py `
  --stage dit `
  --vae-ckpt runs/smoke_vae/checkpoints/last.ckpt `
  --epochs 10 `
  --limit-batches 1 `
  --batch-size 2 `
  --output-dir runs/smoke_dit
```

运行一个测试目标和一次候选采样：

```powershell
python src/test_csp.py `
  --ckpt runs/smoke_dit/checkpoints/last.ckpt `
  --vae-ckpt runs/smoke_vae/checkpoints/last.ckpt `
  --candidates 1 `
  --limit 1 `
  --device cpu `
  --output runs/smoke_test
```

这些命令默认在 `adit` 环境中运行。本机 `--device auto` 会解析为 CPU。

完整冒烟测试会构建：

- 51.4M 参数的 VAE；
- 32.7M 参数的 DiT-S；
- 一个测试目标的候选结构。

如果随机模型生成非法晶格，测试程序会记录该候选为结构无效，而不会中断整个
评价流程。

## 已验证内容

当前 Windows CPU 环境已完成以下验证：

- VAE：单 batch 训练 10 个 epoch，并执行验证。
- DiT：加载冻结 VAE，单 batch 训练 10 个 epoch，并执行验证。
- CSP：一个测试目标、一个候选、100 步采样，并生成汇总指标。
- `experiment/hilo_signal`：5 个原有测试全部通过。
- 全部 `src` Python 文件通过语法检查。

这些验证只说明环境与数据流程可用，不代表模型已经收敛或具备有效生成能力。

## 正式训练

完整的 VAE、DiT 和 CSP 测试命令见 [README.md](README.md)。正式训练前应
确认 checkpoint 输出目录有足够磁盘空间，并在 NVIDIA GPU 环境中使用
`--device cuda`。

## Windows 注意事项

- `joblib_map` 在受限 Windows 环境中会自动退化为顺序执行。
- DataLoader workers 固定为 `0`，以避免 Windows 多进程加载问题。
- 新入口使用 `CSVLogger`，日志输出到指定 `--output-dir` 下。
- 运行化学相关评价时保留 `PYTHONUTF8=1` 和 `PYTHONIOENCODING=utf-8`。
