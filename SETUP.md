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
文件；MP-20、MPTS-52 和 Perov-5 的成分有效性评价会使用 SMACT。

运行实验前，建议先确认实际使用的解释器：

```powershell
python --version
python -c "import torch, lightning, torch_geometric; print(torch.__version__, torch.cuda.is_available(), lightning.__version__, torch_geometric.__version__)"
```

预期能看到 Python 3.10、PyTorch 2.3.1、Lightning 2.4.0 和 PyG 2.6.1。
本机 `torch.cuda.is_available()` 应为 `False`。

## 数据

当前支持四套独立数据：

```text
data/carbon_24/     train.csv、val.csv、test.csv、已有 *_ori.pt
data/mp_20/         train.csv、val.csv、test.csv
data/mpts_52/       train.csv、val.csv、test.csv
data/perov_5/       train.csv、val.csv、test.csv
```

| 数据集 | 训练集 | 验证集 | 测试集 |
| --- | ---: | ---: | ---: |
| `carbon_24` | 6,091 | 2,032 | 2,030 |
| `mp_20` | 27,136 | 9,047 | 9,046 |
| `mpts_52` | 27,380 | 5,000 | 8,096 |
| `perov_5` | 11,356 | 3,787 | 3,785 |

新加入的三套数据都已从 CSV 中的 CIF 完整解析并缓存到各自
`processed/train.pt、val.pt、test.pt`，没有跳过记录。CSV 文件大小或修改
时间发生变化时，缓存会自动重建。

## 冒烟测试（Smoke Test）

验证固定每 10 个 epoch 执行一次。以下命令每个 epoch 运行一个训练 batch，
共执行 10 次训练更新，并在第 10 个 epoch 运行一个验证 batch：

```powershell
python src/train_csp.py `
  --dataset carbon_24 `
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
  --dataset carbon_24 `
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
  --dataset carbon_24 `
  --ckpt runs/smoke_dit/checkpoints/last.ckpt `
  --vae-ckpt runs/smoke_vae/checkpoints/last.ckpt `
  --candidates 1 `
  --limit 1 `
  --device cpu `
  --output runs/smoke_test
```

这些命令默认在 `adit` 环境中运行。本机 `--device auto` 会解析为 CPU。

完整冒烟测试会构建：

- 26.2M 参数的 VAE；
- 16.8M 参数的 DiT-S；
- 一个测试目标的候选结构。

如果随机模型生成非法晶格，测试程序会记录该候选为结构无效，而不会中断整个
评价流程。

## 已验证内容

当前 Windows CPU 环境已完成以下验证：

- `mp_20`、`mpts_52`、`perov_5` 的 train/val/test CIF 全部解析成功。
- 三套新数据的 VAE 单 batch 前向全部通过。
- Perov-5：单 batch VAE 训练 10 个 epoch，并执行多元素验证。
- Perov-5：一个测试目标、一个候选、100 步采样的 CSP 流程通过。
- VAE/DiT 的 dropout、AdamW 权重衰减分组和 checkpoint 往返加载通过。
- 全部 `src` Python 文件通过语法检查。

这些验证只说明环境与数据流程可用，不代表模型已经收敛或具备有效生成能力。

## Loss 曲线

训练完成或触发提前停止后，程序会使用 matplotlib 的无界面后端自动保存：

```text
<output-dir>/loss_curve.png
<output-dir>/loss_history.csv
```

图中蓝线是每个 epoch 的训练 loss 平均值，红线和圆点表示验证 loss。验证
固定每 10 个 epoch 执行一次，因此验证 loss 的点数会少于训练 loss。续训时
程序会读取已有 `loss_history.csv`，保留当前 epoch 之前的曲线数据。

## 提前停止

训练默认监控 `val_mp20/loss`，并使用 `min_delta=1e-4`。指标名称中的
`mp20` 是兼容旧 checkpoint 的遗留前缀，不代表当前一定使用 MP-20。当前每 10 个
epoch 验证一次，默认 `--patience 5`，即连续 5 次验证没有改善后停止，约
等于连续 50 个 epoch 没有改善。

`--epochs` 是最大训练轮数，不是固定执行轮数。若希望更早停止，可以减小
patience；若验证指标仍有明显改善，可以增大 patience 或最大 epoch 数。

## 动态学习率

VAE 和 DiT 都使用 `ReduceLROnPlateau`，监控 `val_mp20/loss`：

- 初始学习率为 `1e-4`。
- 允许连续 2 次验证没有相对改善；如果下一次仍无改善，学习率乘以 `0.5`。
- 最低学习率固定为 `1e-6`。
- 调度器只在验证发生的 epoch 更新，避免在没有新验证指标的 epoch 重复
  读取旧值而误降学习率。

学习率变化会直接显示在控制台。Early stopping 的 patience 为 5 次验证，
因此学习率通常会先下降 1–2 次，之后才触发停止。

## Dropout 与权重衰减

VAE 和 DiT 默认使用：

```text
dropout = 0.05
AdamW weight_decay = 1e-4
```

dropout 应用于 VAE Transformer layer 和 DiT 的 Attention/MLP。权重衰减
只应用于二维及以上的矩阵参数，bias、LayerNorm 和 embedding 不衰减。
训练开始时会输出实际配置，也可以分别运行消融：

```powershell
python src/train_csp.py --dataset carbon_24 --stage vae --dropout 0 --weight-decay 1e-4
python src/train_csp.py --dataset carbon_24 --stage vae --dropout 0.05 --weight-decay 0
python src/train_csp.py --dataset carbon_24 --stage vae --dropout 0.05 --weight-decay 1e-4
```

建议不要只凭训练 loss 选择配置，应比较验证 loss、重建匹配率和测试结果。

## 层间 CKA 分析

DiT 通过 forward hook 捕获输入嵌入、全部 Transformer block 和输出层的内部
表示。每次验证完成后会计算这些层之间的线性 CKA，并保存：

```text
<output-dir>/cka_heatmap.png
<output-dir>/cka_matrix.csv
```

为控制计算量，每次只使用前 4 个验证 batch，并最多均匀抽取 512 个有效
原子 token。热力图对角线为 1；非对角值越接近 1，表示两层保留的表示几何
结构越相似。该指标只用于分析内部表示，不直接评价晶体结构是否有效。

已训练模型无需重新训练，可以直接执行独立分析脚本：

```powershell
python src/eval_cka.py `
  --dataset perov_5 `
  --ckpt runs/perov_5_dit/checkpoints/last.ckpt `
  --vae-ckpt runs/perov_5_vae/checkpoints/last.ckpt `
  --device cpu `
  --split val `
  --batches 4 `
  --output runs/perov_5_cka
```

脚本固定随机种子为 0，只执行验证或测试集前向传播，并生成
`cka_heatmap.png`、`cka_matrix.csv` 和 `cka_metadata.json`。

## 正式训练

完整的 VAE、DiT 和 CSP 测试命令见 [README.md](README.md)。正式训练前应
确认 checkpoint 输出目录有足够磁盘空间，并在 NVIDIA GPU 环境中使用
`--device cuda`。

## Windows 注意事项

- `joblib_map` 在受限 Windows 环境中会自动退化为顺序执行。
- DataLoader workers 固定为 `0`，以避免 Windows 多进程加载问题。
- 新入口使用 `CSVLogger`，日志输出到指定 `--output-dir` 下。
- 运行化学相关评价时保留 `PYTHONUTF8=1` 和 `PYTHONIOENCODING=utf-8`。
