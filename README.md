# 基于 ADiT 的 Carbon-24 晶体结构预测

本项目已精简为面向 Carbon-24 数据集的晶体结构预测（CSP）实验。模型接收
目标成分，在潜空间中生成候选结构，再由 VAE 解码器输出原子类型、分数坐标
和晶格参数。

## 项目范围

- 数据集：Carbon-24，每个晶胞包含 6–24 个碳原子。
- 数据划分：训练集 6,091 条、验证集 2,032 条、测试集 2,030 条。
- 模型：成分条件 VAE + 潜空间流匹配 DiT。
- 输入：100 维元素计数向量。Carbon-24 中仅碳元素计数 `C_n` 非零。
- 输出：候选晶体结构、有效候选的 CIF 文件和 CSP 评价指标。

原先的 QM9/MP20/QMOF 联合训练、Hydra 配置和旧评估入口已从当前支持范围
中移除。

## 模型流程

第一阶段训练 Transformer VAE。编码器接收原子类型、笛卡尔坐标、分数坐标
和 token 位置，将其压缩为 8 维潜变量。解码器同时接收潜变量和目标成分，
预测原子类型、分数坐标以及晶格长度和晶格角。训练损失包含原子类型损失、
成分约束损失、几何重建损失和 KL 正则项。

第二阶段冻结 VAE，在 8 维潜空间中训练流匹配 DiT。DiT 使用目标成分作为
条件，并通过无分类器引导（classifier-free guidance）控制条件强度。当前固定
支持三种规模：

| 规模 | `d_model` | DiT 块数 | Attention 头数 |
| --- | ---: | ---: | ---: |
| `s` | 384 | 6 | 12 |
| `b` | 768 | 12 | 12 |
| `l` | 1024 | 16 | 24 |

当前模型从编码器到解码器均支持成分条件。原项目在 QM9/MP20 上训练的联合
模型权重与本版本的输入词表和条件结构不兼容，不能直接用于 Carbon-24 CSP。

## 目录结构

```text
data/carbon_24/                 train.csv、val.csv、test.csv、train_ori.pt、val_ori.pt
src/
  train_csp.py                  VAE 或 DiT 训练入口
  test_csp.py                   候选结构采样、CIF 输出和 CSP 评价
  data/
    carbon24_datamodule.py
    components/carbon24_dataset.py
    components/cif.py
    components/composition_utils.py
  models/
    vae_module.py               成分条件 VAE
    ldm_module.py               潜空间采样与流匹配模块
    encoders/transformer.py
    decoders/transformer.py
    denoisers/dit.py
    interpolants/flow_matching.py
  eval/
    crystal.py
    crystal_reconstruction.py
```

以下目录被保留，因为其中包含已有实验、预训练权重测试结果或论文图件：

- `experiment/`
- `experiment2/`
- `预训练权重测试/`
- `ccfa-workfiles/`

其中 `experiment/hilo_signal/` 是保留的 HiLo 对比实验，仍使用 MP20 兼容
数据读取与预处理代码。

## 环境激活

```powershell
conda activate adit
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
```

完整环境说明见 [SETUP.md](SETUP.md)。

## 第一阶段：训练 VAE

```powershell
python src/train_csp.py `
  --stage vae `
  --epochs 300 `
  --batch-size 32 `
  --output-dir runs/carbon24_vae
```

在 NVIDIA GPU 机器上增加 `--device cuda`。本机默认使用 CPU。

CPU 快速检查：

```powershell
python src/train_csp.py `
  --stage vae `
  --epochs 10 `
  --limit-batches 1 `
  --batch-size 2 `
  --output-dir runs/smoke_vae
```

`--limit-batches 1` 表示每个 epoch 只运行一个训练 batch。验证固定每 10 个
epoch 执行一次，因此上述命令会执行 10 次训练更新、1 个验证 batch，并生成
按验证指标选择的 checkpoint。

## 第二阶段：训练 DiT

```powershell
python src/train_csp.py `
  --stage dit `
  --vae-ckpt runs/carbon24_vae/checkpoints/last.ckpt `
  --epochs 500 `
  --batch-size 32 `
  --dit-size s `
  --output-dir runs/carbon24_dit
```

DiT 训练期间 VAE 保持冻结。`--dit-size` 可设为 `s`、`b` 或 `l`。

主要 checkpoint：

```text
runs/carbon24_vae/checkpoints/last.ckpt
runs/carbon24_dit/checkpoints/last.ckpt
```

## 测试 CSP

```powershell
python src/test_csp.py `
  --ckpt runs/carbon24_dit/checkpoints/last.ckpt `
  --vae-ckpt runs/carbon24_vae/checkpoints/last.ckpt `
  --candidates 4 `
  --limit 50 `
  --device cpu `
  --output runs/carbon24_test
```

`--limit 0` 表示评价完整的 2,030 条测试结构。输出文件为：

```text
runs/carbon24_test/summary.json
runs/carbon24_test/per_target.json
runs/carbon24_test/generated/*.cif
```

### 评价指标

- `composition_match_rate`：所有候选中，元素计数与目标成分完全一致的候选
  比例。
- `composition_success_at_k`：至少有一个候选成分正确的目标比例。
- `struct_valid_rate`：结构合法候选的比例。合法结构要求晶格参数有限且
  为正、晶胞体积有效，并且任意两个原子间距离不小于 0.5 Å。
- `overall_valid_rate`：同时满足成分正确和结构合法的候选比例。
- `structure_match_at_k`：至少有一个候选被 StructureMatcher 匹配到目标
  结构的比例。
- `mean_best_rms`：每个目标的最佳有限 StructureMatcher RMSD 的均值。

StructureMatcher 固定使用 `ltol=0.3`、`stol=0.5` 和 `angle_tol=10°`。

快速验证检查点仅使用一个 batch 训练，其输出接近随机模型。对应指标只能
证明训练、采样、CIF 导出和指标计算流程可运行，不能作为模型效果结论。

## 训练命令参数

`src/train_csp.py` 已移除大部分低频参数，常用参数如下：

```text
--stage vae|dit
--epochs
--batch-size
--vae-ckpt          仅 DiT 阶段需要
--dit-size s|b|l
--device auto|cpu|cuda
--output-dir
--limit-batches     仅用于快速验证，0 表示使用全部 batch
```

其余训练设置固定在脚本中，以保持命令行简洁：

| 设置 | 固定值 |
| --- | --- |
| VAE 学习率 | `1e-4` |
| DiT 学习率 | `1e-4` |
| 潜变量维度 | `8` |
| 采样步数 | `100` |
| CFG scale | `2.0` |
| 验证间隔 | 每 10 个 epoch |
| DataLoader workers | `0` |
| 随机种子 | `0` |

## 测试命令参数

```text
--ckpt
--vae-ckpt
--candidates
--limit             0 表示评价全部 2,030 条测试结构
--device cpu|cuda
--output
```

## 已移除的旧入口

Hydra 配置和以下旧入口已不再属于当前支持范围：

- `train_autoencoder`
- `train_diffusion`
- `eval_autoencoder`
- `eval_diffusion`
- `eval_csp`

现在统一使用 `src/train_csp.py` 和 `src/test_csp.py`。
