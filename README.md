# 基于 ADiT 的晶体结构预测

本项目支持 Carbon-24、MP-20、MPTS-52 和 Perov-5 四套晶体结构数据。模型
接收目标成分，在潜空间中生成候选结构，再由 VAE 解码器输出原子类型、
分数坐标和晶格参数。每次实验只使用一套数据，不进行跨数据集联合训练。

## 项目范围

- 数据集：`carbon_24`、`mp_20`、`mpts_52`、`perov_5`。
- 模型：HiLo 注意力编码的成分条件 VAE + CoDeMDT 潜空间流匹配。
- 输入：100 维元素计数向量。
- 输出：候选晶体结构、有效候选的 CIF 文件和 CSP 评价指标。

| 数据集参数 | 数据目录 | 训练集 | 验证集 | 测试集 | 原子数 |
| --- | --- | ---: | ---: | ---: | --- |
| `carbon_24` | `data/carbon_24` | 6,091 | 2,032 | 2,030 | 6–24 |
| `mp_20` | `data/mp_20` | 27,136 | 9,047 | 9,046 | 1–20 |
| `mpts_52` | `data/mpts_52` | 27,380 | 5,000 | 8,096 | 1–52 |
| `perov_5` | `data/perov_5` | 11,356 | 3,787 | 3,785 | 5 |

四套数据均直接使用各自已有的 train/val/test 划分，不重新划分。

原先的 QM9/MP20/QMOF 联合训练、Hydra 配置和旧评估入口已从当前支持范围
中移除。

## 模型流程

第一阶段训练 Transformer VAE。编码器接收原子类型、笛卡尔坐标、分数坐标
和 token 位置，将其压缩为 8 维潜变量。编码器默认使用 HiLo 双频注意力：
Hi-Fi 头在周期性最小镜像邻域内建模局部原子相互作用，Lo-Fi 头在池化 token
和晶格描述符上建模全局上下文。解码器预测原子类型、分数坐标以及晶格长度
和晶格角；它保留标准自注意力，因为生成阶段尚未获得目标分数坐标和晶胞，
若使用 HiLo 会引入训练时可见、推理时不可见的信息。

第二阶段冻结 VAE，在 8 维潜空间中训练流匹配去噪器。默认去噪器为
CoDeMDT，使用 MDTv2 式的非对称掩码编码器、U-Net 长跳连、Side Interpolator
（SI）和带噪声潜变量跳连的解码器。训练同时执行普通路径与随机掩码路径，
并在掩码编码器上加入正交、互信息相似度和维度分散三项多样性约束。推理只走
普通路径，并继续使用无分类器引导（classifier-free guidance）。当前固定支持
三种去噪器规模：

| 规模 | `d_model` | `nhead` | `num_layers` |
| --- | ---: | ---: | ---: |
| `s` | 384 | 6 | 12 |
| `b` | 768 | 12 | 12 |
| `l` | 1024 | 16 | 24 |

`nhead` 是每个 block 内部的注意力头数，不增加 block 数量。
`num_layers` 是编码器、SI 和解码器合计的 block 预算。默认
`decode_layer=2`，其余 block 均分给编码器两半，SI 另占一个 block。

原始 `DiT` 与标准 VAE 注意力仍保留为消融基线：

```powershell
python src/train_csp.py --stage vae --vae-attention standard ...
python src/train_csp.py --stage dit --denoiser dit ...
```

默认配置与旧 checkpoint 的模型结构不兼容。旧 VAE 可直接作为新 CoDeMDT
的阶段二输入；若要恢复完整的旧训练路径，应同时显式指定
`--vae-attention standard` 和 `--denoiser dit`。

当前模型从编码器到解码器均支持成分条件。原项目在 QM9/MP20 上训练的联合
模型权重与本版本的输入和条件结构不兼容，不能直接用于这些 CSP 实验。

## 目录结构

```text
data/<dataset>/                 train.csv、val.csv、test.csv、processed/*.pt
src/
  train_csp.py                  VAE 或潜空间去噪器训练入口
  test_csp.py                   候选结构采样、CIF 输出和 CSP 评价
  eval_cka.py                   独立计算已训练去噪器的层间 CKA
  data/
    crystal_datamodule.py       数据集选择与 DataModule
    components/carbon24_dataset.py
    components/crystal_csv_dataset.py
    components/cif.py
    components/composition_utils.py
  models/
    vae_module.py               成分条件 VAE
    ldm_module.py               潜空间采样与流匹配模块
    components/hiloattention.py HiLo 周期晶体注意力
    encoders/transformer.py
    decoders/transformer.py
    denoisers/dit.py
    denoisers/code_mdt.py       CoDeMDT、多样性损失和有界权重
    interpolants/flow_matching.py
  eval/
    crystal.py
    crystal_reconstruction.py
tests/
  test_integrated_models.py      HiLo + CoDeMDT 的 CPU 集成测试
```

以下目录被保留，因为其中包含已有测试结果或论文图件：

- `experiment2/`
- `预训练权重测试/`
- `ccfa-workfiles/`

## 环境激活

```powershell
conda activate adit
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
```

完整环境说明见 [SETUP.md](SETUP.md)。

## 选择数据集

四个入口都通过 `--dataset` 独立选择数据：

```powershell
$DATASET = "perov_5"
```

可选值为 `carbon_24`、`mp_20`、`mpts_52`、`perov_5`。默认值为
`carbon_24`。新加入的三套数据首次加载时会解析 CSV 中的 CIF，并缓存到
`data/<dataset>/processed/train.pt、val.pt、test.pt`。当前缓存已经生成，
后续启动可直接加载。

## 训练控制台输出

训练使用自定义 TQDM 进度条，不使用 Lightning 默认进度条，避免旧验证指标
停留在训练条中。训练条持续显示当前 batch loss、epoch 平均 loss 和学习率：

```text
Train 10/3000: 100%|...| 190/190 [.., batch_loss=0.2935, epoch_avg=0.4040, lr=1.0e-04]
```

`batch_loss` 是当前 batch 的原始 loss，`epoch_avg_loss` 是当前 epoch 到
目前为止的平均 loss。每个 epoch 结束会先输出训练汇总，再切换为 Validation
进度条。验证固定每 10 个 epoch 执行一次，结束后会显示验证 loss、结构
匹配率和 RMSD：

```text
[训练完成] epoch 10/3000 train_loss=0.404000 lr=1.00e-04
[验证] epoch 10/3000 loss=3.715295 match_rate=0.000000 rms_dist=10.000000
```

每个 epoch 结束后，还会写入一份完整的训练指标：

```text
<output-dir>/train_metrics.csv
<output-dir>/validation_metrics.csv
```

`train_metrics.csv` 每行对应一个 epoch，包含所有训练分项 loss、学习率和
权重衰减。`validation_metrics.csv` 每行对应一次验证，包含验证分项 loss、
匹配率和 RMSD。Lightning 原生的 `logs/.../metrics.csv` 仍会保留用于调试，
但它是稀疏宽表：训练行中的验证列为空是正常现象，应优先使用上述两个文件。

训练默认启用提前停止（early stopping），监控 `val/loss`。连续 5 次
验证没有改善时会自动停止；每 10 个 epoch 验证一次，因此 patience 为 5
约等于连续 50 个 epoch 没有改善。`--epochs` 仅表示最大训练轮数。
学习率使用 `ReduceLROnPlateau` 动态调整，同样监控 `val/loss`。
允许连续 2 次验证没有改善；如果下一次仍无改善，学习率乘以 0.5，最低降至
`1e-6`。学习率发生变化时会输出：

```text
[学习率] 1.00e-04 -> 5.00e-05
```

每个 epoch 的学习率也会写入 CSV 日志，字段名由 Lightning 生成，通常为
`lr-AdamW`，可用于绘制学习率曲线。

训练完成后会自动绘制训练 loss 与验证 loss 随 epoch 变化的曲线。提前停止
时也会在停止后生成图表：

```text
<output-dir>/loss_curve.png
<output-dir>/loss_history.csv
```

`loss_history.csv` 保存每个 epoch 的平均训练 loss，以及发生验证的 epoch 的
验证 loss。控制台会输出图表保存路径。

## 第一阶段：训练 VAE

```powershell
python src/train_csp.py `
  --dataset $DATASET `
  --stage vae `
  --epochs 300 `
  --batch-size 32 `
  --output-dir "runs/${DATASET}_vae"
```

在 NVIDIA GPU 机器上增加 `--device cuda`。本机默认使用 CPU。

CPU 快速检查：

```powershell
python src/train_csp.py `
  --dataset carbon_24 `
  --stage vae `
  --epochs 10 `
  --limit-batches 1 `
  --batch-size 2 `
  --output-dir runs/smoke_vae
```

`--limit-batches 1` 表示每个 epoch 只运行一个训练 batch。验证固定每 10 个
epoch 执行一次，因此上述命令会执行 10 次训练更新、1 个验证 batch，并生成
按验证指标选择的 checkpoint。

## 第二阶段：训练 CoDeMDT

```powershell
python src/train_csp.py `
  --dataset $DATASET `
  --stage dit `
  --vae-ckpt "runs/${DATASET}_vae/checkpoints/last.ckpt" `
  --epochs 500 `
  --batch-size 32 `
  --dit-size s `
  --denoiser code_mdt `
  --mask-ratio 0.3 `
  --mask-ratio-range 0.2 `
  --decode-layer 2 `
  --output-dir "runs/${DATASET}_codemdt"
```

CoDeMDT 训练期间 VAE 保持冻结。`--dit-size` 可设为 `s`、`b` 或 `l`。
使用 `--denoiser dit` 可训练原始 DiT 基线。

CoDeMDT 的总损失为：

$$
\mathcal L =
\mathcal L_{\mathrm{diff}}
+ w_{\mathrm{mask}}\mathcal L_{\mathrm{masked}}
+ w_{\mathrm{div}}\mathcal L_{\mathrm{div}}.
$$

默认 `w_mask=1.0`。多样性权重默认按扩散损失与多样性损失的比率自适应，
并限制在 `[0.1, 0.5]`；使用 `--div-fixed` 可改为固定权重。逐项消融可设置
`--lambda-orth 0`、`--lambda-mi 0` 或 `--lambda-disp 0`。

主要 checkpoint：

```text
runs/<dataset>_vae/checkpoints/last.ckpt
runs/<dataset>_codemdt/checkpoints/last.ckpt
```

每次训练还会在对应 `--output-dir` 下生成：

```text
loss_curve.png
loss_history.csv
```

## 潜空间模型层间 CKA 热力图

潜空间去噪器训练使用 PyTorch forward hook 捕获以下层的输出：

- 输入嵌入 `embedding`
- 每个 Transformer block，例如 `block_01` 至 `block_06`
- 输出层 `output`

每次验证时，程序从最多 4 个验证 batch 中收集有效原子 token 的表示，并最多
均匀抽取 512 个 token 计算线性 CKA 矩阵。验证结束后保存：

```text
<output-dir>/cka_heatmap.png
<output-dir>/cka_matrix.csv
```

热力图颜色范围为 0–1。CKA 越接近 1，表示两层内部表示的几何结构越相似；
对角线固定为 1。CKA 只描述表示之间的相似性，不代表生成质量或结构有效性。

核心函数 `linear_cka` 也可用于两个模型：只要两边的特征矩阵使用完全相同的
样本顺序，就可以直接计算跨模型 CKA。

### 对已训练模型单独运行 CKA

CKA 不依赖训练过程。训练完成后可以直接运行：

```powershell
python src/eval_cka.py `
  --dataset $DATASET `
  --ckpt "runs/${DATASET}_codemdt/checkpoints/last.ckpt" `
  --vae-ckpt "runs/${DATASET}_vae/checkpoints/last.ckpt" `
  --device cpu `
  --split val `
  --batches 4 `
  --output "runs/${DATASET}_cka"
```

该脚本只执行模型前向传播，不更新权重、不写入 checkpoint，也不会启动训练。
输出包括：

```text
runs/<dataset>_cka/cka_heatmap.png
runs/<dataset>_cka/cka_matrix.csv
runs/<dataset>_cka/cka_metadata.json
```

`cka_metadata.json` 记录 checkpoint、数据划分、batch 数、随机种子、模型维度
和参与计算的层名，便于复现实验。

## 测试 CSP

```powershell
python src/test_csp.py `
  --dataset $DATASET `
  --ckpt "runs/${DATASET}_codemdt/checkpoints/last.ckpt" `
  --vae-ckpt "runs/${DATASET}_vae/checkpoints/last.ckpt" `
  --candidates 4 `
  --limit 50 `
  --device cpu `
  --output "runs/${DATASET}_test"
```

`--limit 0` 表示评价所选数据集的完整测试集。输出文件为：

```text
runs/<dataset>_test/summary.json
runs/<dataset>_test/per_target.json
runs/<dataset>_test/generated/*.cif
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
--dataset carbon_24|mp_20|mpts_52|perov_5
--stage vae|dit
--epochs
--batch-size
--dropout              默认 0.05，作用于 VAE 和去噪器的 Attention/MLP
--weight-decay         AdamW 权重衰减，默认 1e-4
--vae-ckpt          仅阶段二需要
--dit-size s|b|l
--denoiser code_mdt|dit         默认 code_mdt
--vae-attention hi_lo|standard 默认 hi_lo
--hilo-low-freq-heads           默认 2
--hilo-cutoff                   默认 5.0 Å
--hilo-pool-type grid|chunk|none
--mask-ratio                    默认 0.3
--mask-ratio-range              默认 0.2
--mask-loss-weight              默认 1.0
--decode-layer                  CoDeMDT 解码器 block 数，默认 2
--rel-pos-bias 1d|none          默认 1d
--div-weight / --div-weight-min / --div-weight-max
--div-warmup / --div-fixed
--lambda-orth / --lambda-mi / --lambda-disp
--device auto|cpu|cuda
--output-dir
--limit-batches     仅用于快速验证，0 表示使用全部 batch
--patience          连续未改善的验证次数，默认 5
```

其余训练设置固定在脚本中，以保持命令行简洁：

| 设置 | 固定值 |
| --- | --- |
| VAE 学习率 | `1e-4` |
| 去噪器学习率 | `1e-4` |
| Dropout | `0.05` |
| AdamW weight decay | `1e-4` |
| 潜变量维度 | `8` |
| 采样步数 | `100` |
| CFG scale | `2.0` |
| 验证间隔 | 每 10 个 epoch |
| Early stopping patience | `5` 次验证 |
| Early stopping min_delta | `1e-4` |
| 学习率调度器 | `ReduceLROnPlateau` |
| 学习率下降倍数 | `0.5` |
| 学习率下降 patience | `2` 次验证 |
| 最低学习率 | `1e-6` |
| CKA 最多验证 batch 数 | `4` |
| CKA 最多 token 数 | `512` |
| DataLoader workers | `0` |
| 随机种子 | `0` |

权重衰减只作用于二维及以上的矩阵参数；bias、LayerNorm 和 embedding
使用 `weight_decay=0`。VAE 和去噪器的 Attention 与 MLP 均使用 `0.05`
dropout。可以通过命令行调整这两个值，例如：

```powershell
python src/train_csp.py `
  --dataset carbon_24 `
  --stage vae `
  --dropout 0.05 `
  --weight-decay 1e-4
```

## 测试命令参数

```text
--dataset carbon_24|mp_20|mpts_52|perov_5
--ckpt
--vae-ckpt
--candidates
--limit             0 表示评价所选数据集的全部测试结构
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
