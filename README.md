# 基于 ADiT 的晶体结构预测

本项目支持 Carbon-24、MP-20、MPTS-52 和 Perov-5 四套晶体结构数据。模型
接收目标成分，在潜空间中生成候选结构，再由 VAE 解码器输出原子类型、
分数坐标和晶格参数。每次实验只使用一套数据，不进行跨数据集联合训练。

## 项目范围

- 数据集：`carbon_24`、`mp_20`、`mpts_52`、`perov_5`。
- 模型：HiLo 注意力编码的成分条件 VAE + 潜空间流匹配。第二阶段支持
  CoDeMDT、原始 DiT 和 DiverseDiT。
- 可选 PA-OTFM 训练模块：Minibatch OT coupling 与 Terminal posterior-MMD。
  两者独立开关，不增加网络参数。
- 可选 Decoder-aware FM 与 Crystal-MM-FM：前者增加冻结 Decoder 的
  晶胞、周期 RDF 和 Hungarian 周期坐标监督，后者用仅由训练集拟合的
  GMM 模式源替换标准高斯源。
  两者均只作用于第二阶段，可分别或同时用于 DiT、CoDeMDT 和 DiverseDiT；
  OT coupling 与这两个新模块互斥。
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

DiverseDiT 是在原始 DiT 外新增的独立第二阶段模型。它保留标准 DiT
backbone，不在普通路径中引入 MDT 的掩码编码器、Side Interpolator 或
CoDeMDT 结构。

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
    decoder_struct_loss.py      Decoder-aware FM 的晶胞、RDF 与坐标损失
    periodic_coord_loss.py      周期平移 + Hungarian 原子坐标监督
    crystal_mmfm_prior.py       Crystal-MM-FM 的 GMM 模式源
    components/hiloattention.py HiLo 周期晶体注意力
    components/ot_coupling.py   Minibatch Gaussian-posterior OT 配对
    components/posterior_mmd.py 生成潜变量与 VAE 后验的终点 MMD
    encoders/transformer.py
    decoders/transformer.py
    denoisers/dit.py
    denoisers/diverse_dit.py    DiverseDiT、长残差和层间多样性损失
    denoisers/code_mdt.py       CoDeMDT、多样性损失和有界权重
    interpolants/flow_matching.py
  eval/
    crystal.py
    crystal_reconstruction.py
scripts/
  build_crystal_gmm.py          从训练集拟合 Crystal-MM-FM 模式先验
tests/
  test_integrated_models.py      HiLo + CoDeMDT 的 CPU 集成测试
  test_diverse_dit.py            DiverseDiT 前向、CFG 和反传测试
  test_posterior_alignment.py    OT coupling、MMD 和可反传 ODE 测试
  test_decoder_aware_and_mmfm.py 两个新模块的 CPU 集成测试
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
启用 Decoder-aware FM 或 Crystal-MM-FM 时，`w_mask` 仍由
`--mask-loss-weight` 控制，默认仍是 `1.0`；新增的 cell/geo/coord 项和 GMM
源只影响各自对应的部分，不会改变这个权重。完整组合公式见
“与已有模型的组合”。

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

## 第二阶段：训练 DiverseDiT

DiverseDiT 的实现参考
[Towards Diverse Representation Learning in Diffusion Transformers](https://arxiv.org/abs/2603.04239)。
它只复用标准 `DiT` 的输入、条件、block 和输出接口，并在其后增加：

- 前半段 block 与镜像后半段 block 之间的长残差融合；
- `orthogonality + mutual-information proxy + feature dispersion` 层间
  多样性损失；
- 按多样性损失值在 `[0.1, 0.5]` 区间内平滑启用的自适应权重。

冻结 VAE 不变，重新训练第二阶段：

```powershell
python -m ADITcsp.src.train_csp `
  --dataset carbon_24 `
  --stage dit `
  --vae-ckpt "runs/carbon24_hilovae/checkpoints/last.ckpt" `
  --denoiser diverse_dit `
  --dit-size s `
  --dit-layers 6 `
  --diverse-long-residual `
  --diverse-blocks 6 `
  --diverse-pairs 10 `
  --diverse-loss-weight 1.0 `
  --diverse-orth-weight 0.33 `
  --diverse-mi-weight 0.33 `
  --diverse-disp-weight 0.33 `
  --epochs 500 `
  --batch-size 32 `
  --device cuda `
  --output-dir "runs/carbon24_hilovae_diversedit"
```

MP-20 使用相同的第二阶段流程，只需替换数据集、VAE checkpoint 和输出目录。
`--diverse-blocks` 表示参与损失计算的 block 数；block 多于该值时，每次
训练前向随机选择对应数量的 block。`--diverse-pairs 0` 表示使用所选 block
的全部 pair。

DiverseDiT 的总体损失为：

$$
\mathcal L =
\mathcal L_{\mathrm{diff}}
+ w_{\mathrm{div}}\mathcal L_{\mathrm{div}},
\qquad
\mathcal L_{\mathrm{div}} =
0.33\mathcal L_{\mathrm{orth}}
+ 0.33\mathcal L_{\mathrm{MI}}
+ 0.33\mathcal L_{\mathrm{disp}}.
$$

`w_div` 由当前 `L_div` 自适应决定：不大于 `0.1` 时为 0，在
`(0.1, 0.5]` 内线性增加，大于 `0.5` 时固定为 1。
训练日志中的 `div_raw_loss` 是原始 `L_div`，`div_loss` 是乘上
`w_div` 和 `--diverse-loss-weight` 后实际加入总损失的值。
DiverseDiT 与两个新模块组合时，`w_div` 也保持不变；新模块只在总损失上
追加自己的项或替换源分布。

DiverseDiT checkpoint 与原始 DiT、CoDeMDT checkpoint 不兼容，必须从新的
第二阶段任务开始训练。测试命令与普通第二阶段相同：

```powershell
python -m ADITcsp.src.test_csp `
  --dataset carbon_24 `
  --ckpt "runs/carbon24_hilovae_diversedit/checkpoints/last.ckpt" `
  --vae-ckpt "runs/carbon24_hilovae/checkpoints/last.ckpt" `
  --candidates 20 `
  --limit 0 `
  --device cuda `
  --output "runs/carbon24_hilovae_diversedit_test"
```

## PA-OTFM：Posterior-Aligned OT Flow Matching

PA-OTFM 在不修改 HiLo VAE 和 DiT 网络结构的前提下增加两个独立模块：

- **Minibatch OT Coupling**：按完整元素组成分组，在每个 group 内使用
  Hungarian 算法将高斯噪声与真实 posterior latent 一对一配对；只改变
  Flow Matching 路径的端点对应关系。
- **Terminal MMD Alignment**：保留计算图执行完整 flow ODE，将最终生成
  latent 与冻结 VAE posterior latent 做多尺度 RBF MMD。

两个模块都没有可训练网络参数。OT 和 MMD 可单独启用，也可同时启用。
MMD 的对齐 ODE 使用条件 denoiser、关闭 CFG，直接约束条件潜变量分布与
冻结 VAE 后验分布。
OT coupling 不能与 Decoder-aware FM 或 Crystal-MM-FM 同时启用；MMD
可以与这两个新模块相加，具体见后文的兼容矩阵。
下面的命令使用标准 DiT；把 `--denoiser dit` 改为 `--denoiser diverse_dit`
即可测试 DiverseDiT + PA-OTFM。

三种训练模式使用完全独立的开关：

| 模式 | 开关 | 训练内容 |
| --- | --- | --- |
| OT-only | `--ot-coupling` | Flow Matching + OT 配对 |
| MMD-only | `--mmd-alignment` | Flow Matching + 终点 MMD |
| Joint | `--ot-coupling --mmd-alignment` | Flow Matching + OT + MMD |

同一组对比实验应使用相同 VAE、相同基础 DiT checkpoint、相同 batch size
和相同总 epoch 上限。

### 阶段 1：仅 OT-FM

```powershell
python -m ADITcsp.src.train_csp `
  --dataset carbon_24 `
  --stage dit `
  --vae-ckpt "runs/carbon24_hilovae/checkpoints/last.ckpt" `
  --denoiser dit `
  --dit-layers 6 `
  --ot-coupling `
  --epochs 300 `
  --batch-size 32 `
  --device cuda `
  --output-dir "runs/carbon24_hilovae_dit_otfm"
```

### 阶段 2：OT-FM + Terminal MMD

从阶段一的 checkpoint 继续训练。MMD 通过完整 ODE 反传，默认每 4 个
Flow Matching step 执行一次：

```powershell
python -m ADITcsp.src.train_csp `
  --dataset carbon_24 `
  --stage dit `
  --vae-ckpt "runs/carbon24_hilovae/checkpoints/last.ckpt" `
  --denoiser dit `
  --dit-layers 6 `
  --ot-coupling `
  --mmd-alignment `
  --mmd-weight 0.05 `
  --mmd-every 4 `
  --mmd-steps 100 `
  --mmd-batch-size 16 `
  --mmd-kernel-scales "0.5,1,2,4" `
  --resume "runs/carbon24_hilovae_dit_otfm/checkpoints/last.ckpt" `
  --epochs 500 `
  --batch-size 32 `
  --device cuda `
  --output-dir "runs/carbon24_hilovae_dit_otfm_mmd"
```

### 独立模式：仅 Terminal MMD

不启用 OT，只从同一个基础 DiT checkpoint 开始训练 MMD：

```powershell
python -m ADITcsp.src.train_csp `
  --dataset carbon_24 `
  --stage dit `
  --vae-ckpt "runs/carbon24_hilovae/checkpoints/last.ckpt" `
  --denoiser dit `
  --dit-layers 6 `
  --mmd-alignment `
  --mmd-weight 0.05 `
  --mmd-every 4 `
  --mmd-steps 100 `
  --mmd-batch-size 16 `
  --mmd-kernel-scales "0.5,1,2,4" `
  --resume "runs/carbon24_hilovae_standard_dit/checkpoints/last.ckpt" `
  --epochs 500 `
  --batch-size 32 `
  --device cuda `
  --output-dir "runs/carbon24_hilovae_dit_mmd"
```

如果希望从头训练任一模式，删除 `--resume`；如果希望比较三种模式，则应
为三个 run 使用相同的 checkpoint 和总 epoch 上限。

`--mmd-steps` 应尽量与正式推理步数一致；快速调试可临时设为
`10`。完整 ODE 会显著增加显存和训练时间，`--mmd-batch-size` 可独立于
Flow Matching batch 设置。续训时 `--epochs` 是整个训练过程的总上限，
不是追加轮数；例如阶段一训练到 300 epoch，可用 `--epochs 500` 再训练
200 epoch。

OT 只在 batch 内具有相同完整成分且至少有两个样本的 group 上执行；没有
可配对 group 时 `ot_cost=0`。

MMD 输入是 padded latent token 序列。所有样本 mask 相同时，会先展平
有效 token；长度不一致时使用 masked mean，以保持批内可比较。带宽采用
合并样本两两距离中位数的多尺度 RBF 核。

训练 CSV 会记录：

```text
ot_cost        OT 配对后的平均归一化平方代价
mmd_raw_loss   未乘权重的 Terminal MMD
mmd_loss       实际加入总损失的 MMD 项
mmd_bandwidth  median heuristic 得到的基准带宽
```

独立模块测试：

```powershell
python -B tests/test_posterior_alignment.py
```

## Decoder-aware FM 与 Crystal-MM-FM

两个模块是独立开关，默认关闭。关闭时仍使用原来的 Gaussian FM、独立配对和现有
checkpoint 行为。两者都不能与 `--ot-coupling` 同时启用，因为 OT 会重排目标与源
的对应关系，和这里的配对语义冲突。

### Decoder-aware FM

在原有潜变量 Flow Matching Loss 之外，把 DiT 预测的终点潜变量送入冻结的 VAE
Decoder，增加晶胞、周期 RDF 和周期坐标辅助损失：

```text
L = L_FM
    + cell_loss_weight * L_cell
    + geo_loss_weight * L_geo
    + coord_loss_weight * L_coord
```

Decoder 参数冻结，但前向不使用 `torch.no_grad()`，梯度会通过 Decoder 回传到 DiT。
VAE、DiT 网络结构均不改变。
其中 `L_coord` 使用元素约束下的 Hungarian 匹配和周期坐标误差；完整
数学定义见下文“核心数学模型”。

下面的命令用 `--init-ckpt` 只加载修复 bug 后重训的 standard DiT 权重，不恢复
旧 optimizer 和 epoch，并给 50 epoch 的有限微调预算。验证仍有改善时可以延长；
已经平台化时直接停止。`--init-ckpt` 使用严格加载，目标模型必须与 checkpoint
的 VAE/DiT 架构一致；若需要无缝续训并恢复 optimizer，则使用原有的 `--resume`。

先只启用晶胞损失：

```powershell
python src/train_csp.py `
  --dataset carbon_24 `
  --stage dit `
  --vae-ckpt "runs/carbon_24_hilo_vae/checkpoints/last.ckpt" `
  --init-ckpt "runs/carbon24_hilovae_retrain_standard_dit/checkpoints/last.ckpt" `
  --denoiser dit `
  --decoder-aware-fm `
  --cell-loss-weight 0.1 `
  --geo-loss-weight 0 `
  --coord-loss-weight 0 `
  --epochs 50 `
  --batch-size 32 `
  --device cuda `
  --output-dir "runs/carbon24_hilo_da_fm_cell"
```

再启用完整结构辅助损失：

```powershell
python src/train_csp.py `
  --dataset carbon_24 `
  --stage dit `
  --vae-ckpt "runs/carbon_24_hilo_vae/checkpoints/last.ckpt" `
  --init-ckpt "runs/carbon24_hilovae_retrain_standard_dit/checkpoints/last.ckpt" `
  --denoiser dit `
  --decoder-aware-fm `
  --cell-loss-weight 0.1 `
  --geo-loss-weight 0.1 `
  --coord-loss-weight 0.1 `
  --epochs 50 `
  --batch-size 32 `
  --device cuda `
  --output-dir "runs/carbon24_hilo_da_fm_full"
```

训练 CSV 会新增 `train/cell_loss`、`train/geo_loss` 和
`train/coord_loss`。
若要复现加入坐标项之前的 Decoder-aware FM，只保留原有 cell/geo，设置
`--coord-loss-weight 0`。

### Crystal-MM-FM

先用训练集和已训练的 VAE 构建 GMM 源先验。默认 K=8，建议分别尝试
`--num-modes 4`、`8`、`16`：

```powershell
python scripts/build_crystal_gmm.py `
  --dataset carbon_24 `
  --vae-ckpt "runs/carbon_24_hilo_vae/checkpoints/last.ckpt" `
  --num-modes 8 `
  --batch-size 32 `
  --device cuda `
  --output "runs/carbon_24_hilo_mmfm_prior.pt"
```

脚本只读取 train split，保存每个训练晶体的软分配概率、每个模式的 token-level
均值和方差、以及成分到模式比例的映射。输出旁边会生成同名 `.json` 摘要。
成分模式比例默认加入 `--composition-smoothing 1e-3` 的均匀平滑，避免某个
模式在训练成分中概率变成严格零。当前 DiT 直接使用原始后验潜变量，因此
GMM 先验与 DiT 处于同一潜空间；以后若增加潜变量标准化，必须先逆变换再解码。

只用 GMM 源，不做目标模式配对（MM-FM-Prior）：

```powershell
python src/train_csp.py `
  --dataset carbon_24 `
  --stage dit `
  --vae-ckpt "runs/carbon_24_hilo_vae/checkpoints/last.ckpt" `
  --init-ckpt "runs/carbon24_hilovae_retrain_standard_dit/checkpoints/last.ckpt" `
  --denoiser dit `
  --mmfm-prior "runs/carbon_24_hilo_mmfm_prior.pt" `
  --no-mmfm-mode-coupling `
  --epochs 50 `
  --batch-size 32 `
  --device cuda `
  --output-dir "runs/carbon24_hilo_mmfm_prior"
```

启用目标模式配对（MM-FM-Full）：

```powershell
python src/train_csp.py `
  --dataset carbon_24 `
  --stage dit `
  --vae-ckpt "runs/carbon_24_hilo_vae/checkpoints/last.ckpt" `
  --init-ckpt "runs/carbon24_hilovae_retrain_standard_dit/checkpoints/last.ckpt" `
  --denoiser dit `
  --mmfm-prior "runs/carbon_24_hilo_mmfm_prior.pt" `
  --mmfm-mode-coupling `
  --epochs 50 `
  --batch-size 32 `
  --device cuda `
  --output-dir "runs/carbon24_hilo_mmfm_full"
```

推理时默认按成分的模式比例采样。候选数不少于模式数时，加 `--mmfm-balanced`
可以按比例分层分配模式，减少 K=20 中重复落在同一模式的情况：

```powershell
python src/test_csp.py `
  --dataset carbon_24 `
  --ckpt "runs/carbon24_hilo_mmfm_full/checkpoints/last.ckpt" `
  --vae-ckpt "runs/carbon_24_hilo_vae/checkpoints/last.ckpt" `
  --candidates 20 `
  --limit 0 `
  --device cuda `
  --mmfm-balanced `
  --output "runs/carbon24_hilo_mmfm_full_test"
```

先验路径会写入 checkpoint 的 hparams。若服务器路径变化，测试时可用
`--mmfm-prior` 覆盖。

### 两个模块组合

```powershell
python src/train_csp.py `
  --dataset carbon_24 `
  --stage dit `
  --vae-ckpt "runs/carbon_24_hilo_vae/checkpoints/last.ckpt" `
  --init-ckpt "runs/carbon24_hilovae_retrain_standard_dit/checkpoints/last.ckpt" `
  --denoiser dit `
  --decoder-aware-fm `
  --cell-loss-weight 0.1 `
  --geo-loss-weight 0.1 `
  --coord-loss-weight 0.1 `
  --mmfm-prior "runs/carbon_24_hilo_mmfm_prior.pt" `
  --mmfm-mode-coupling `
  --epochs 50 `
  --batch-size 32 `
  --device cuda `
  --output-dir "runs/carbon24_hilo_da_fm_mmfm"
```

如果只想控制成本，先固定同一 VAE、同一训练预算，只跑 Baseline、
DA-FM-Full、MM-FM-Full 三条。确认方向后再拆 Cell-only、MM-FM-Prior
和组合，避免一次展开过多实验。

### 与已有模型的组合

两个新模块只作用于第二阶段（`--stage dit`），不改变 VAE Encoder、
VAE Decoder 或任何去噪器主体的结构。兼容关系如下：

| 已有配置 | 加 Decoder-aware FM | 加 Crystal-MM-FM | 说明 |
| --- | --- | --- | --- |
| 标准 DiT（`--denoiser dit`） | 可以 | 可以 | 默认仍为原始 Gaussian FM |
| CoDeMDT（`--denoiser code_mdt`） | 可以 | 可以 | `mask_loss_weight` 和多样性权重不变 |
| DiverseDiT（`--denoiser diverse_dit`） | 可以 | 可以 | 层间多样性权重不变 |
| OT-only（`--ot-coupling`） | 不可 | 不可 | OT 是替代配对方式，程序会拒绝组合 |
| Terminal MMD（`--mmd-alignment`） | 可以 | 可以 | 损失相加；MM-FM 的 MMD 分支使用成分级 GMM 源 |
| CKA、CSP 测试、重建测试 | 可以 | 可以 | 新模块只增加训练损失/源采样 |
| 第一阶段 VAE 训练 | 不适用 | 不适用 | 新参数只在 `--stage dit` 有效 |

各项损失的实际组合如下。只有显式启用对应开关时，方括号中的项才存在。

标准 DiT：

$$
\mathcal L =
\mathcal L_{\mathrm{FM}}
+\left[\lambda_{\mathrm{cell}}\mathcal L_{\mathrm{cell}}
+\lambda_{\mathrm{geo}}\mathcal L_{\mathrm{geo}}
+\lambda_{\mathrm{coord}}\mathcal L_{\mathrm{coord}}\right]
+\left[w_{\mathrm{mmd}}\mathcal L_{\mathrm{mmd}}\right].
$$

CoDeMDT：

$$
\mathcal L =
\mathcal L_{\mathrm{FM}}
+w_{\mathrm{mask}}\mathcal L_{\mathrm{masked}}
+w_{\mathrm{div}}\mathcal L_{\mathrm{div}}
+\left[\lambda_{\mathrm{cell}}\mathcal L_{\mathrm{cell}}
+\lambda_{\mathrm{geo}}\mathcal L_{\mathrm{geo}}
+\lambda_{\mathrm{coord}}\mathcal L_{\mathrm{coord}}\right]
+\left[w_{\mathrm{mmd}}\mathcal L_{\mathrm{mmd}}\right].
$$

DiverseDiT：

$$
\mathcal L =
\mathcal L_{\mathrm{FM}}
+w_{\mathrm{div}}\mathcal L_{\mathrm{div}}
+\left[\lambda_{\mathrm{cell}}\mathcal L_{\mathrm{cell}}
+\lambda_{\mathrm{geo}}\mathcal L_{\mathrm{geo}}
+\lambda_{\mathrm{coord}}\mathcal L_{\mathrm{coord}}\right]
+\left[w_{\mathrm{mmd}}\mathcal L_{\mathrm{mmd}}\right].
$$

几个关键关系：

- CoDeMDT 的 `w_mask` 仍由 `--mask-loss-weight` 控制，默认仍为 `1.0`。
  Decoder-aware FM 只是在总损失末尾增加 cell/geo/coord 项，不会乘到
  `w_mask` 上，也不会改变 `L_masked` 的定义。
- CoDeMDT 和 DiverseDiT 的 `w_div` 仍由原有的
  `--div-*` / `--diverse-*` 参数决定；新模块不会修改它。
- Crystal-MM-FM 改变的是源潜变量 `z0` 的采样分布和训练配对，
  不增加网络损失项；`L_FM` 的定义仍是原来的 Flow Matching loss。
  CoDeMDT 的普通路径和 masked 路径都使用同一个 GMM 源。
- Decoder-aware FM 的 cell/geo/coord 项只从普通路径的 `pred` 计算；
  CoDeMDT 的 masked 路径仍由原来的 `L_masked` 训练，不会被这些结构项替代。
- MMD 与 MM-FM 同时启用时，MMD 对齐路径使用成分级模式权重；
  `--mmfm-mode-coupling` 的逐晶体 responsibility 主要用于主 FM 路径。
  若要求 MMD 也严格使用逐晶体配对，需要在后续实现中扩展该分支。
- `--mmfm-mode-coupling` 是训练期的模式配对；
  `--mmfm-balanced` 只是推理期的候选分散。训练代码中
  `_prepare_batch` 固定 `balanced=False`，所以把
  `--mmfm-balanced` 写在训练命令里不会改变训练，只会在 checkpoint
  hparams 中记录一个推理默认值。建议只在 `test_csp.py` 中加它。

### 核心数学模型

记晶体 i 的原子数为 `N_i`，成分条件为 `c_i`。冻结 VAE Encoder 给出
token-level 后验：

$$
q_i(z_{in}\mid X_i)=
\mathcal N\left(\mu_{in},\operatorname{diag}(\sigma_{in}^2)\right),
$$

第二阶段的目标 `z1` 是该后验的一次采样。原始 Gaussian FM 使用
`z0 ~ N(0, I)`；Crystal-MM-FM 只把 `z0` 换成对应于成分和结构模式的
混合源，DiT 的主体和 `x1` 预测形式不变：

$$
z_t=(1-t)z_0+t z_1,\qquad
\hat z_1=f_\theta(z_t,t,c).
$$

这里的 `f_theta` 是标准 DiT、CoDeMDT 或 DiverseDiT 的 clean-latent
预测头；`t` 是 Flow Matching 时间，`c` 是成分条件。

#### Crystal-MM-FM 的 GMM

对每个训练晶体构造全局描述符：

$$
h_i=\left[
\bar\mu_i;\ \operatorname{std}(\mu_i);\ \ell_i;\ \operatorname{RDF}_i
\right],
$$

其中

$$
\ell_i=\left[
\log\frac{a_i}{N_i^{1/3}},
\log\frac{b_i}{N_i^{1/3}},
\log\frac{c_i}{N_i^{1/3}},
\cos\alpha_i,\cos\beta_i,\cos\gamma_i
\right],
$$

`RDF_i` 是周期最小镜像距离的归一化直方图。训练集描述符拟合 K 个高斯
成分：

$$
p(h)=\sum_{k=1}^{K}\pi_k
\mathcal N(h;m_k^h,\Sigma_k^h),
$$

代码使用对角协方差 `Sigma_k^h`，`K` 对应 `--num-modes`。

并得到每个晶体的软模式分配：

$$
r_{ik}=
\frac{\pi_k\mathcal N(h_i;m_k^h,\Sigma_k^h)}
{\sum_{j=1}^{K}\pi_j\mathcal N(h_i;m_j^h,\Sigma_j^h)}.
$$

每个模式 k 的 token-level 源分布为
`N(m_k, diag(s_k^2))`，其参数由训练集和 responsibility 加权：

$$
m_{kd}=
\frac{\sum_i\sum_n r_{ik}u_{in}\mu_{ind}}
{\sum_i\sum_n r_{ik}u_{in}},
$$

$$
s_{kd}^2=
\frac{\sum_i\sum_n r_{ik}u_{in}
\left[\sigma_{ind}^2+(\mu_{ind}-m_{kd})^2\right]}
{\sum_i\sum_n r_{ik}u_{in}},
$$

其中 `u_in` 在有效原子 token 上为 1，padding 上为 0。这样 GMM 只在
晶体级别识别模式，而源噪声在原子 token 级别统计；同一晶体的所有有效
token 共享一个模式编号，但每个 token 独立采样。

训练时，普通 MM-FM 按成分模式比例采样模式；启用
`--mmfm-mode-coupling` 后，每个晶体按自己的 `r_i` 采样模式：

$$
k_i\sim\operatorname{Cat}(w_{c_i})\quad\text{或}\quad
k_i\sim\operatorname{Cat}(r_i),
$$

$$
z_{0,in}\sim
\mathcal N(m_{k_i},\operatorname{diag}(s_{k_i}^2)).
$$

成分模式比例由训练晶体聚合后按成分归一化，并默认加入 `1e-3` 的均匀
平滑：

$$
w_{c,k}=\frac{\sum_{i:c_i=c}r_{ik}}
{\sum_{j:c_j=c}\sum_l r_{jl}}.
$$

推理时对成分 `c` 从 `w_c` 采样模式；`--mmfm-balanced` 用按比例分层
分配代替独立抽样，使 K 个候选覆盖更多模式。MM-FM 不改变 DiT 参数或
Flow Matching loss，只改变 `z0` 的条件混合源。

#### Decoder-aware FM

DiT 预测 clean latent 后，冻结 Decoder 把它映射为晶体几何：

$$
\hat z_1=f_\theta(z_t,t,c),\qquad
\left(\hat a,\hat b,\hat c,\hat\alpha,\hat\beta,\hat\gamma,
\hat f\right)=D_\phi(\hat z_1,c).
$$

Decoder 参数冻结，但前向保留计算图，所以梯度可以通过
`D_phi` 回传到 `f_theta`。两个辅助项为：

$$
\mathcal L_{\mathrm{cell}}=
\frac{1}{B}\sum_{b=1}^{B}\left[
\sum_{q\in\{a,b,c\}}
\left|\log\frac{\hat q}{q}\right|
+\sum_{\theta\in\{\alpha,\beta,\gamma\}}
\left|\cos\hat\theta-\cos\theta\right|
+\text{validity penalty}
\right],
$$

$$
g_l(X)=\frac{1}{N}\sum_{i<j}
\exp\left[-\frac{(d_{ij}-\rho_l)^2}{2\sigma^2}\right],
\qquad
\mathcal L_{\mathrm{geo}}=
\frac{1}{B}\sum_{b=1}^{B}\sum_l
\left|g_l(\hat X_b)-g_l(X_b)\right|.
$$

`d_ij` 使用周期最小镜像距离；代码用整数平移搜索，并对零距离和退化晶格
做数值稳定处理。总训练目标中的新增项为

$$
\lambda_{\mathrm{cell}}\mathcal L_{\mathrm{cell}}
+\lambda_{\mathrm{geo}}\mathcal L_{\mathrm{geo}}
+\lambda_{\mathrm{coord}}\mathcal L_{\mathrm{coord}},
$$

默认三个权重都是 `0.1`。这些项只从普通路径的终点 `pred` 计算，且只
看一次 denoiser 预测；正式测试仍使用完整的 100 步 ODE 和 CFG。因此
它优化的是“单步终点解码质量”，不是完整的采样轨迹。

#### 周期坐标监督

设预测分数坐标为 `f_hat_i`，真实分数坐标为 `f_j`，参考晶格为真实晶格
`C`。对整体平移 `s`，先构造元素约束下的周期代价矩阵：

$$
D_{ij}(s)=
\min_{\mathbf n\in\mathcal N}
\left\|(\hat f_i+s-f_j+\mathbf n)C\right\|_2^2,
$$

其中 \(\mathcal N\) 由 `--coord-max-shifts` 控制，默认搜索
`{-1,0,1}^3` 的整数平移。代码从预测/目标的相对分数位移中提取平移
候选，并对每个候选执行 Hungarian 匹配：

$$
s^*=\arg\min_s\min_{\pi\in\Pi_T}
\sum_i D_{i,\pi(i)}(s),
$$

\(\Pi_T\) 只允许元素类型相同的原子互相匹配；对于碳数据，所有 C 原子
之间都允许匹配。平移搜索和 Hungarian 匹配在 `torch.no_grad()` 下完成，
匹配关系固定后才计算可微误差：

$$
\mathcal L_{\mathrm{coord}}=
\frac{1}{B}\sum_{b=1}^{B}\frac{1}{N_b}
\sum_i
\left\|
\operatorname{MI}\left(
\hat f_i+s^*-f_{\pi^*(i)}
\right)C
\right\|_2^2.
$$

`MI` 表示周期最小镜像。这里用真实晶格 `C` 作为距离度量，预测晶格由
`L_cell` 单独监督，避免晶格尺度错误和坐标错误重复计分。损失按每个晶体
的有效原子数归一化，padding 不参与匹配。若预测与目标的原子数不同或元素
约束下不存在完整匹配，该晶体跳过坐标项；Carbon-24 不会触发这种情况。
坐标项公式中的 `B` 是本次参与匹配的晶体数；默认每个训练步抽 8 个。

默认 `--coord-loss-weight 0.1`、`--coord-shift-candidates 4`、
`--coord-max-shifts 1`、`--coord-batch-size 8`。每个训练步随机抽取
8 个晶体做 Hungarian 匹配，cell/RDF 仍对全 batch 计算；把
`--coord-batch-size` 设为 `0` 则对所有晶体匹配。把权重设为 `0` 即可
关闭坐标项。增大 `--coord-shift-candidates` 或缩小
`--coord-batch-size` 会改变匹配覆盖/成本；前者更充分但更慢。
当前实现是受控候选搜索，不是对全部 N² 相对位移穷举；默认候选包括
零平移、均值/中位相对位移和量化后的高频位移，并做一次匹配后修正。

当前实现假设预测和真实晶体使用相同的晶胞基底；晶胞旋转、轴置换或
等价晶胞表示差异不在这一项的规范化范围内。若需要比较这类表示，应先做
晶胞规范化，再计算周期坐标损失。

两个模块正交地作用于同一套 VAE/DiT：MM-FM 改变 `z0` 的条件源，
DA-FM 改变训练损失；关闭对应开关即恢复原始 Gaussian FM。两者的组合
不会修改 CoDeMDT 的 `w_mask` 或任何去噪器的多样性权重。

### 损失项定义与日志

`L_FM` 是普通路径的 Flow Matching loss。当前模型预测的是 clean latent
`z1`（不是速度），代码先计算
`(z1 - pred) / (1 - min(t, 0.9))`，再对有效原子和潜变量维度求均方：

$$
\mathcal L_{\mathrm{FM}} =
\frac{1}{B}\sum_{b=1}^{B}
\frac{1}{N_b D}\sum_{n,d}
\left(
\frac{\hat z_{1,b,n,d}-z_{1,b,n,d}}
{1-\min(t_b,0.9)}
\right)^2 .
$$

其中 `N_b` 是第 b 个晶体的有效原子数，padding 不参与。Crystal-MM-FM
只改变 `z0` 的来源，`L_FM` 的形式不变；`z0` 来自 GMM 模式源，而不是
标准高斯。

`L_masked` 是 CoDeMDT 的掩码路径 Flow Matching loss，形式与 `L_FM`
相同，但 denoiser 使用随机 mask 分支。它只存在于 CoDeMDT：

$$
\mathcal L_{\mathrm{masked}} = \mathrm{FM}(\hat z_1^{\mathrm{mask}}, z_1).
$$

`w_mask` 对应 `--mask-loss-weight`，默认 `1.0`。DA-FM 和 MM-FM 不会
改变它，也不会改变 masked 路径的目标；MM-FM 只会让 masked 路径和普通
路径共享同一个 GMM 源。

`L_div` 是 CoDeMDT 或 DiverseDiT 的潜在表示多样性项。CoDeMDT 默认使用
`lambda_orth=1.0`、`lambda_mi=1.0`、`lambda_disp=0.1`；DiverseDiT
默认三项各 `0.33`。实际加入总损失前还会乘以自适应或固定的 `w_div`：

$$
\mathcal L_{\mathrm{div}} =
\lambda_{\mathrm{orth}}\mathcal L_{\mathrm{orth}}
+\lambda_{\mathrm{mi}}\mathcal L_{\mathrm{mi}}
+\lambda_{\mathrm{disp}}\mathcal L_{\mathrm{disp}}.
$$

日志中的 `div_raw_loss` 是未乘 `w_div` 的原始值，`div_loss` 是实际
加入总损失的加权值，`div_weight` 是当前权重。新模块不会改写这三项，
只是在总损失中额外加入自己的项。

Decoder-aware FM 引入的三个结构项是：

$$
\mathcal L_{\mathrm{cell}} =
\frac{1}{B}\sum_b\left[
\sum_{q\in\{a,b,c\}}\left|\log\frac{\hat q}{q}\right|
+\sum_{\theta\in\{\alpha,\beta,\gamma\}}
\left|\cos\hat\theta-\cos\theta\right|
+\text{validity penalty}
\right],
$$

其中 `validity penalty` 来自代码对负长度和越界角度的约束；预测长度由
Decoder 的 scaled length 乘以 `N^(1/3)` 恢复为实际长度。此处的 `q`
表示晶格长度 `a,b,c`，不是 VAE 的后验分布。

$$
g_l(X)=\frac{1}{N}\sum_{i<j}
\exp\left[-\frac{(d_{ij}-\rho_l)^2}{2\sigma^2}\right],
\qquad
\mathcal L_{\mathrm{geo}}=
\frac{1}{B}\sum_b\sum_l
\left|g_l(\hat X_b)-g_l(X_b)\right|.
$$

`d_ij` 是按整数平移搜索得到的周期最小镜像距离；默认 32 个 `rho` 分箱、
`sigma=0.5`。`L_geo` 是距离分布的聚合误差，不是逐原子 RMS，因此
`L_geo` 很小不保证 StructureMatcher 的 RMS 很小。

`L_coord` 是周期坐标项：先搜索整体平移并执行元素约束下的 Hungarian
匹配，再固定匹配关系，用真实晶格计算可微的周期最小镜像平方距离。
匹配不回传梯度，只有固定匹配后的坐标误差回传。它默认按每个晶体的
有效原子数归一化，因此大晶胞不会因为原子更多而线性放大损失。公式见
上一节的“周期坐标监督”。

三个结构项只在 `--decoder-aware-fm` 打开时加入，权重分别是
`--cell-loss-weight`、`--geo-loss-weight` 和 `--coord-loss-weight`，
默认都为 `0.1`。它们只从
普通路径的 `pred` 计算；CoDeMDT 的 masked 分支仍只使用 `L_masked`。

`L_mmd` 是 Terminal MMD，只在 `--mmd-alignment` 打开时加入；权重为
`--mmd-weight`。MM-FM 开启时，MMD 分支使用成分级 GMM 源，
`--mmfm-mode-coupling` 的逐晶体 responsibility 主要用于主 FM 路径。

`ot_cost` 不是损失项，也不会加入总损失。OT 只改变噪声和后验的配对关系，
日志中的 `ot_cost` 是配对诊断量。

训练日志和 `loss_history.csv` 的字段含义：

| 字段 | 含义 |
| --- | --- |
| `train/loss` | 当前训练配置的总损失，包含下方所有已启用项 |
| `train/x_loss` | 普通路径 Flow Matching loss |
| `train/plain_loss` | CoDeMDT 普通路径 loss（可视为 `x_loss`） |
| `train/m_loss` / `m_x_loss` | CoDeMDT masked 路径 loss |
| `train/div_raw_loss` / `div_loss` / `div_weight` | 原始多样性项 / 加权后实际加入项 / 当前权重 |
| `train/cell_loss` / `geo_loss` / `coord_loss` | 未乘 DA-FM 权重的原始结构项 |
| `train/mmd_loss` / `mmd_raw_loss` / `mmd_bandwidth` | 加权 MMD / 原始 MMD / 带宽 |
| `ot_cost` | OT 配对诊断量，不是 loss |
| `loss_history.csv: train_loss` | 上述总损失 |
| `loss_history.csv: val_loss` | 目前只验证普通 Flow Matching loss |

因此启用 DA-FM、MMD 或 CoDeMDT 的 mask/diversity 项后，不能把
`train_loss` 和 `val_loss` 当作同一目标直接比较；前者是总损失，后者
只是验证集上的 FM loss。

### 用 CoDeMDT 或 DiverseDiT 组合两个模块

先构建与训练 VAE 完全相同的 GMM 先验。下面使用 K=16；若沿用 K=8，
把路径换成 `runs/carbon_24_hilo_mmfm_k8.pt`：

```powershell
python scripts/build_crystal_gmm.py `
  --dataset carbon_24 `
  --vae-ckpt "runs/carbon_24_hilo_vae/checkpoints/last.ckpt" `
  --num-modes 16 `
  --batch-size 32 `
  --device cuda `
  --output "runs/carbon_24_hilo_mmfm_k16.pt"
```

CoDeMDT + DA-FM + MM-FM：

```powershell
python -m ADITcsp.src.train_csp `
  --dataset carbon_24 `
  --stage dit `
  --vae-ckpt "runs/carbon_24_hilo_vae/checkpoints/last.ckpt" `
  --denoiser code_mdt `
  --dit-size s `
  --decode-layer 2 `
  --mask-ratio 0.3 `
  --mask-ratio-range 0.2 `
  --mask-loss-weight 1.0 `
  --decoder-aware-fm `
  --cell-loss-weight 0.1 `
  --geo-loss-weight 0.1 `
  --coord-loss-weight 0.1 `
  --mmfm-prior "runs/carbon_24_hilo_mmfm_k16.pt" `
  --mmfm-mode-coupling `
  --epochs 300 `
  --patience 3 `
  --batch-size 32 `
  --device cuda `
  --output-dir "runs/carbon24_hilo_codemdt_da_fm_mmfm_k16"
```

如果要热启动，增加 `--init-ckpt`，但 checkpoint 必须是 CoDeMDT，
并与当前的 `--dit-size`、`--dit-layers` 和 `--decode-layer` 一致。
`--init-ckpt` 只加载权重；未指定时从随机初始化开始训练。

DiverseDiT + DA-FM + MM-FM：

```powershell
python -m ADITcsp.src.train_csp `
  --dataset carbon_24 `
  --stage dit `
  --vae-ckpt "runs/carbon_24_hilo_vae/checkpoints/last.ckpt" `
  --init-ckpt "runs/carbon24_hilovae_retrain_diversedit_12/checkpoints/last.ckpt" `
  --denoiser diverse_dit `
  --dit-layers 12 `
  --diverse-long-residual `
  --diverse-blocks 6 `
  --diverse-pairs 10 `
  --diverse-loss-weight 1.0 `
  --decoder-aware-fm `
  --cell-loss-weight 0.1 `
  --geo-loss-weight 0.1 `
  --coord-loss-weight 0.1 `
  --mmfm-prior "runs/carbon_24_hilo_mmfm_k16.pt" `
  --mmfm-mode-coupling `
  --epochs 300 `
  --patience 3 `
  --batch-size 32 `
  --device cuda `
  --output-dir "runs/carbon24_hilo_diversedit12_da_fm_mmfm_k16"
```

`--init-ckpt` 的 `denoiser` 类型、层数和长残差设置必须与 checkpoint
一致。标准 DiT 的 checkpoint 不能加载到 `code_mdt` 或 `diverse_dit`；
否则会报 `Unexpected key(s)` 或缺少参数。

### 组合模块的测试

以 CoDeMDT 组合为例；DiverseDiT 只替换 `--ckpt` 为对应的输出目录：

```powershell
python -m ADITcsp.src.test_csp `
  --dataset carbon_24 `
  --ckpt "runs/carbon24_hilo_codemdt_da_fm_mmfm_k16/checkpoints/last.ckpt" `
  --vae-ckpt "runs/carbon_24_hilo_vae/checkpoints/last.ckpt" `
  --candidates 20 `
  --limit 0 `
  --device cuda `
  --mmfm-prior "runs/carbon_24_hilo_mmfm_k16.pt" `
  --mmfm-balanced `
  --output "runs/carbon24_hilo_codemdt_da_fm_mmfm_k16_test"
```

测试时优先检查 `checkpoints/` 下的 best checkpoint，而不是固定使用
`last.ckpt`；`last.ckpt` 可能是验证 loss 变差的后期权重。

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

## 测试 VAE 条件重建上界

`src/test_vae.py` 用真实晶体的编码结果评估冻结 VAE 的重建能力。编码器只运行
一次，随后分别使用后验均值和多次后验采样进行解码；解码器始终接收真实目标
成分。该指标用于判断第二阶段是否已经接近 VAE 潜空间和解码器的表达上界。

```powershell
python src/test_vae.py `
  --dataset carbon_24 `
  --ckpt "runs/carbon_24_vae/checkpoints/last.ckpt" `
  --split test `
  --candidates 20 `
  --limit 0 `
  --device cuda `
  --output "runs/carbon_24_vae_test_reconstruction"
```

输出包括：

```text
runs/<dataset>_vae_test_reconstruction/
  summary.json
  per_target.json
  generated/                  仅在指定 --save-cifs 时创建
```

`mode_reconstruction` 使用后验均值，每个目标只有一个候选。
`posterior_reconstruction_at_k` 从每个目标的后验分布采样 K 次并取最佳匹配。
两项都报告成分正确率、结构合法率、结构匹配率、命中目标数和匹配目标的
归一化 RMSD。未命中的目标不会进入 `mean_best_rms`。

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
--init-ckpt        仅加载模型权重，不恢复优化器和 epoch
--dit-size s|b|l
--dit-layers                    覆盖 --dit-size 的 block 数
--denoiser code_mdt|dit|diverse_dit 默认 code_mdt
--diverse-long-residual         DiverseDiT 长残差，默认开启
--diverse-blocks / --diverse-pairs
--diverse-loss-weight
--diverse-orth-weight / --diverse-mi-weight / --diverse-disp-weight
--ot-coupling                   Minibatch posterior-aligned OT 配对
--mmd-alignment                 Terminal posterior MMD
--mmd-weight / --mmd-every
--mmd-steps / --mmd-batch-size
--mmd-kernel-scales
--decoder-aware-fm             Decoder-aware 晶胞/周期结构辅助损失
--cell-loss-weight             默认 0.1
--geo-loss-weight              默认 0.1
--coord-loss-weight            周期坐标损失权重，默认 0.1
--coord-max-shifts             坐标匹配整数平移半径，默认 1
--coord-shift-candidates       每晶体平移候选上限，默认 4
--coord-batch-size             每步做坐标匹配的晶体数，默认 8；0=全部
--mmfm-prior                   Crystal-MM-FM GMM 先验文件
--mmfm-mode-coupling           目标模式配对；默认关闭
--mmfm-balanced                推理时分层分配 GMM 模式
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
--mmfm-prior        覆盖 checkpoint 中保存的 GMM 先验路径
--mmfm-balanced     覆盖 checkpoint 中的推理分层设置
--cfg               默认 2.0
--steps             默认 100
```

## 已移除的旧入口

Hydra 配置和以下旧入口已不再属于当前支持范围：

- `train_autoencoder`
- `train_diffusion`
- `eval_autoencoder`
- `eval_diffusion`
- `eval_csp`

现在统一使用 `src/train_csp.py` 和 `src/test_csp.py`。
