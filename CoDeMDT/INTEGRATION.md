# CoDeMDT 与 HiLo 接入 ADITcsp 说明

两个模块已经接入主训练流程，不再需要手工复制源码或修改 `training_step`。
默认架构为：

```text
crystal batch
  -> HiLo 编码器
  -> 8 维 VAE 潜变量
  -> CoDeMDT 普通路径 + 掩码路径
  -> flow-matching 去噪
  -> 标准注意力 VAE 解码器
  -> atom types / fractional coordinates / lattice
```

## 1. 文件位置

主流程使用的规范副本：

```text
src/models/components/hiloattention.py
src/models/denoisers/code_mdt.py
```

`CoDeMDT/` 保留原始参考实现、单测与集成说明。主流程不直接依赖这个目录，
因此移动或删除参考目录不会破坏模型导入。

## 2. HiLo 接入

`TransformerEncoder` 和 `TransformerDecoder` 都新增了
`attention_type="standard"|"hi_lo"` 及 HiLo 参数。VAE 构造时：

- 编码器按 `--vae-attention hi_lo` 使用 `HiLoTransformerEncoder`；
- 编码器先把 PyG 的 `frac_coords`、`cell` 按 `batch` 转成稠密张量；
- 周期几何、最小镜像距离和边特征每次前向只构建一次，供所有编码层复用；
- 输出重新按有效 token 展平，保持原有 VAE 契约 `(num_nodes, d_model)`；
- padding token 在注意力与 FFN 后都被清零。

解码器默认仍为标准自注意力。原因是 CSP 推理时只有目标成分，没有目标分数
坐标和晶胞；这些几何量是 HiLo 的必需输入，若在解码器训练时使用真值、
推理时另造几何，会造成训练/推理条件不一致。`TransformerDecoder` 仍保留
`hi_lo` 能力，可用于输入几何已知的纯重建消融，但不属于默认生成路径。

主要开关：

```text
--vae-attention hi_lo|standard
--hilo-low-freq-heads 2
--hilo-cutoff 5.0
--hilo-max-shifts 1
--hilo-num-freqs 4
--hilo-pool-type grid|chunk|none
--hilo-pool-grid-size 3
--hilo-pool-chunk-size 4
```

8 个头中默认 2 个为 Lo-Fi 全局头，6 个为 Hi-Fi 周期局部头。

## 3. CoDeMDT 接入

`src/train_csp.py` 新增 `--denoiser code_mdt|dit`，默认 `code_mdt`。
CoDeMDT 保留 MDTv2 式结构：

```text
en_in  -> en_out（U-Net skip） -> SI -> de（noisy-latent skip）
```

默认 `--decode-layer 2`，剩余 block 必须均分给 `en_in` 和 `en_out`；
SI 另占一个 block。随机掩码只作用于训练，推理走普通路径。

`LatentDiffusionLitModule.training_step` 现在执行：

```text
L = L_diff(plain)
  + mask_loss_weight * L_diff(masked)
  + w_div * L_div
```

掩码路径位于多样性项之前，保证 `diversity_term()` 能读取同一次前向缓存的
编码器层表示。`w_div` 是 detached 的 Python 标量，只有多样性损失本身进入
梯度图。验证和采样不执行掩码路径，也不计算多样性损失。

多样性损失只作用于编码器，不在解码器上增加约束：

```text
L_div = lambda_orth * L_orth
      + lambda_mi   * L_MI
      + lambda_disp * L_disp
```

默认权重：

```text
adaptive: raw = 0.02 * L_diff / L_div
fixed:    raw = div_weight
w_div = clamp(raw, [0.1, 0.5]) * warmup_ramp
```

## 4. 训练命令

默认使用 HiLo VAE 与 CoDeMDT：

```powershell
python src/train_csp.py `
  --dataset carbon_24 `
  --stage vae `
  --vae-attention hi_lo `
  --output-dir runs/carbon_24_vae

python src/train_csp.py `
  --dataset carbon_24 `
  --stage dit `
  --vae-ckpt runs/carbon_24_vae/checkpoints/last.ckpt `
  --denoiser code_mdt `
  --mask-ratio 0.3 `
  --mask-ratio-range 0.2 `
  --mask-loss-weight 1.0 `
  --decode-layer 2 `
  --div-weight 0.2 `
  --div-weight-min 0.1 `
  --div-weight-max 0.5 `
  --output-dir runs/carbon_24_codemdt
```

原始基线：

```powershell
python src/train_csp.py --stage vae --vae-attention standard ...
python src/train_csp.py --stage dit --denoiser dit ...
```

多样性消融：

```text
--div-fixed                 固定权重而非自适应
--lambda-orth 0             关闭正交项
--lambda-mi 0               关闭互信息相似度项
--lambda-disp 0             关闭维度分散项
--mask-loss-weight 0        去掉掩码重建项
```

`mask-loss-weight=0` 时仍会运行掩码路径，以便在启用多样性约束时构造其编码器
表示；如果同时关闭全部多样性权重，该路径对总目标没有贡献。

## 5. 日志与 CKA

训练 CSV 会新增或包含：

```text
plain_loss
m_loss
m_x_loss
div_loss
loss_orth
loss_mi
loss_disp
div_weight
```

验证指标仍为普通去噪路径的 `loss` 和 `x_loss`。CKA 对 CoDeMDT 捕获
embedding、编码器 block、解码器 block 和输出；SI 不在普通推理路径上，
因此不注册 hook。

## 6. checkpoint 兼容性

- 旧标准 VAE checkpoint 可以继续作为 CoDeMDT 阶段二的输入。
- 新的 HiLo VAE 不能加载到旧 `TransformerEncoder` 结构，反之亦然。
- 新 CoDeMDT 与旧 `DiT` 的 block 布局和参数不同，不能互相加载权重。
- 恢复旧模型时需使用 `--vae-attention standard --denoiser dit`，并保证
  checkpoint 的 VAE 结构一致。

## 7. 验证

CPU 集成测试覆盖：

- 变长晶体 batch 的 HiLo 编码与展平契约；
- CoDeMDT padding 隔离和掩码多样性损失；
- 真实 VAE checkpoint 加载；
- 普通损失、掩码损失、多样性项的联合反传；
- 两步 CFG 采样与解码输出有限性。

执行：

```powershell
conda activate adit
python -m py_compile src/train_csp.py src/models/ldm_module.py
python -B -c "import runpy; ns=runpy.run_path('tests/test_integrated_models.py'); [ns[n]() for n in sorted(ns) if n.startswith('test_')]"
python -B CoDeMDT/test_code_mdt.py
```

如果有可用的真实 VAE checkpoint，也可进一步运行：

```powershell
python -B CoDeMDT/integration_check.py --vae-ckpt "<...>/vae/checkpoints/last.ckpt"
```

该脚本会使用真实 carbon_24 batch 执行两路训练损失、多样性反传、CFG 采样和
CKA hook，并打印 `INTEGRATION_OK`。
