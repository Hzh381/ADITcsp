# 预训练权重两阶段测试

## 第一阶段

使用 `data/checkpoints/vae.ckpt`，对固定测试集中 100 个 MP20 结构做
编码/解码。后验使用 deterministic posterior mean。RMSE 和 MAE 是配对重构
误差，坐标单位见字段名。

注意：预训练 MP20 的损失配置中 `loss_pos=0`，因此原始 Cartesian `pos`
输出头没有被直接监督。`cartesian_rmse_angstrom` 和
`cartesian_mae_angstrom` 只保留作诊断；结构重构质量应主要依据分数坐标、
晶格长度/角度误差、有效率和匹配率。

## 第二阶段

使用 `data/checkpoints/ldm.ckpt` 和 `data/checkpoints/vae.ckpt`。对每个测试
结构，以对应的原子数作为条件，从标准高斯先验开始执行 250 步 CFG flow
matching，再用冻结 VAE 解码，共生成 100 个结构。

第二阶段是无条件生成，不输入真实坐标、晶格或原子类型，因此生成结构与
测试结构没有一一对应的原子映射。`stage2_summary.json` 中的配对 RMSE/MAE
被明确保留为 `null`。其中 Match rate 是相对对应测试参考结构的命中率，
只用于描述性比较，不是重构准确率。

## 与预训练划分的重叠

这 100 条来自本实验重新冻结的 MP20 split，不等同于公开 checkpoint 原始
训练脚本中的 test split。按 MP20 原始顺序统计，100 条中有 57 条位于公开
模型的训练区间、21 条位于验证区间、22 条位于其测试区间。因此主结果应
称为“固定100样本评估”，不能称为严格独立 held-out 测试。

`pretrained_heldout_22_summary.json` 仅对落入公开模型原始 test 区间的 22
条结构单独汇总，作为严格 held-out 的补充参考；样本数较少，不替代100条
主结果。

## 文件

- `summary.json`：两阶段汇总。
- `stage1_summary.json`：VAE 配对重构汇总。
- `stage1_reconstruction_samples.csv`：100 个测试结构的逐样本重构指标。
- `stage1_predictions.npz`：预测和真实数组及 atom offsets。
- `stage1_reconstruction_cifs/`：VAE 重构 CIF。
- `ground_truth_cifs/`：测试集真实 CIF。
- `stage2_summary.json`：LDM 生成汇总。
- `stage2_generated_samples.csv`：100 个生成结构的逐样本有效性/参考匹配。
- `stage2_generated_predictions.npz`：生成结构数组及 atom offsets。
- `stage2_generated_cifs/`：生成结构 CIF。
- `pretrained_heldout_22_summary.json`：公开 checkpoint 原始测试区间的
  22 条结构补充汇总。
