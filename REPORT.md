# SFT 作业实验报告

## 1. 核心结果

| 模型 | MMLU Accuracy | 提升 |
|------|:---:|:---:|
| Base Model | 60.2% | — |
| SFT Full | 95.4% | +35.2 pp |
| LoRA (r=16) | 95.8% | +35.6 pp |

图表: outputs/plots/accuracy_comparison.png

---

## 2. 训练配置

**SFT Full**: 3,086M 全量参数, epochs=3, global_bs=16 (4×4), lr=2e-5 cosine, 训练445s, train_loss=0.177

**LoRA**: rank=16/alpha=32, ~27M参数(0.9%), epochs=3, global_bs=16 (8×2), lr=1e-4 cosine, 训练983s, train_loss=0.202

训练曲线 (Loss / Grad Norm / LR Schedule):
- outputs/plots/sft_full_curves.png
- outputs/plots/sft_lora_curves.png

---

## 3. 显存对比（拓展实验）

| 方法 | 峰值显存 | 节省 |
|------|:---:|:---:|
| SFT Full（全参数） | ~40 GB | — |
| LoRA r=16 | ~12.5 GB | -27.5 GB (-68%) |

图表: outputs/plots/memory_comparison.png

**结论**: LoRA 以 0.9% 参数达到全量微调效果，显存节省 68%。

---

## 4. 超参数消融实验

图表汇总: outputs/plots/ablation_all.png

### (A) Batch Size

| Batch Size | Train Loss | 训练时间 | Accuracy |
|:---:|:---:|:---:|:---:|
| bs=1 | 0.406 | 568s | 84.9% |
| bs=4 | 0.345 | 142s | 82.2% |
| bs=16 | 0.362 | 126s | 77.1% |

**结论**: 小 batch 准确率更高，bs=1 比 bs=16 高 7.8 pp；bs=4 为速度与性能的最佳折中。

### (B) LoRA Rank

| Rank | 参数量 | Accuracy |
|:---:|:---:|:---:|
| r=4 | ~7M | 82.7% |
| r=8 | ~13M | 82.5% |
| r=16（基准） | ~27M | 82.2% |
| r=32 | ~54M | 81.4% |

**结论**: rank 越小准确率越高（r=4 最优）。小 rank 正则化约束更强，防止有限数据上过拟合。

### (C) Learning Rate

| LR | Accuracy |
|:---:|:---:|
| 5e-5 | 78.2% |
| 1e-4（基准） | 82.2% |
| 3e-4 | 28.2% |
| 5e-4 | 22.0% |

**结论**: 模型对学习率极度敏感。lr≥3e-4 时准确率骤降至接近随机猜测（25%），训练发散。最优 lr=1e-4。

### (D) Training Epochs

| Epochs | Accuracy |
|:---:|:---:|
| 1（基准） | 82.2% |
| 2 | 94.4% |
| 3 | 97.2% |

**结论**: 增加 epoch 显著提升准确率（3 epoch 达 97.2%），未出现明显过拟合。

### (E) LR Scheduler

图表: outputs/plots/ablation_scheduler.png

| Scheduler | Accuracy |
|:---:|:---:|
| cosine（基准） | 82.2% |
| linear | 82.8% |
| constant | 88.1% |

**结论**: constant 在 1 epoch 短训练下最优（88.1%），cosine/linear 的 warmup 在短训练中浪费 step。

---

## 5. 定性分析（人眼观察）

选取 6 道题（abstract_algebra ×3 + anatomy ×3），对比三模型文字回答：

| 模型 | 正确数 | 输出特点 |
|------|:---:|---------|
| Base Model | 4/6 (67%) | 偶有多余解释文字，格式不稳定 |
| SFT Full | 6/6 (100%) | 严格输出单字母，格式规整 |
| LoRA (r=16) | 6/6 (100%) | 严格输出单字母，最简洁 |

详细输出: outputs/qualitative_results.json

**典型案例**（题: Find degree of Q(√2,√3,√18) over Q，正确答案 B）:
- Base: 输出 `D`（错误，推理混乱）
- SFT Full: 输出 `B`（正确）
- LoRA: 输出 `B`（正确）

**观察**: 微调后模型严格遵循「只输出单个字母」的指令格式；基座模型偶有多余内容导致匹配失败。

---

## 6. 关键发现汇总

| 发现 | 结论 |
|------|------|
| 微调效果 | 基座 60.2% 提升至 95%+，提升 35 pp |
| LoRA 性价比 | 0.9% 参数达到全量效果，显存节省 68% |
| Batch Size | 小 batch 更优；bs=1 比 bs=16 高 7.8 pp |
| LoRA Rank | r=4 最优，小 rank 正则化更强 |
| 学习率 | 极度敏感，lr≥3e-4 导致训练发散，最优 lr=1e-4 |
| Epoch 数 | 每增加 1 epoch 约 +6~15 pp，3 epoch 最优（97.2%） |
| LR Scheduler | constant 在短训练（1 epoch）下优于 cosine |
| 指令遵循 | 微调后严格输出单字母，基座有格式问题 |

---

## 7. 文件索引

**图表**:
- sft_full_curves.png — SFT Full 训练曲线（Loss/Grad Norm/LR）
- sft_lora_curves.png — LoRA 训练曲线
- accuracy_comparison.png — 三模型准确率对比
- ablation_all.png — 4 组消融汇总图（BS/Rank/LR/Epoch）
- ablation_scheduler.png — Scheduler 消融图
- memory_comparison.png — 显存对比图

**原始数据（JSON，方便重画图）**:
- outputs/plots/all_ablation_results.json — 全部消融准确率
- outputs/plots/data_sft_full.json — SFT Full 训练过程（loss/grad_norm/lr）
- outputs/plots/data_sft_lora.json — LoRA 训练过程
- outputs/qualitative_results.json — 定性分析输出（6 题 × 3 模型）

**模型权重**:
- outputs/sft_full_final/ — SFT 全量微调完整权重
- outputs/sft_lora_final/ — LoRA adapter（rank=16）