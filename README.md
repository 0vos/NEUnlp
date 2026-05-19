# Glamdring: MMLU (MCQ) SFT (Full) + LoRA

目标：在 `glamdring/` 下用开源训练框架（优先 LLaMA-Factory）完成多选任务（MMLU）的指令微调（SFT）与 LoRA 微调，并用 **accuracy** 对比微调前后性能，同时可视化训练指标（loss / lr /（可选）grad-norm）并对比显存开销。

你已说明环境：`conda activate glamdring`。

## 0. 安装依赖（数据准备 / 评测 / 监控）

如果你遇到 `OSError: [Errno 28] No space left on device`（常见于根分区 `/` 或 `/tmp` 空间不足导致 pip/conda 解包失败），建议先把临时目录指到大盘（例如 `/mnt/data`）：

```bash
mkdir -p /mnt/data/algo/tmp_glamdring
export TMPDIR=/mnt/data/algo/tmp_glamdring
export TEMP=$TMPDIR
export TMP=$TMPDIR
```

在 conda 环境中：

```bash
cd /home/algo/video_agent_group/qianqian/glamdring
pip install -r requirements.txt
```

GPU + torch 请按你机器/驱动自行安装（建议 CUDA 版本匹配）。

## 1. 安装训练框架：LLaMA-Factory

注意：较新的 LLaMA-Factory 版本通常要求 **Python >= 3.11**。如果你当前环境是 Python 3.10，建议新建环境（名字你可按需调整）：

```bash
conda create -n glamdring311 python=3.11 -y
conda activate glamdring311
```

由于本仓库当前未内置 LLaMA-Factory 代码，这里采用 **clone 到 `third_party/` + editable 安装** 的方式（更可控）：

```bash
cd /home/algo/video_agent_group/qianqian/glamdring
mkdir -p third_party
cd third_party
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
```

安装完成后，你应当能运行 `llamafactory-cli -h`（不同版本命令名可能略有差异；如果你的版本是 `llamafactory-cli`/`llamafactory`/`llama-factory` 之一，按实际为准）。

## 2. 数据准备：HuggingFace `cais/mmlu`

默认把 MMLU 转成 LLaMA-Factory 可用的 Alpaca 风格 JSONL（字段：instruction/input/output）。

```bash
cd /home/algo/video_agent_group/qianqian/glamdring
python -m common.scripts.prepare_mmlu \
  --output_dir data \
  --train_split dev \
  --eval_split test \
  --subjects all
```

产物：
- `data/mmlu_train.jsonl`
- `data/mmlu_eval.jsonl`
- `data/mmlu_meta.json`
- `data/dataset_info.json`（LLaMA-Factory 自定义数据集注册文件，训练配置直接用 `dataset: mmlu_mcq_train` / `eval_dataset: mmlu_mcq_eval`）

快速小跑（只抽样一部分，便于你调超参/验证流程）：

```bash
python -m common.scripts.prepare_mmlu \
  --output_dir data \
  --train_split dev \
  --eval_split test \
  --subjects abstract_algebra,anatomy \
  --max_train 2000 \
  --max_eval 500
```

## 3. 评测：MMLU accuracy（微调前/后通用）

评测脚本采用 **log-likelihood** 方式对 A/B/C/D 四个选项打分并取最大者作为预测，避免“生成文本再解析”不稳定。

### 3.1 评测 base 模型（微调前）

```bash
python -m common.scripts.eval_mmlu_accuracy \
  --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
  --eval_jsonl data/mmlu_eval.jsonl \
  --batch_size 8 \
  --max_samples 2000
```

### 3.2 评测全量 SFT checkpoint（微调后）

```bash
python -m common.scripts.eval_mmlu_accuracy \
  --model_name_or_path outputs/sft_full/checkpoint-XXXX \
  --eval_jsonl data/mmlu_eval.jsonl
```

### 3.3 评测 LoRA adapter（微调后）

```bash
python -m common.scripts.eval_mmlu_accuracy \
  --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
  --lora_adapter_path outputs/sft_lora/adapter \
  --eval_jsonl data/mmlu_eval.jsonl
```

## 4. 训练：Full SFT vs LoRA（分文件夹运行）

### 4.1 Full SFT

见目录：`sft_full/`。

```bash
cd /home/algo/video_agent_group/qianqian/glamdring
bash sft_full/run_full_sft.sh
```

### 4.2 LoRA SFT

见目录：`sft_lora/`。

```bash
cd /home/algo/video_agent_group/qianqian/glamdring
bash sft_lora/run_lora_sft.sh
```

## 7. 超参数消融（顺序跑多份 YAML）

把不同超参数写成多份 YAML（例如复制 `sft_lora/configs/lora_sft.yaml` 改 `learning_rate` 或 `lora_rank`），然后顺序跑并自动评测：

```bash
bash common/experiments/run_sweep.sh \
  --eval_jsonl data/mmlu_eval.jsonl \
  --configs sft_lora/configs/lora_sft.yaml,sft_full/configs/full_sft.yaml
```

汇总结果：

```bash
python -m common.scripts.summarize_metrics --metrics_glob outputs/metrics/*.json
```

## 5. 训练指标可视化（TensorBoard）

LLaMA-Factory / Transformers 默认会把 loss/lr 写入 TensorBoard event 文件（取决于配置 `report_to: tensorboard`）。

```bash
cd /home/algo/video_agent_group/qianqian/glamdring
tensorboard --logdir outputs
```

如果你希望导出成静态图片（方便写报告），可以从 `trainer_state.json` 画曲线（loss/lr/grad-norm 若存在）：

```bash
python -m common.scripts.plot_trainer_state \
  --trainer_state outputs/sft_full/trainer_state.json \
  --out_dir outputs/plots/sft_full
```

## 6. 显存对比（Full vs LoRA）

我们提供一个简单的显存轮询脚本写 CSV：

```bash
python -m common.scripts.monitor_gpu --out_csv outputs/gpu_full.csv --interval_sec 1
```

训练前在另一个终端启动（或用 `nohup`），训练完成后对比 `gpu_full.csv` vs `gpu_lora.csv` 的峰值 `memory.used`。

## 7. 超参数消融

你可以在 `sft_full/configs/` 与 `sft_lora/configs/` 下复制 YAML 改参数。
建议一次只动一个维度，并在同一 `data/mmlu_eval.jsonl` 上评测，结果写入 `outputs/metrics/*.json` 方便汇总。

## HF Token（可选）

如果你要切到 Llama3 等 gated 模型，需要先：

```bash
huggingface-cli login
```
