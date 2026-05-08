# 基于 Qwen3-VL 的证据约束金融财报多模态问答系统

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![Qwen3-VL](https://img.shields.io/badge/Model-Qwen3--VL-purple)](https://github.com/QwenLM/Qwen)

本项目旨在解决金融财报场景下复杂图表、长文本与密集数据交织导致的信息检索与问答难题。基于 **Qwen3-VL** 多模态大模型，我们构建了一个“证据约束”的端到端问答系统。它不仅能回答问题，更能提供精确的页面溯源和计算逻辑，确保金融场景下严苛的“可审计性”。

本项目**未依赖 LangChain 等第三方编排框架**，全链路从零手撕多模态 RAG Pipeline，并针对特定场景对 Qwen3-VL 进行了 LoRA 指令微调与严格多维量化评测，并最终封装为 WebUI 供端侧交互。

---

## 新增/修改内容

1. `financial_vl_system/task2_rag/web_ui.py`
   - Gradio WebUI：上传 PDF、输入问题、自动执行 task2 解析/索引/检索/重排/问答。
   - 展示答案、完整 JSON 树、source pages、运行日志。
   - 对溯源页面画红框；如果后续 parser 输出 `bbox` / `layout_blocks[*].bbox`，会高亮具体区块。

2. `financial_vl_system/common/json_guard.py`
   - 对模型输出做 JSON fence 清理、宽松解析、schema 校验和字段补齐。
   - 统一输出 `json_ok`、`schema_ok`、`schema_errors`，避免“解析失败就完全不可用”。

3. `financial_vl_system/task2_rag/answerer.py`
   - 接入 `json_guard.coerce_answer_json()`。
   - 强制 source_pages 限定在检索到的页面集合内；缺失时回退到 retrieval pages。

4. `financial_vl_system/task1_compare_eval/metrics.py`
   - 增加 `answer_f1`，保留 `answer_em/scale_em/joint_em/json_parse_rate`。

5. `financial_vl_system/task1_compare_eval/run_compare_eval.py`
   - 在 summary 中增加 `base_answer_f1/ft_answer_f1/delta_answer_f1`。

6. `financial_vl_system/requirements.txt`
   - 增加 `gradio>=4.44.0`。

7. `financial_vl_system/examples/run_web_ui.sh`
   - WebUI 一键启动脚本。
---

## ✨ 核心架构

区别于常规的“黑盒 QA”或简单的“图文对话”，本系统的核心设计理念是** “可审计与强约束” **。任务流被严密拆解为三层架构：

### 层 1：文档解析层 (Document Parsing)
* **核心动作**：页面 OCR、表格/图表/段落结构识别、页面级 Chunk 化、生成结构化表示。
* **技术实现**：充分利用 Qwen3-VL 强大的原生 document parsing 和 KIE（关键信息提取）能力，将非结构化的 PDF 财报无损转化为包含丰富视觉语义的结构化多模态切片。

### 层 2：证据检索层 (Evidence Retrieval)
* **核心动作**：从长篇幅多页财报中召回高度相关页，利用 Embedding + Reranker 进行双重精排。
* **技术实现**：采用 `Qwen3-VL-Embedding-4B/8B` 提取多模态特征进行初步粗排（Dense Retrieval）；利用微调后的 Qwen3-VL 作为交叉编码器（Cross-encoder Reranker）对 Top-K 进行细粒度重排，为最终生成层提供高信息密度的“证据包”。

### 层 3：证据生成问答层 (Evidence-Constrained Generation)
* **核心动作**：输出精确答案、输出证据页、输出引用片段/表项、输出计算步骤。
* **规则约束**：强制模型以规范的 JSON 格式输出，同时设定了**拒答机制**——当检索到的证据不足以支撑问题时，系统主动拒答，从根本上遏制金融数值幻觉。

---

## 🛠️ 大模型微调细节 (SFT Details)

针对金融图表理解与推理任务，我们使用 TAT-DQA 数据集构建了高质量的多模态指令微调数据集，并实施了高效参数微调。

* **数据格式转换**：为最大化适配 Qwen3-VL 官方推荐结构，将数据集统一转换为 `messages + images` 的 JSONL 格式。
* **架构认知与参数冻结策略**：
  * **策略**：冻结 ViT (视觉特征提取层) 和 Aligner (模态对齐层)，仅使用 LoRA 对 LLM 的全线性层 (`all-linear`) 进行微调。
  * **动机**：Qwen3-VL 预训练的 ViT 已经具备极强的 OCR 和页面布局理解能力，冻结它可以极大节省显存开销，同时让模型聚焦于学习复杂的“基于图表进行多步数值计算与 JSON 格式化输出”的逻辑。
* **核心超参数**：
  * `tuner_type`: lora (`rank=8`, `alpha=32`)
  * `torch_dtype`: bfloat16 (防溢出)
  * `learning_rate`: 1e-4 / `warmup_ratio`: 0.05
  * `gradient_checkpointing`: true (显存优化)
  * `IMAGE_MAX_TOKEN_NUM`: 1024 (限制单图 token 数，平衡分辨率与上下文长度)
  * 全局 Batch Size: 48 (单卡 bs=1 * 累加=8 * 6卡)

### 📈 训练曲线
<div align="center">
  <img src="./images/training_curve.png" alt="Training Curve" width="60%">
  <p><em>图：SwanLab 记录的 LoRA 微调 Loss 与 Learning Rate 等 曲线（训练平稳收敛，未出现过拟合）</em></p>
</div>

<div align="center">
  <img src="./images/eval_curve.png" alt="Evaluation Curve" width="60%">
  <p><em>图：SwanLab 记录的验证集上的 eval 曲线</em></p>
</div>

---

## 📊 多维量化评测结果 (Evaluation)

金融级系统不能仅看 ROUGE/BLEU。我们在验证集上基于严格的数值与逻辑约束设计了自定义量化指标：
* `answer_em`: 答案精确匹配率 (数值本身正确)
* `scale_em`: 量级精确匹配率 (单位如 thousand/million 判断正确)
* `joint_em`: 联合精确匹配率 (数值与量级**同时正确**才算对)
* `json_parse_rate`: 结构化 JSON 解析成功率
* `answer_f1`: answer 字段的 token-level F1，用于 list/span 类答案的部分匹配诊断；金融数值类题目仍以 `joint_em` 作为主指标。

**Base 模型 vs Fine-tuned 模型效果对比 (示例数据)：**

| 评测指标 | Qwen3-VL-8B (Base) | Qwen3-VL-8B (LoRA SFT) | 绝对提升 |
| :--- | :---: | :---: | :---: |
| **JSON Parse Rate** | 82.5% | **99.2%** | 🚀 +16.7% |
| **Answer EM** | 41.2% | **76.5%** | 🚀 +35.3% |
| **Scale EM** | 58.0% | **88.1%** | 🚀 +30.1% |
| **Joint EM (核心指标)** | 36.5% | **73.8%** | 🚀 +37.3% |


---

## 🖥️ WebUI 系统交互展示

为了提供完整的工程交付体验，本系统基于 Gradio / Streamlit 构建了交互式 Web 界面。
> **特性**：支持用户直接上传 PDF 财报，输入自然语言提问；系统会分屏展示推理结果、完整的 JSON 解析树，并高亮溯源原始财报的对应页面与图表区块。
- 支持上传 PDF 财报；
- 输入自然语言问题；
- 自动跑完整 Task2 pipeline：PDF 解析 → 索引构建 → Dense Retrieval → Rerank → Answer；
- 分屏展示：
  - 推理结果；
  - 完整 JSON 解析树；
  - source_pages 溯源页面；
  - 运行日志；
- 对溯源页画红框高亮。

  当前 doc_parser.py 的 chunk 主要是文本 chunk，没有 bbox 坐标，所以我做成了“页面级高亮 + 兼容未来 bbox 区块高亮”。如果后续 parser 输出 layout_blocks[*].bbox 或 chunks[*].bbox，WebUI 会自动画具体图表/表格区块；现有 parser 的 chunk 结构确实没有 bbox。

启动：
```Bash
cd financial_vl_system
bash examples/run_web_ui.sh
```

<div align="center">
  <img src="./images/webui_demo.png" alt="WebUI Interface Demo" width="80%">
  <p><em>图：基于 Qwen3-VL 的多模态 RAG WebUI 交互界面</em></p>
</div>

---

##  快速实现 (Quick Start)

### 1. 环境准备
```bash
pip install -r requirements.txt
# 核心依赖: transformers>=4.57.0, peft, datasets, accelerate
```
### 2. 模型微调 (SFT)
项目提供了两种微调启动方式：

方式 A: 基于 ms-swift 分布式微调 (推荐)

```Bash
bash swift_sft.sh
```
方式 B: 基于原生 Transformers + PEFT 的单/多卡微调

```Bash
python pure_transformers_qwen3_vl_sft.py \
  --model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
  --train_file ./outputs/tatdqa_train_swift.jsonl \
  --eval_file ./outputs/tatdqa_dev_swift.jsonl \
  --output_dir ./outputs/qwen3_vl_8b_tatdqa_lora \
  --use_lora \
  --bf16 \
  --gradient_checkpointing
```
### 3. Task 1: 模型独立评测 (Compare Evaluation)
运行评测脚本，对比 Base 模型与 LoRA Adapter 模型在测试集上的量化指标差异：
```Bash
python task1_compare_eval/run_compare_eval.py \
  --data_file ./outputs/tatdqa_test_swift.jsonl \
  --base_model Qwen/Qwen3-VL-8B-Instruct \
  --adapter_path ./outputs/qwen3_vl_8b_tatdqa_lora/checkpoint-xxx \
  --output_dir ./outputs/task1_compare_results
```
### 4. Task 2: 端到端多模态 RAG 流水线 (Pipeline)
你可以通过一步指令运行包含解析、索引、检索、重排和问答的全链路 Pipeline：

```Bash
python task2_rag/pipeline.py \
  --reports_dir ./financial_vl_system/apple_financial_rag_sample/reports \
  --questions_file ./financial_vl_system/apple_financial_rag_sample/questions/apple_2023_questions.jsonl \
  --work_dir /path/to/work_dir \
  --parse_model Qwen/Qwen3-VL-8B-Instruct \
  --embed_model Qwen/Qwen3-VL-Embedding-8B \
  --rerank_model Qwen/Qwen3-VL-8B-Instruct \
  --answer_adapter_path ./outputs/qwen3_vl_8b_tatdqa_lora/checkpoint-xxx
```
或者分步运行（对应 task2_rag/ 目录下的 doc_parser.py, build_index.py, dense_retriever.py, reranker.py, answerer.py）。