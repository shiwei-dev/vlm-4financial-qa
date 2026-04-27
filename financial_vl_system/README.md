# Financial VL System

A modular project for:
1. Comparing base Qwen3-VL-4B vs your fine-tuned best checkpoint.
2. Building a retrieval + QA pipeline over financial report PDFs.
3. A WebUI demo.
## Directory

- `common/`: shared utilities
- `task1_compare_eval/`: base vs fine-tuned evaluation
- `task2_rag/`: parser, dense retrieval, reranking, QA pipeline
- `examples/`: example shell scripts

## Task 1: Compare base vs fine-tuned checkpoint

From this project root:

```bash
python task1_compare_eval/run_compare_eval.py \
  --data_file ./vlm/outputs/tatdqa_test_swift.jsonl \
  --base_model Qwen/Qwen3-VL-4B-Instruct \
  --adapter_path ./vlm/outputs/qwen3_vl_4b_tatdqa_lora/v2-20260416-152316/checkpoint-1100 \
  --output_dir ./vlm/outputs/task1_compare_results \
  --dtype bf16 \
  --device cuda
```

Outputs:
- `predictions_base.jsonl`
- `predictions_ft.jsonl`
- `evaluation_metrics.json`
- `comparison_analysis.csv`
- `comparison_analysis.jsonl`
- `result_analysis_summary.json`

## Task 2: Retrieval + QA pipeline

### Step A: Parse reports into structured pages

```bash
python task2_rag/doc_parser.py \
  --input_dir /path/to/report_pdfs \
  --output_dir /path/to/work/parsed \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --device cuda \
  --dtype bf16
```

### Step B: Build dense page index

```bash
python task2_rag/build_index.py \
  --parsed_pages /path/to/work/parsed/parsed_pages.jsonl \
  --output_dir /path/to/work/index \
  --embedding_model Qwen/Qwen3-VL-Embedding-4B
```

### Step C: Retrieve top-k pages

```bash
python task2_rag/dense_retriever.py \
  --index_dir /path/to/work/index \
  --questions_file ./vlm/outputs/tatdqa_test_swift.jsonl \
  --output_file /path/to/work/retrieval/retrieved_topk.jsonl \
  --embedding_model Qwen/Qwen3-VL-Embedding-4B \
  --top_k 10
```

### Step D: Rerank with multimodal Qwen3-VL scorer

```bash
python task2_rag/reranker.py \
  --retrieval_file /path/to/work/retrieval/retrieved_topk.jsonl \
  --output_file /path/to/work/retrieval/reranked_topk.jsonl \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --device cuda \
  --dtype bf16 \
  --top_k 10 \
  --keep_k 5
```

### Step E: Answer from reranked top-k pages

```bash
python task2_rag/answerer.py \
  --retrieval_file /path/to/work/retrieval/reranked_topk.jsonl \
  --output_file /path/to/work/answers/qa_outputs.jsonl \
  --base_model Qwen/Qwen3-VL-4B-Instruct \
  --adapter_path ./vlm/outputs/qwen3_vl_4b_tatdqa_lora/v2-20260416-152316/checkpoint-1100 \
  --device cuda \
  --dtype bf16 \
  --top_k 5
```

### One-command pipeline

```bash
python task2_rag/pipeline.py \
  --reports_dir /path/to/report_pdfs \
  --questions_file ./vlm/outputs/tatdqa_test_swift.jsonl \
  --work_dir /path/to/work_dir \
  --parse_model Qwen/Qwen3-VL-4B-Instruct \
  --embed_model Qwen/Qwen3-VL-Embedding-4B \
  --rerank_model Qwen/Qwen3-VL-4B-Instruct \
  --answer_base_model Qwen/Qwen3-VL-4B-Instruct \
  --answer_adapter_path ./vlm/outputs/qwen3_vl_4b_tatdqa_lora/v2-20260416-152316/checkpoint-1100 \
  --device cuda \
  --dtype bf16
```

## Extra feature

Evaluate final QA outputs:

```bash
python task2_rag/evaluate_qa.py \
  --gold_file ./vlm/outputs/tatdqa_test_swift.jsonl \
  --pred_file /path/to/work_dir/answers/qa_outputs.jsonl \
  --output_file /path/to/work_dir/answers/qa_metrics.json
```

## Task 3: WebUI demo
