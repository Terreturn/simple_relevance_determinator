# Paper Relevance Evaluation Module

This repository implements a **lightweight LLM-based relevance evaluation pipeline** for judging how relevant academic papers are to a given research query.  
The system is modular, reproducible, and designed for research and experimentation.

---

## Project Structure

```text
Project_paper_finder/
│
├── config.py                # Global configuration & environment variables
├── LLM_calls.py             # LLM client (DeepSeek / OpenAI compatible)
├── text_cutting.py          # Text truncation / chunking utilities
│
├── relevance_model.py       # Core data models
├── relevance_prompt.py      # Prompt templates for relevance judgement
├── relevance_judge.py       # Main relevance judgement logic
│
├── relevant_snippets.py     # Relevant snippet extraction utilities
├── eval_utils.py            # Evaluation & threshold calibration
├── tracing.py               # Logging & tracing
│
├── main.py                  # Entry point
└── README.md
```

## Module Overview

### 1. Basic Settings
- **config.py**  
  Defines global configuration parameters, including:
  - LLM model name  
  - concurrency level  
  - retry count  
  - maximum input length  

- **LLM_calls.py**  
  Implements the LLM API client abstraction:
  - Handles communication with the LLM (e.g. DeepSeek / OpenAI-compatible APIs)
  - Manages retries and error handling
  - Returns structured, parsed outputs

- **text_cutting.py**  
  Provides text preprocessing utilities:
  - Truncates or splits long documents
  - Prevents token or character limit overflow during LLM calls

---

### 2. Relevance Model
- **relevance_model.py**  
  Defines the core data structures used throughout the system:
  - `Document`
  - `RelevanceCriterion`
  - `RelevanceCriterionJudgement`
  - `RelevanceJudgement`

---

### 3. Relevance Prompt
- **relevance_prompt.py**  
  Defines structured prompt templates for relevance evaluation:
  - Specifies relevance labels
  - Enforces JSON-formatted output
  - Requests short relevance summaries for interpretability

---

### 4. Relevance Judging
- **relevance_judge.py**  
  Implements the main relevance evaluation logic:
  - Calls the LLM for each document
  - Aggregates criterion-level judgements using weighted scoring
  - Produces interpretable relevance levels (0–3)

---

### 5. Utilities
- **relevant_snippets.py**  
  Extracts and aligns relevant text snippets returned by the LLM with the original document content.

- **eval_utils.py**  
  Provides evaluation and calibration utilities:
  - Precision / Recall
  - AUROC
  - Precision@K

- **tracing.py**  
  Sets up logging and tracing:
  - Records document-level results
  - Tracks errors such as API failures or parsing issues

---

### 6. Main Execution
- **main.py**  
  Serves as the entry point of the system:
  - Orchestrates the full relevance evaluation pipeline
  - Loads documents and relevance criteria
  - Supports JSONL input/output
  - Optionally runs evaluation and threshold calibration

## Environment Setup

It is recommended to use Conda to create an isolated environment.

```bash
conda create -n trec python=3.9
conda activate trec

Set the API key for the LLM provider (DeepSeek example).

```bash
export DEEPSEEK_API_KEY="your_api_key_here"
```

## Run the system
### Basic demo

```bash
python main.py --query "graph neural networks for traffic prediction"

### Save Results to JSONL
Write relevance evaluation results to a JSONL file.
```

```bash
mkdir -p out
python main.py \
  --query "graph neural networks for traffic prediction" \
  --out-jsonl out/results.jsonl
```

## Output Format
### Result Structure
Each document produces a structured relevance result.

```bash
- relevance_score (0–1)
- relevance_level (0–3)
- relevance_summary
- criterion-level judgements
- optional debug information
```
