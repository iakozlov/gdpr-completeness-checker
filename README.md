# DPA Completeness Checker

A system for evaluating the completeness of Data Processing Agreements (DPAs) against GDPR requirements using Large Language Models (LLMs) and Answer Set Programming (ASP).

## Overview

This project implements an automated pipeline for analyzing Data Processing Agreements (DPAs) to determine if they fulfill the requirements specified in the GDPR. The system combines:

1. LLM-based semantic analysis of DPA text
2. Deontic logic representation of GDPR requirements
3. Answer Set Programming for completeness evaluation

The completeness checker determines whether a DPA satisfies all required GDPR obligations, providing detailed analysis on which requirements are met and which are missing.

## System Architecture

The system works through the following pipeline:

1. **Requirement Translation**: Converts GDPR requirements from natural language to formal deontic logic representations
2. **LP File Generation**: Extracts facts from DPA segments and generates logic programming files
3. **Deolingo Solver**: Runs the symbolic solver to evaluate DPA segments against requirements
4. **Completeness Evaluation**: Aggregates results to determine overall DPA completeness

## Installation

### Prerequisites

- Python 3.8+
- Deolingo solver
- Choice of LLM:
  - Llama-based LLM (recommended: Meta-Llama-3.1-8B-Instruct)
  - GPT-4o via OpenAI API (alternative implementation)

### Setup

1. Clone the repository:
   ```
   git clone <repository-url>
   cd dpa-completeness-checker
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv dpa_env
   source dpa_env/bin/activate  # On Windows: dpa_env\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Choose your LLM implementation:
   
   **Option 1: Local Llama model**
   - Download and place the LLM model in the `models/` directory
   - Recommended: `models/Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf`
   
   **Option 2: GPT-4o via API**
   - Set your OpenAI API key as an environment variable:
     ```
     export OPENAI_API_KEY=your-api-key-here
     ```

## Usage

The system can be run using one of the main shell scripts that orchestrate the entire pipeline:

```bash
# For Llama model implementation
bash run_dpa_completeness_ollama.sh

# For GPT-4o API implementation
bash run_dpa_completeness_gpt4o.sh
```

This will display a menu allowing you to run individual steps or the entire pipeline.

### Command Line Options

The script accepts the following parameters:

- `--req_ids`: Comma-separated list of requirement IDs to process (default: "all")
- `--max_segments`: Maximum number of segments to process (default: 0, meaning all)
- `--target_dpa`: Target DPA to evaluate (default: "Online 1")

Example:
```bash
bash run_dpa_completeness_ollama.sh --req_ids 6,8,12 --max_segments 50 --target_dpa "Online 2"
```

### Pipeline Steps

The system can run any of these steps individually:

1. **Translate requirements to deontic logic**:
   
   Using Llama:
   ```
   python translate_requirements.py --requirements data/requirements/ground_truth_requirements.txt --model models/Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf --output results/requirements_deontic.json
   ```
   
   Using GPT-4o:
   ```
   python translate_requirements.py --requirements data/requirements/ground_truth_requirements.txt --model gpt-4o --output results/gpt4o_experiment/requirements_deontic.json
   ```

2. **Generate LP files**:
   
   Using Llama:
   ```
   python generate_lp_files.py --requirements results/requirements_deontic.json --dpa data/train_set.csv --model models/Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf --output results/lp_files --target_dpa "Online 1" --req_ids 6 --max_segments 30
   ```
   
   Using GPT-4o:
   ```
   python generate_lp_files.py --requirements results/gpt4o_experiment/requirements_deontic.json --dpa data/train_set.csv --model gpt-4o --output results/gpt4o_experiment/lp_files --target_dpa "Online 1" --req_ids 6 --max_segments 30
   ```

3. **Run Deolingo solver** (handled by the shell script)

4. **Evaluate completeness**:
   ```
   python evaluate_completeness.py --results results/deolingo_results.txt --dpa data/train_set.csv --output results/evaluation_results.json --target_dpa "Online 1" --req_ids 6
   ```

## Key Components

- **translate_requirements.py**: Converts GDPR requirements to deontic logic format
- **generate_lp_files.py**: Extracts facts from DPA segments and creates LP files
 - **run_dpa_completeness_ollama.sh**: Master script to orchestrate the entire pipeline using Ollama
- **evaluate_completeness.py**: Evaluates completeness of DPAs against requirements

## Configuration

The system uses various configuration files:

- LLM configuration: `config/llm_config.py`
- Data paths: Defined in the scripts or passed as command-line arguments
- Requirements file: `data/requirements/ground_truth_requirements.txt`

## Notes on Requirements

In the requirements file (`ground_truth_requirements.txt`), the requirements are indexed as follows:
- R5 is the first requirement
- R6 is the second requirement
- And so on...

The evaluation results will reference these requirement IDs.

## Output

The system generates several output files:

- `results/requirements_deontic.json`: Deontic logic translations of requirements
- `results/lp_files/`: Directory containing the LP files for each segment and requirement
- `results/deolingo_results.txt`: Raw output from the Deolingo solver
- `results/evaluation_results.json`: Final evaluation results with completeness assessment

The evaluation results show which requirements are satisfied, which are missing, and provide segment-level details about requirement satisfaction.

## Comparing Baseline and Symbolic Approaches

After running both the baseline pipeline (`run_baseline_evaluation.sh`) and the symbolic pipeline (`run_dpa_completeness_ollama.sh` with predefined requirements), you can generate a detailed comparison of their predictions using `compare_experiments.py`:

```bash
python compare_experiments.py \
  --baseline_dir results/my_experiment/baseline \
  --deolingo_results results/my_experiment/symbolic/deolingo_results.txt \
  --lp_root results/my_experiment/symbolic/lp_files \
  --output results/my_experiment/comparison.json
```

You can also run `run_full_experiment.sh` to execute both pipelines and generate
the comparison in one step.

The output lists segments where the baseline classified a requirement as satisfied while the symbolic approach did not, and vice versa. For each case the JSON includes the ground truth label, the baseline prediction, the symbolic prediction, and the facts extracted by the LLM for the symbolic method.
