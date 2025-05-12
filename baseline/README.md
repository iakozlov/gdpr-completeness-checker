# Baseline DPA Completeness Checker

This is a simplified baseline implementation of the DPA completeness checker that uses direct LLM comparison to evaluate requirement satisfaction.

## Overview

The baseline implementation:
1. Takes a DPA and a set of requirements as input
2. For each requirement-segment pair, uses an LLM to determine if the segment satisfies the requirement
3. Computes completeness metrics based on the satisfaction results
4. Outputs detailed results in JSON format

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have the LLM model file in the correct location (default: `../models/Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf`)

## Usage

Run the baseline checker using the provided shell script:

```bash
./run_baseline.sh [options]
```

Available options:
- `--req_ids`: Comma-separated list of requirement IDs to evaluate (default: "1,2,3,4,5,6")
- `--max_segments`: Maximum number of segments to process (default: 20)
- `--target_dpa`: Name of the DPA to evaluate (default: "Online 1")

Example:
```bash
./run_baseline.sh --req_ids "1,2,3" --max_segments 10 --target_dpa "Online 1"
```

## Output

The script generates a JSON file in the `results` directory containing:
- Requirement satisfaction results for each requirement
- List of segments that satisfy each requirement
- Overall completeness metrics
- Satisfaction matrix showing which segments satisfy which requirements

## Metrics

The baseline implementation computes the following metrics:
- Total number of requirements evaluated
- Number of satisfied requirements
- Total number of segments processed
- Completeness score (ratio of satisfied requirements to total requirements)
- Satisfaction matrix showing the relationship between requirements and segments 