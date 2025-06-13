# RCV Approach: Requirement Classification and Verification

## Overview

The RCV (Requirement Classification and Verification) approach is an improved method for processing DPA (Data Processing Agreement) text segments that addresses the noise problem of checking every segment against every requirement.

## Key Improvements

### Traditional Approach Problems
- **Computational Inefficiency**: Every segment was checked against all 19 GDPR requirements
- **Noise Amplification**: Many irrelevant segments (headings, boilerplate) generated false positives
- **Poor Signal-to-Noise Ratio**: Relevant information was buried in irrelevant classifications

### RCV Approach Solutions
- **Two-Step Process**: First classify relevance, then verify satisfaction
- **Focused Analysis**: Only perform detailed verification on relevant segments
- **Reduced Noise**: Administrative text and headings are filtered out as "NONE"
- **Symbolic Reasoning**: Uses ASP (Answer Set Programming) for formal verification

## How It Works

### Step 1: Classification
For each DPA segment, the LLM determines:
- Which single GDPR requirement (if any) it is most relevant to
- Returns either a requirement ID (e.g., "3") or "NONE"

### Step 2: Verification (Conditional)
If and only if a requirement was identified:
- Extract symbolic facts specific to that requirement
- Use ASP solver to formally verify if the obligation is met
- Combine facts with logic rules for final determination

## Architecture

```
DPA Segment → Classification LLM → Requirement ID or "NONE"
                                          ↓
                                   [If not "NONE"]
                                          ↓
                              Verification LLM → Extract Facts
                                          ↓
                              Logic Program Generator
                                          ↓
                                 ASP Solver (deolingo)
                                          ↓
                                  Final Prediction
```

## Usage

### Prerequisites

1. **Install Dependencies**:
   ```bash
   pip3 install -r requirements.txt
   ```

2. **Install Deolingo ASP Solver**:
   ```bash
   pip3 install git+https://github.com/ovidiomanteiga/deolingo.git@main
   ```

3. **Start Ollama Server**:
   ```bash
   ollama serve
   ```

4. **Pull Required Model**:
   ```bash
   ollama pull llama3.3:70b
   ```

### Basic Usage

```bash
python3 classify_and_verify.py \
  --requirements data/requirements/requirements_deontic_ai_generated.json \
  --dpa_segments data/test_set.csv \
  --target_dpa "Online 124" \
  --output_dir results/rcv_output \
  --model llama3.3:70b \
  --max_segments 10 \
  --verbose
```

### Parameters

- `--requirements`: Path to requirements JSON file
- `--dpa_segments`: Path to DPA segments CSV file  
- `--target_dpa`: Name of the DPA to process
- `--output_dir`: Directory for output files
- `--model`: Ollama model to use (default: llama3.3:70b)
- `--max_segments`: Limit number of segments (0 = all)
- `--verbose`: Enable detailed logging

### Test Run

Use the included test script for a quick demonstration:

```bash
python3 test_rcv_approach.py
```

This will process 5 segments from "Online 124" and demonstrate the complete workflow.

## Output Files

### Results CSV
Contains the main results with columns:
- `Segment_ID`: Unique identifier for the segment
- `DPA`: Name of the DPA
- `Segment_Text`: Original text of the segment
- `LLM_Classification`: Which requirement ID was classified (or "NONE")
- `Extracted_Facts`: Symbolic facts extracted from the segment
- `Final_Prediction`: Final ASP solver prediction
- `Ground_Truth`: Known correct answer (if available)

### Logic Program Files
- Generated in `output_dir/lp_files/DPA_NAME/`
- One `.lp` file per segment
- Contains the complete logic program used by the ASP solver

## Logic Program Structure

Each generated logic program contains three parts:

### 1. Static Verification Engine
```prolog
% Rules for all 19 requirements
requirement_satisfied(1) :- role(processor), not authorization(controller), engage_sub_processor.
requirement_satisfied(2) :- role(processor), general_written_authorization, inform_controller_changes.
% ... (all 19 requirements)
```

### 2. Dynamic Facts from LLM
```prolog
% Classification result
classified_as(3).

% Extracted facts specific to the classified requirement
role(processor).
process_data_on_documented_instructions.
```

### 3. Final Prediction Logic
```prolog
% Determine final answer based on facts and rules
final_prediction(R) :- requirement_satisfied(R).
final_prediction(none) :- classified_as(R), R != none, not requirement_satisfied(R).
final_prediction(none) :- classified_as(none).
#show final_prediction/1.
```

## Example Workflow

### Input Segment
```
"The processor may only act and process the Personal Data in accordance with the documented instruction from the Controller."
```

### Step 1: Classification
- **LLM Analysis**: This segment describes processing instructions
- **Result**: Classified as requirement "3" (processor shall process data only on documented instructions)

### Step 2: Verification
- **Available Atoms**: `["role(processor)", "process_data_on_documented_instructions"]`
- **LLM Extraction**: `role(processor);process_data_on_documented_instructions`

### Step 3: Logic Program Generation
```prolog
% Static engine (abbreviated)
requirement_satisfied(3) :- role(processor), process_data_on_documented_instructions.

% Dynamic facts
classified_as(3).
role(processor).
process_data_on_documented_instructions.

% Final logic
final_prediction(R) :- requirement_satisfied(R).
```

### Step 4: ASP Solver Result
- **Final Prediction**: "3" (requirement is satisfied)

## Comparison with Traditional Approach

| Aspect | Traditional | RCV Approach |
|--------|------------|--------------|
| LLM Calls per Segment | 19 (one per requirement) | 1-2 (classification + verification if needed) |
| Noise Handling | Poor (all segments processed equally) | Good (irrelevant segments filtered out) |
| Computational Efficiency | Low (O(n×m) where n=segments, m=requirements) | High (O(n) for most segments) |
| False Positives | High (many irrelevant matches) | Low (focused verification) |
| Symbolic Reasoning | Per-requirement basis | Unified ASP engine |

## Benefits

1. **Efficiency**: Reduces LLM calls by ~90% for irrelevant segments
2. **Accuracy**: Focused analysis reduces false positives
3. **Scalability**: Linear complexity instead of quadratic
4. **Maintainability**: Single unified logic engine
5. **Interpretability**: Clear two-step decision process

## Troubleshooting

### Common Issues

1. **Ollama Server Not Running**:
   ```
   Error: Ollama server is not running. Please start it first.
   ```
   Solution: Run `ollama serve` in a separate terminal

2. **Model Not Available**:
   ```
   Error: Model 'llama3.3:70b' not found
   ```
   Solution: Run `ollama pull llama3.3:70b`

3. **Deolingo Not Installed**:
   ```
   Error: 'deolingo' command not found
   ```
   Solution: Install with pip3 as shown in prerequisites

4. **No Segments Found**:
   ```
   Error: No segments found for DPA: "YourDPA"
   ```
   Solution: Check DPA name matches exactly (case-sensitive)

### Performance Tips

- Use `--max_segments` for testing to limit processing time
- Enable `--verbose` for debugging classification/verification steps
- Monitor Ollama server logs for model performance
- Check ASP solver output in verbose mode for logic errors

## Future Enhancements

- Support for multiple classification models
- Batch processing for multiple DPAs
- Integration with evaluation metrics
- Export to different output formats
- Real-time processing capabilities 