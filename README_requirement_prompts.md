# Requirement-Specific System Prompts

This document describes the new system for generating requirement-specific system prompts with examples for the DPA completeness checker.

## Problem Addressed

Previously, the system used the same generic system prompt for all requirements when analyzing DPA segments. This approach didn't leverage the requirement-specific context and examples that could improve the accuracy of fact extraction.

## Solution

We've implemented a two-step solution:

### 1. Generate Requirement-Specific Prompts (`generate_requirement_prompts.py`)

This script creates tailored system prompts for each requirement by:
- Analyzing the training data (`train_set.csv`) to find examples for each requirement
- Categorizing segments into:
  - **Satisfying**: Segments that fulfill the requirement (labeled with R-labels)
  - **No facts**: Segments labeled as "other" that contain no relevant facts
- Generating requirement-specific system prompts with contextualized examples

#### Usage:

```bash
python generate_requirement_prompts.py \
    --train_data data/train_set.csv \
    --requirements data/requirements/requirements_deontic_ai_generated.json \
    --output requirement_prompts.json \
    --seed 42 \
    --max_examples 3
```

#### Arguments:
- `--train_data`: Path to training data CSV file
- `--requirements`: Path to requirements JSON file
- `--output`: Output JSON file for requirement-specific prompts
- `--seed`: Random seed for reproducible examples
- `--max_examples`: Maximum number of examples per category

### 2. Modified LP File Generation (`generate_lp_files.py`)

The `generate_lp_files.py` script has been enhanced to:
- Load requirement-specific prompts from a JSON file
- Use the appropriate prompt for each requirement during fact extraction
- Fall back to a generic prompt if no specific prompt is available

#### New Usage:

```bash
python generate_lp_files.py \
    --requirements data/requirements/requirements_deontic_ai_generated.json \
    --dpa data/test_set.csv \
    --target_dpa "Online 126" \
    --req_ids "1,2,3" \
    --max_segments 10 \
    --use_ollama \
    --model "gemma3:27b" \
    --output results/lp_files \
    --requirement_prompts requirement_prompts.json
```

#### New Argument:
- `--requirement_prompts`: Path to requirement-specific prompts JSON file (default: `requirement_prompts.json`)

## Data Structure

The generated `requirement_prompts.json` file has the following structure:

```json
{
  "1": {
    "system_prompt": "You are a legal-text expert... [tailored prompt with examples]",
    "requirement_text": "The processor shall not engage a sub-processor...",
    "requirement_symbolic": "&obligatory{-engage_sub_processor} :- role(processor)...",
    "examples_count": {
      "satisfying": 191,
      "no_facts": 8997
    }
  },
  "2": {
    "system_prompt": "You are a legal-text expert... [tailored prompt with examples]",
    ...
  }
}
```

## Label Mapping

The system uses the mapping from `evaluate_completeness.py` to convert between R-labels (R10, R11, etc.) and requirement numbers (1, 2, etc.):

- R10 → Requirement 1
- R11 → Requirement 2
- R12 → Requirement 3
- And so on...

## Benefits

1. **Contextualized Analysis**: Each requirement gets examples specific to its domain
2. **Improved Accuracy**: Examples help the LLM understand what constitutes satisfying vs. non-satisfying segments
3. **Consistency**: Reproducible prompts ensure consistent results across runs
4. **Flexibility**: System gracefully falls back to generic prompts when specific ones aren't available

## Integration with Existing Workflow

The new system integrates seamlessly with the existing pipeline:

1. **Step 1**: Generate requirement-specific prompts (one-time setup)
2. **Step 2**: Use existing `run_dpa_completeness_ollama.sh` or call `generate_lp_files.py` directly
3. **Step 3**: Continue with normal evaluation using `evaluate_completeness.py`

## Example Generated Prompt

Here's an example of a requirement-specific prompt for Requirement 1:

```
You are a legal-text expert that extracts facts from Data-Processing-Agreement (DPA) segments...

SPECIFIC REQUIREMENT CONTEXT:
This analysis focuses on: The processor shall not engage a sub-processor without a prior specific or general written authorization of the controller.
Symbolic representation: &obligatory{-engage_sub_processor} :- role(processor), not authorization(controller).

EXAMPLES for this requirement:

EXAMPLES OF SATISFYING SEGMENTS:
1. DPA: Online 90
   CLAUSE: Name (written out in full): Jerome Ternynck Position: CEO & Founder
   EXPECTED: [relevant predicates based on requirement]

EXAMPLES OF NO_FACTS SEGMENTS:
1. DPA: Online 11
   CLAUSE: If the affected service is part of a suite...
   EXPECTED: NO_FACTS

Now process the given CLAUSE according to these patterns.
```

## Files Created/Modified

### New Files:
- `generate_requirement_prompts.py` - Script to generate requirement-specific prompts
- `requirement_prompts.json` - Generated prompts for each requirement
- `README_requirement_prompts.md` - This documentation

### Modified Files:
- `generate_lp_files.py` - Enhanced to use requirement-specific prompts

## Future Improvements

1. **Enhanced Categorization**: Better distinguish between fully satisfying, partially satisfying, and violating segments
2. **Dynamic Examples**: Adjust examples based on the current DPA being analyzed
3. **Example Quality**: Improve example selection based on semantic similarity
4. **Partial Labels**: Handle "Partial" labels in the training data more intelligently 