#!/usr/bin/env python3
"""
Generate requirement-specific system prompts with examples.

This script creates a JSON file containing system prompts tailored for each requirement,
with examples from the training data showing different types of segments:
- Satisfying (segments that fulfill the requirement)
- Partially satisfying (segments that only partially fulfill the requirement)
- No facts (segments labeled as "other" with no relevant facts)
"""

import pandas as pd
import json
import random
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

def map_r_label_to_req_number(r_label: str) -> str:
    """Map R-label to requirement number (from evaluate_completeness.py)."""
    if not r_label.startswith('R'):
        return None
        
    try:
        r_number = int(r_label.replace('R', ''))
    except ValueError:
        return None
    
    mapping = {
        10: 1, 11: 2, 12: 3, 13: 4, 15: 5, 16: 6, 17: 7, 18: 8, 19: 9,
        20: 10, 21: 11, 22: 12, 23: 13, 24: 14, 25: 15, 26: 16, 27: 17,
        28: 18, 29: 19, 30: 20, 31: 21, 32: 22, 33: 23, 34: 24, 35: 25,
        36: 26, 37: 27, 38: 28
    }
    
    return str(mapping.get(r_number, r_number))

def req_number_to_r_label(req_number: str) -> str:
    """Inverse mapping from requirement number to R-label."""
    inverse_mapping = {
        '1': '10', '2': '11', '3': '12', '4': '13', '5': '15', '6': '16', '7': '17',
        '8': '18', '9': '19', '10': '20', '11': '21', '12': '22', '13': '23', '14': '24',
        '15': '25', '16': '26', '17': '27', '18': '28', '19': '29', '20': '30', '21': '31',
        '22': '32', '23': '33', '24': '34', '25': '35', '26': '36', '27': '37', '28': '38'
    }
    
    return inverse_mapping.get(str(req_number), str(req_number))

def load_requirements(requirements_file: str) -> Dict:
    """Load requirements from JSON file."""
    with open(requirements_file, 'r') as f:
        return json.load(f)

def categorize_segments_by_requirement(df: pd.DataFrame) -> Dict[str, Dict[str, List[Tuple[str, str]]]]:
    """
    Categorize segments by requirement and satisfaction level.
    
    Returns:
        Dict[req_number, Dict[category, List[(sentence, dpa_name)]]]
        where category is 'satisfying', 'partial', or 'no_facts'
    """
    categorized = defaultdict(lambda: defaultdict(list))
    
    for _, row in df.iterrows():
        sentence = row['Sentence']
        dpa_name = row['DPA']
        
        # Get all R-labels for this row
        row_r_labels = set()
        for req_col in ['Requirement-1', 'Requirement-2', 'Requirement-3']:
            label = row[req_col]
            if pd.notna(label) and label.startswith('R'):
                row_r_labels.add(label)
        
        # Check if this row has "other" label (no facts case)
        has_other = any(row[req_col] == 'other' for req_col in ['Requirement-1', 'Requirement-2', 'Requirement-3'] if pd.notna(row[req_col]))
        
        # Process each requirement (1-28)
        for req_num in range(1, 29):
            req_str = str(req_num)
            r_label = f"R{req_number_to_r_label(req_str)}"
            
            if r_label in row_r_labels:
                # This segment satisfies this requirement
                categorized[req_str]['satisfying'].append((sentence, dpa_name))
            elif has_other and len(row_r_labels) == 0:
                # This segment is labeled as "other" with no R-labels - it's a no_facts example
                categorized[req_str]['no_facts'].append((sentence, dpa_name))
            elif len(row_r_labels) > 0 and r_label not in row_r_labels:
                # This segment has R-labels for other requirements but not this one
                # It mentions processor context but doesn't satisfy this specific requirement (partial)
                categorized[req_str]['partial'].append((sentence, dpa_name))
    
    return categorized

def select_examples(segments: List[Tuple[str, str]], max_examples: int = 3, used_examples: set = None) -> List[Tuple[str, str]]:
    """Select a random sample of examples, avoiding duplicates."""
    if used_examples is None:
        used_examples = set()
    
    # Filter out already used examples
    available_segments = [seg for seg in segments if seg[0] not in used_examples]
    
    if len(available_segments) <= max_examples:
        selected = available_segments
    else:
        selected = random.sample(available_segments, max_examples)
    
    # Add selected examples to used set
    for seg in selected:
        used_examples.add(seg[0])
    
    return selected

def generate_requirement_prompt(req_id: str, req_info: Dict, examples: Dict[str, List[Tuple[str, str]]], global_used_examples: set = None) -> Dict:
    """Generate a system prompt for a specific requirement with examples."""
    
    if global_used_examples is None:
        global_used_examples = set()
    
    # Track examples used within this requirement to avoid internal duplicates
    local_used_examples = set()
    
    base_instructions = """You are a legal-text expert that extracts facts from Data-Processing-Agreement (DPA) segments based on semantic and contextual similarity with GDPR regulatory requirements.

Input always contains:
1. "REQUIREMENT" – text of the GDPR requirement
2. "SYMBOLIC" – symbolic representation of the requirement in deontic logic via Answer Set Programming (ASP)
3. "PREDICATES" – ASP atoms from the requirement (semicolon-separated)
4. "CLAUSE" – one DPA segment

TASK:
Decide which (if any) predicates are explicitly fully mentioned in the CLAUSE and output them separated by semicolon

INSTRUCTIONS:
1) Output a predicate from symbolic rule's body only if the CLAUSE explicitly and fully mentions the same concept this predicate mentions in the REQUIREMENT.
2) Output a predicate from symbolic rule's head only if the CLAUSE describes a rule for a processor and this rule is semantically the same as the REQUIREMENT
3) If no predicated are entailed, output exactly NO_FACTS
4) If the CLAUSE explicitly violates a predicate, output it prefixed by - (e.g. -encrypt_data)
5) Output ONLY extracted predicates or NO_FACTS, do not output explanation or something else."""

    req_text = req_info.get('text', '')
    req_symbolic = req_info.get('symbolic', '')
    req_predicates = req_info.get('atoms', [])
    
    # Generate examples as structured data
    prompt_examples = []
    example_count = 1
    
    # Add satisfying examples (avoid both global and local duplicates)
    satisfying_examples = select_examples(examples.get('satisfying', []), 2, global_used_examples.union(local_used_examples))
    if satisfying_examples:
        for i, (sentence, dpa) in enumerate(satisfying_examples):
            example = {
                "example_number": example_count,
                "type": "satisfying",
                "requirement": req_text,
                "symbolic": req_symbolic,
                "predicates": '; '.join(req_predicates),
                "clause": sentence,
                "expected_output": '; '.join(req_predicates),
                "dpa_source": dpa
            }
            prompt_examples.append(example)
            local_used_examples.add(sentence)
            example_count += 1
    
    # Add partial examples (avoid both global and local duplicates)
    partial_examples = select_examples(examples.get('partial', []), 1, global_used_examples.union(local_used_examples))
    if partial_examples:
        sentence, dpa = partial_examples[0]
        expected_output = "role(processor)" if 'role(processor)' in req_predicates else "NO_FACTS"
        example = {
            "example_number": example_count,
            "type": "partial",
            "requirement": req_text,
            "symbolic": req_symbolic,
            "predicates": '; '.join(req_predicates),
            "clause": sentence,
            "expected_output": expected_output,
            "dpa_source": dpa
        }
        prompt_examples.append(example)
        local_used_examples.add(sentence)
        example_count += 1
    
    # Add violation example (synthetic - always unique)
    violation_clause = create_violation_example(req_text, req_symbolic)
    violated_predicate = get_main_predicate_from_symbolic(req_symbolic)
    if violated_predicate and violated_predicate in req_predicates:
        other_predicates = [p for p in req_predicates if p != violated_predicate]
        violation_output = f"-{violated_predicate}"
        if other_predicates:
            violation_output += f"; {'; '.join(other_predicates)}"
    else:
        violation_output = "NO_FACTS"
    
    example = {
        "example_number": example_count,
        "type": "violation",
        "requirement": req_text,
        "symbolic": req_symbolic,
        "predicates": '; '.join(req_predicates),
        "clause": violation_clause,
        "expected_output": violation_output,
        "dpa_source": "synthetic"
    }
    prompt_examples.append(example)
    example_count += 1
    
    # Add no_facts example (avoid both global and local duplicates)
    no_facts_examples = select_examples(examples.get('no_facts', []), 1, global_used_examples.union(local_used_examples))
    if no_facts_examples:
        sentence, dpa = no_facts_examples[0]
        example = {
            "example_number": example_count,
            "type": "no_facts",
            "requirement": req_text,
            "symbolic": req_symbolic,
            "predicates": '; '.join(req_predicates),
            "clause": sentence,
            "expected_output": "NO_FACTS",
            "dpa_source": dpa
        }
        prompt_examples.append(example)
        local_used_examples.add(sentence)
    
    # Add all local examples to global used set
    global_used_examples.update(local_used_examples)
    
    # Create the full system prompt by combining instructions and examples
    full_prompt = base_instructions + "\n\nExamples:\n"
    for example in prompt_examples:
        full_prompt += f"Example {example['example_number']}:\n"
        full_prompt += f"REQUIREMENT: {example['requirement']}\n"
        full_prompt += f"SYMBOLIC: {example['symbolic']}\n"
        full_prompt += f"PREDICATES: {example['predicates']}\n"
        full_prompt += f"CLAUSE: {example['clause']}\n"
        full_prompt += f"Expected output: {example['expected_output']}\n\n"
    
    return {
        'system_prompt': full_prompt,
        'base_instructions': base_instructions,
        'examples': prompt_examples,
        'requirement_text': req_text,
        'requirement_symbolic': req_symbolic
    }

def create_violation_example(req_text: str, req_symbolic: str) -> str:
    """Create a synthetic violation example based on the requirement."""
    
    # Map requirements to violation examples
    violation_examples = {
        "sub-processor": "The processor will engage sub-processors without seeking any authorization from the controller.",
        "documented instructions": "The processor will process personal data according to its own internal procedures without following controller instructions.",
        "confidentiality": "The processor allows all staff members to access personal data without any confidentiality agreements.",
        "security": "The processor stores all personal data in unencrypted plain text files.",
        "assist the controller": "The processor refuses to provide any assistance to the controller regarding data subject requests.",
        "return or delete": "The processor will retain all personal data indefinitely after contract termination.",
        "inform the controller": "The processor will not notify the controller of any data processing issues or legal requirements.",
        "compliance information": "The processor will not provide any documentation or information about its data processing practices.",
        "audits": "The processor prohibits any audits or inspections of its data processing facilities.",
        "sub-processor obligations": "The processor allows sub-processors to operate without any contractual obligations or oversight.",
        "liable": "The processor disclaims all liability for sub-processor actions and data breaches.",
        "assess": "The processor does not conduct any risk assessments for data security."
    }
    
    # Find matching violation example
    req_lower = req_text.lower()
    for keyword, violation in violation_examples.items():
        if keyword in req_lower:
            return violation
    
    # Default generic violation
    return "The processor will not comply with this data protection requirement."

def get_main_predicate_from_symbolic(req_symbolic: str) -> str:
    """Extract the main predicate from the symbolic representation."""
    
    # Extract predicate from deontic operators
    if "&obligatory{" in req_symbolic:
        predicate = req_symbolic.split("&obligatory{")[1].split("}")[0]
        # Remove negation prefix if present
        if predicate.startswith('-'):
            return predicate[1:]
        return predicate
    elif "&forbidden{" in req_symbolic:
        predicate = req_symbolic.split("&forbidden{")[1].split("}")[0]
        return predicate
    elif "&permitted{" in req_symbolic:
        predicate = req_symbolic.split("&permitted{")[1].split("}")[0]
        return predicate
    
    return None

def main():
    parser = argparse.ArgumentParser(description="Generate requirement-specific system prompts")
    parser.add_argument("--train_data", type=str, default="data/train_set.csv",
                        help="Path to training data CSV file")
    parser.add_argument("--requirements", type=str, default="data/requirements/requirements_deontic_ai_generated.json",
                        help="Path to requirements JSON file")
    parser.add_argument("--output", type=str, default="requirement_prompts.json",
                        help="Output JSON file for requirement-specific prompts")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible examples")
    parser.add_argument("--max_examples", type=int, default=3,
                        help="Maximum number of examples per category")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    print(f"Loading training data from: {args.train_data}")
    df = pd.read_csv(args.train_data)
    print(f"Loaded {len(df)} training examples")
    
    print(f"Loading requirements from: {args.requirements}")
    requirements = load_requirements(args.requirements)
    print(f"Loaded {len(requirements)} requirements")
    
    print("Categorizing segments by requirement...")
    categorized_segments = categorize_segments_by_requirement(df)
    
    # Generate prompts for each requirement
    requirement_prompts = {}
    global_used_examples = set()  # Track examples used across all requirements
    
    for req_id, req_info in requirements.items():
        print(f"Generating prompt for requirement {req_id}...")
        
        # Get examples for this requirement
        examples = categorized_segments.get(req_id, {})
        
        # Generate the prompt
        prompt_data = generate_requirement_prompt(req_id, req_info, examples, global_used_examples)
        
        requirement_prompts[req_id] = {
            'system_prompt': prompt_data['system_prompt'],
            'base_instructions': prompt_data['base_instructions'],
            'examples': prompt_data['examples'],
            'requirement_text': req_info.get('text', ''),
            'requirement_symbolic': req_info.get('symbolic', ''),
            'examples_count': {
                'satisfying': len(examples.get('satisfying', [])),
                'no_facts': len(examples.get('no_facts', [])),
                'partial': len(examples.get('partial', []))
            }
        }
    
    # Save to JSON file
    print(f"Saving requirement prompts to: {args.output}")
    with open(args.output, 'w') as f:
        json.dump(requirement_prompts, f, indent=2)
    
    print("Done!")
    print(f"Generated prompts for {len(requirement_prompts)} requirements")
    
    # Print summary statistics
    total_satisfying = sum(data['examples_count']['satisfying'] for data in requirement_prompts.values())
    total_no_facts = sum(data['examples_count']['no_facts'] for data in requirement_prompts.values())
    total_partial = sum(data['examples_count']['partial'] for data in requirement_prompts.values())
    
    print(f"Total satisfying examples: {total_satisfying}")
    print(f"Total no_facts examples: {total_no_facts}")
    print(f"Total partial examples: {total_partial}")

if __name__ == "__main__":
    main() 