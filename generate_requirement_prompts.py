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

def get_head_predicate_from_symbolic(req_symbolic: str) -> str:
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

def generate_requirement_prompt(req_id: str, req_info: Dict, examples: Dict[str, List[Tuple[str, str]]], global_used_examples: set = None) -> Dict:
    """Generate a requirement-specific system prompt with minimal examples to reduce false positives."""
    
    if global_used_examples is None:
        global_used_examples = set()
    
    req_text = req_info.get('text', '')
    req_symbolic = req_info.get('symbolic', '')
    req_predicates = req_info.get('atoms', [])
    
    # Get the head predicate (main obligation)
    head_predicate = get_head_predicate_from_symbolic(req_symbolic)
    
    base_instructions = """You are a legal-text expert that extracts facts from Data-Processing-Agreement (DPA) segments based on semantic and contextual similarity with GDPR regulatory requirements.

Your task is to extract ONLY the predicates that are EXPLICITLY and CLEARLY mentioned in the segment for the specific requirement below.

CRITICAL RULES:
1. Be EXTREMELY strict - only extract predicates if they are clearly and explicitly present
2. If role(processor) is not mentioned, output: NO_FACTS
3. If only processor role is mentioned but requirement not satisfied, output: role(processor)
4. Only output predicates that are semantically relevant to the specific requirement
5. Do NOT infer or assume predicates that are not explicitly stated
6. Use EXACT predicate names from the requirement specification
7. Be VERY conservative - when in doubt, output less rather than more"""

    # Get examples for this requirement
    satisfying_examples = examples.get('satisfying', [])
    no_facts_examples = examples.get('no_facts', [])
    
    prompt_examples = []
    
    # 1. Add exactly ONE satisfying example
    satisfying_added = 0
    for segment, facts in satisfying_examples:
        segment_id = f"{req_id}_{hash(segment)}"
        if segment_id not in global_used_examples and satisfying_added == 0:
            global_used_examples.add(segment_id)
            
            # For satisfying examples, include the head predicate + relevant body predicates
            if head_predicate and 'role(processor)' in req_predicates:
                # Include role(processor) and the head predicate for satisfying examples
                if len(req_predicates) > 2:  # If there are other body predicates
                    other_predicates = [p for p in req_predicates if p not in ['role(processor)', head_predicate]]
                    # Include one other body predicate if it makes sense
                    actual_facts = f"role(processor); {other_predicates[0]}; {head_predicate}" if other_predicates else f"role(processor); {head_predicate}"
                else:
                    actual_facts = f"role(processor); {head_predicate}"
            else:
                # Fallback to original logic
                actual_facts = '; '.join(req_predicates[:2])
            
            prompt_examples.append({
                "segment": segment,
                "expected_output": actual_facts,
                "type": "satisfying"
            })
            satisfying_added += 1
            break
    
    # 2. Add 2 examples that should output only role(processor)
    processor_only_added = 0
    for segment, facts in satisfying_examples[1:]:  # Skip the first one we already used
        if processor_only_added >= 2:
            break
        segment_id = f"{req_id}_{hash(segment)}"
        if segment_id not in global_used_examples:
            global_used_examples.add(segment_id)
            # Create a simplified version that mentions processor but doesn't fully satisfy
            simplified_segment = segment.split('.')[0] + ". The processor handles data processing operations."
            prompt_examples.append({
                "segment": simplified_segment,
                "expected_output": "role(processor)",
                "type": "processor_only"
            })
            processor_only_added += 1
    
    # 3. Add 3 NO_FACTS examples from different sources
    no_facts_added = 0
    for segment, _ in no_facts_examples[:10]:  # Try first 10 to find good ones
        if no_facts_added >= 3:
            break
        segment_id = f"{req_id}_{hash(segment)}"
        if segment_id not in global_used_examples and len(segment.split()) > 10:  # Ensure meaningful length
            global_used_examples.add(segment_id)
            prompt_examples.append({
                "segment": segment,
                "expected_output": "NO_FACTS",
                "type": "no_facts"
            })
            no_facts_added += 1
    
    # 4. If we need more examples, add some generic discriminating examples
    example_count = len(prompt_examples)
    while example_count < 6:
        if example_count == 3:
            prompt_examples.append({
                "segment": "This Data Processing Addendum (DPA) supplements the processor controller Agreement available at as updated from time to time between controller and processor, or other agreement between controller and processor governing controller's use of the Service Offerings (the Agreement) when the GDPR applies to your use of the processor Services to process controller Data.",
                "expected_output": "NO_FACTS",
                "type": "no_facts"
            })
        elif example_count == 4:
            prompt_examples.append({
                "segment": "This DPA is an agreement between you and the entity you represent (controller, you or your) and the applicable Amazon Web Services contracting entity under the Agreement (processor).",
                "expected_output": "NO_FACTS",
                "type": "no_facts"
            })
        elif example_count == 5:
            prompt_examples.append({
                "segment": "Unless otherwise defined in this DPA or in the Agreement, all capitalised terms used in this DPA will have the meanings given to them in Section 17 of this DPA.",
                "expected_output": "NO_FACTS",
                "type": "no_facts"
            })
        example_count += 1
    
    return {
        'requirement_id': req_id,
        'system_prompt': base_instructions,
        'requirement_text': req_text,
        'requirement_symbolic': req_symbolic, 
        'expected_predicates': req_predicates,
        'examples': prompt_examples,
        'examples_used': len(prompt_examples),
        'satisfying_examples': satisfying_added,
        'processor_only_examples': processor_only_added,
        'no_facts_examples': no_facts_added + (len(prompt_examples) - satisfying_added - processor_only_added)
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
            'requirement_text': prompt_data['requirement_text'],
            'requirement_symbolic': prompt_data['requirement_symbolic'],
            'expected_predicates': prompt_data['expected_predicates'],
            'examples': prompt_data['examples'],
            'examples_used': prompt_data['examples_used'],
            'satisfying_examples': prompt_data['satisfying_examples'],
            'processor_only_examples': prompt_data['processor_only_examples'],
            'no_facts_examples': prompt_data['no_facts_examples'],
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