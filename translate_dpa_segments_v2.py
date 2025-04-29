import os
import json
import argparse
import pandas as pd
import re
from tqdm import tqdm
from models.llm import LlamaModel
from solver.deontic_translator import DeonticTranslator
from config.llm_config import LlamaConfig

def extract_strict_vocabulary(requirements_file):
    """
    Extract strictly controlled vocabulary from requirements symbolic representations.
    
    Args:
        requirements_file: Path to requirements symbolic JSON file
        
    Returns:
        Dictionary containing allowed actions, predicates, and example patterns
    """
    try:
        with open(requirements_file, 'r') as f:
            requirements_data = json.load(f)
    except Exception as e:
        print(f"Error loading requirements file: {e}")
        return {
            "allowed_actions": set(),
            "allowed_predicates": set(),
            "example_patterns": []
        }
    
    # Extract vocabulary
    allowed_actions = set()
    allowed_predicates = set()
    example_patterns = []
    
    # Extract actions from deontic operators
    action_pattern = r'&(?:obligatory|permitted|forbidden){([^}]+)}'
    
    for req_text, symbolic in requirements_data.items():
        # Extract actions from within deontic operators
        action_matches = re.findall(action_pattern, symbolic)
        for action in action_matches:
            action = action.strip()
            if action:
                # Add the entire action including arguments
                allowed_actions.add(action)
                
                # Also extract the predicate part (before parentheses)
                if '(' in action:
                    predicate = action.split('(')[0].strip()
                    allowed_predicates.add(predicate)
                else:
                    allowed_predicates.add(action)
        
        # Extract full rule patterns for examples
        for line in symbolic.split('\n'):
            line = line.strip()
            if not line.startswith('%') and "&" in line and "{" in line and "}" in line and ":-" in line:
                example_patterns.append(line)
        
        # Extract predicates from rule conditions
        for line in symbolic.split('\n'):
            if ':-' in line and not line.strip().startswith('%'):
                body = line.split(':-')[1].strip()
                if body.endswith('.'):
                    body = body[:-1]
                
                # Split by common operators
                for op in [',', '&', '|', ';']:
                    if op in body:
                        parts = body.split(op)
                        for part in parts:
                            clean_part = part.strip()
                            if clean_part and not clean_part.startswith('&') and not clean_part.startswith('%'):
                                # Add full predicate with arguments
                                allowed_predicates.add(clean_part)
                                
                                # Also add the name only (before parentheses)
                                if '(' in clean_part:
                                    pred_name = clean_part.split('(')[0].strip()
                                    allowed_predicates.add(pred_name)
    
    # Add default actions if none found
    if not allowed_actions:
        allowed_actions = {"comply(processor, controller)", "process_data(processor)", "generic_action(processor)"}
    
    # Add default predicates if none found
    if not allowed_predicates:
        allowed_predicates = {"controller", "processor", "comply", "data", "authorized"}
    
    # Get at most 5 example patterns
    example_patterns = example_patterns[:5]
    
    # Add default patterns if none found
    if not example_patterns:
        example_patterns = [
            "&obligatory{comply(processor, controller)} :- authorized(controller).",
            "&forbidden{process_data(processor)} :- not authorized(controller)."
        ]
    
    return {
        "allowed_actions": allowed_actions,
        "allowed_predicates": allowed_predicates,
        "example_patterns": example_patterns
    }

def create_few_shot_examples(vocabulary):
    """
    Create few-shot examples using the vocabulary.
    
    Args:
        vocabulary: Dictionary with allowed actions, predicates, and example patterns
        
    Returns:
        String with few-shot examples
    """
    examples = [
        {
            "input": "The processor shall obtain written authorization before engaging any sub-processor.",
            "output": "&obligatory{process_personal_data} :- documented_instruction(controller, processor)."
        },
        {
            "input": "The controller may access the personal data processed by the processor at any time.",
            "output": "&permitted{assist} :- not &forbidden{assist}."
        },
        {
            "input": "The processor shall not disclose personal data to third parties without authorization.",
            "output": "&forbidden{engage_sub_processor_sub_processor} :- not prior_written_authorization(controller)."
        }
    ]
    
    # Replace example outputs with actual patterns from vocabulary if possible
    if vocabulary["example_patterns"]:
        for i in range(min(len(examples), len(vocabulary["example_patterns"]))):
            examples[i]["output"] = vocabulary["example_patterns"][i]
    
    few_shot_text = "\nExamples of translations:\n\n"
    for ex in examples:
        few_shot_text += f"DPA segment: {ex['input']}\nSymbolic representation: {ex['output']}\n\n"
    
    return few_shot_text

def translate_dpa_segments(dpa_file, requirements_file, model_path, output_file, target_dpa=None):
    """
    Translate DPA segments to symbolic representations using strict vocabulary from requirements.
    
    Args:
        dpa_file: Path to DPA segments CSV file
        requirements_file: Path to requirements symbolic JSON file
        model_path: Path to LLM model file
        output_file: Output file path for JSON results
        target_dpa: Optional DPA ID to filter (if None, use first DPA found)
    """
    # Initialize LLM and translator
    print(f"Initializing LLM with model: {model_path}")
    llm_config = LlamaConfig(model_path=model_path, temperature=0.1)
    llm_model = LlamaModel(llm_config)
    translator = DeonticTranslator()
    print("LLM initialized successfully")
    
    # Extract vocabulary from requirements
    print(f"Extracting vocabulary from requirements: {requirements_file}")
    vocabulary = extract_strict_vocabulary(requirements_file)
    
    allowed_actions_text = "\n".join(sorted(vocabulary["allowed_actions"]))
    allowed_predicates_text = "\n".join(sorted(vocabulary["allowed_predicates"]))
    
    print(f"Extracted {len(vocabulary['allowed_actions'])} actions and {len(vocabulary['allowed_predicates'])} predicates")
    
    # Create few-shot examples
    few_shot_examples = create_few_shot_examples(vocabulary)
    
    # Read CSV file
    print(f"Reading DPA segments from: {dpa_file}")
    df = pd.read_csv(dpa_file)
    
    # Filter to target DPA or select first one
    if target_dpa:
        filtered_df = df[df["DPA"] == target_dpa]
        if filtered_df.empty:
            print(f"Error: DPA '{target_dpa}' not found in file")
            return
    else:
        # Get unique DPAs and select the first one if no target specified
        unique_dpas = df["DPA"].unique()
        if len(unique_dpas) == 0:
            print("Error: No DPAs found in file")
            return
        target_dpa = unique_dpas[0]
        filtered_df = df[df["DPA"] == target_dpa]
    
    print(f"Processing {len(filtered_df)} segments for DPA: {target_dpa}")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Process segments
    dpa_data = {
        "dpa_id": target_dpa,
        "segments": {}
    }
    
    # Create a strict vocabulary prompt - using double braces to escape the braces
    system_prompt = """
You are a specialized translator for DPA (Data Processing Agreement) segments into formal symbolic representations.

IMPORTANT: You MUST ONLY use the EXACT vocabulary provided below. Do not invent new predicates or actions.

Your task is to translate the given DPA segment into a symbolic representation using deontic logic operators.
Each segment should be analyzed for obligations (&obligatory), permissions (&permitted), or prohibitions (&forbidden).

STRICT RULES TO FOLLOW:
1. ONLY use predicates and actions from the ALLOWED LIST below
2. If no appropriate predicate/action exists, use 'comply(processor, controller)' as a fallback
3. Do not invent new predicates or modify the allowed ones
4. Maintain the exact syntax: &operator{{action}} :- conditions.
5. Follow the provided examples closely for format
"""

    strict_vocab_prompt = f"""
ALLOWED ACTIONS (use these exact strings, including arguments):
{allowed_actions_text}

ALLOWED PREDICATES (use these exact strings):
{allowed_predicates_text}

{few_shot_examples}

Instructions for this specific translation:
1. Identify if the segment describes an obligation, permission, or prohibition
2. Select ONLY FROM THE ALLOWED ACTIONS above that best matches the segment
3. If no match exists, use 'comply(processor, controller)'
4. Format as: &obligatory{{action}} :- conditions.  (or &permitted/&forbidden)
5. Only use conditions from the ALLOWED PREDICATES list
"""
    
    print("Translating segments...")
    for _, row in tqdm(filtered_df.iterrows(), total=len(filtered_df)):
        segment_id = row["ID"]
        segment_text = row["Sentence"]
        
        # Find related requirements to further improve vocabulary alignment
        related_reqs = []
        for col in ["Requirement-1", "Requirement-2", "Requirement-3"]:
            if col in row and pd.notna(row[col]) and row[col]:
                related_reqs.append(row[col])
        
        # Add related requirements context to the prompt if available
        segment_context = segment_text
        if related_reqs:
            segment_context = f"This segment is related to requirements: {', '.join(related_reqs)}.\nSegment: {segment_text}"
        
        # Full prompt combining system prompt and specific instructions
        user_prompt = f"{strict_vocab_prompt}\n\nTranslate this DPA segment into symbolic representation:\n{segment_context}\n\nSymbolic representation:"
        
        # Generate symbolic representation with strict vocabulary control
        llm_output = llm_model.generate_symbolic_representation(segment_context, system_prompt + user_prompt)
        symbolic = translator.translate(llm_output)
        
        # Store in dictionary with metadata
        dpa_data["segments"][segment_text] = {
            "id": segment_id,
            "symbolic": symbolic,
            "requirements": related_reqs
        }
    
    # Save to JSON file
    print(f"Saving symbolic representations to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(dpa_data, f, indent=2)
    
    print("DPA segment translation complete!")

def main():
    parser = argparse.ArgumentParser(description="Translate DPA segments to symbolic representation")
    parser.add_argument("--dpa", type=str, default="data/train_set.csv",
                        help="Path to DPA segments CSV file")
    parser.add_argument("--requirements", type=str, default="data/processed/requirements_symbolic.json",
                        help="Path to requirements symbolic JSON file")
    parser.add_argument("--model", type=str, default="models/Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf",
                        help="Path to LLM model file")
    parser.add_argument("--output", type=str, default="data/processed/dpa_segments_symbolic_v2.json",
                        help="Output JSON file path")
    parser.add_argument("--target", type=str, default="Online 1",
                        help="Target DPA to process (default: Online 1)")
    args = parser.parse_args()
    
    translate_dpa_segments(args.dpa, args.requirements, args.model, args.output, args.target)

if __name__ == "__main__":
    main()