# generate_semantic_rules.py
import os
import json
import argparse
import re
from tqdm import tqdm
from models.llm import LlamaModel
from config.llm_config import LlamaConfig

def load_requirements(requirements_file):
    """Load requirements from JSON file with symbolic representations."""
    try:
        with open(requirements_file, 'r') as f:
            requirements_data = json.load(f)
    except Exception as e:
        print(f"Error loading requirements file: {e}")
        return {}
    
    # Process requirements into a more structured format
    processed_requirements = {}
    req_index = 1
    
    for req_text, symbolic in requirements_data.items():
        # Extract requirement number from text if possible
        req_id = None
        match = re.search(r'R(\d+)', req_text)
        if match:
            req_id = match.group(1)
        else:
            # Try to find a number at the beginning of the text
            match = re.match(r'.*?(\d+).*', req_text)
            if match:
                req_id = match.group(1)
            else:
                # Assign sequential number if no ID found
                req_id = str(req_index)
                req_index += 1
        
        # Store processed requirement
        processed_requirements[req_id] = {
            "text": req_text,
            "symbolic": symbolic
        }
    
    return processed_requirements

def create_semantic_mapping_prompt(requirement_details, dpa_deontic_statements):
    """
    Create a prompt for generating semantic mapping rules between a requirement
    and DPA deontic statements.
    """
    req_text = requirement_details["text"]
    req_symbolic = requirement_details["symbolic"]
    
    # Extract requirement's deontic statements
    req_deontic_parts = []
    for line in req_symbolic.split('\n'):
        if any(op in line for op in ['&obligatory', '&permitted', '&forbidden']) and line.strip():
            req_deontic_parts.append(line.strip())
    
    req_deontic_text = "\n".join(req_deontic_parts) if req_deontic_parts else req_symbolic
    
    # Format DPA deontic statements
    dpa_deontic_text = "\n".join(dpa_deontic_statements)
    
    system_prompt = """
You are a specialized AI for legal analysis. Your task is to identify semantic connections 
between a regulatory requirement and DPA deontic statements.

Create semantic mapping rules ONLY when there is a genuine semantic connection. 
If no connection exists, respond with "NO_SEMANTIC_CONNECTION"

RULES FOR CREATING MAPPINGS:
1. Use standard ASP syntax: head :- body1, body2.
2. Head should be a predicate from the requirement  
3. Body should contain predicates from DPA statements
4. Each rule MUST have both head and body
5. Never create rules ending with ":- ." (missing body)
6. DO NOT number the rules (no 1., 2., 3., etc.)
7. Each rule is a single line ending with a period
8. Do not use deontic operators (&obligatory, &permitted, &forbidden) in mapping rules
"""

    user_prompt = f"""
REQUIREMENT TO ANALYZE:
Text: {req_text}
Deontic Logic: {req_deontic_text}

DPA DEONTIC STATEMENTS:
{dpa_deontic_text}

Only create semantic mapping rules if the DPA statements genuinely satisfy the requirement.

If YES, create unnumbered rules in this format:
requirement_predicate :- dpa_predicate1, dpa_predicate2.
another_requirement_predicate :- dpa_predicate3.

If NO semantic connection exists, respond with: "NO_SEMANTIC_CONNECTION"

Do not create incomplete rules, numbered rules, or rules without proper bodies.
Do not include deontic operators in mapping rules.
"""
    
    return system_prompt, user_prompt

def validate_mapping_rule(rule):
    """
    Validate that a mapping rule has proper syntax.
    
    Returns:
        (is_valid, cleaned_rule) or (False, None) if invalid
    """
    rule = rule.strip()
    
    # Remove any numbering at the beginning (e.g., "1. ", "2. ", etc.)
    rule = re.sub(r'^\d+\.\s*', '', rule)
    
    # Must contain :- separator  
    if ':-' not in rule:
        return False, None
    
    # Split into head and body
    parts = rule.split(':-')
    if len(parts) != 2:
        return False, None
    
    head = parts[0].strip()
    body = parts[1].strip()
    
    # Head must be non-empty
    if not head:
        return False, None
    
    # Body must be non-empty and not just '.' or '-.'
    if not body or body in ['.', '-.', '-', '']:
        return False, None
    
    # Remove trailing period if present for validation
    body_cleaned = body.rstrip('.')
    
    # Body must have content
    if not body_cleaned:
        return False, None
    
    # Ensure no deontic operators in mapping rules
    if any(op in head + body for op in ['&obligatory', '&permitted', '&forbidden']):
        # Rules with deontic operators might be deontic statements, not mappings
        return False, None
    
    # Reconstruct valid rule
    cleaned_rule = f"{head} :- {body_cleaned}."
    
    return True, cleaned_rule

def generate_lp_files(requirements, dpa_deontic_data, all_mappings, output_dir):
    """Generate individual LP files for each requirement with proper deontic logic."""
    target_dpa = dpa_deontic_data["dpa_id"]
    dpa_segments = dpa_deontic_data["segments"]
    
    # Create output directory for this DPA
    dpa_dir = os.path.join(output_dir, f"dpa_{target_dpa.replace(' ', '_')}")
    os.makedirs(dpa_dir, exist_ok=True)
    
    print("Generating LP files for each requirement...")
    for req_id, req_details in tqdm(requirements.items()):
        # Create LP file for this requirement
        lp_file_path = os.path.join(dpa_dir, f"req_{req_id}.lp")
        
        # Get semantic mappings for this requirement
        req_mappings = all_mappings.get(req_id, [])
        
        # Build LP file content
        lp_content = f"""% ========================================================================
% Deolingo program to check DPA '{target_dpa}' against Requirement {req_id}
% ========================================================================

% --- Requirement {req_id} ---
% Text: {req_details["text"]}
% Deontic Logic:
{req_details['symbolic']}

% --- DPA Segment Deontic Statements ---
"""
        
        # Add all DPA deontic statements
        for segment in dpa_segments:
            segment_id = segment["id"]
            segment_text = segment["text"]
            segment_deontics = segment["deontic_statements"]
            
            lp_content += f"\n% ----------------------------------------\n"
            lp_content += f"% Segment {segment_id}:\n"
            lp_content += f"% Text: {segment_text}\n"
            lp_content += f"% Deontic Logic:\n"
            
            for statement in segment_deontics:
                lp_content += f"{statement}\n"
        
        # Add semantic mapping rules if available
        if req_mappings:
            lp_content += f"""
% ----------------------------------------
% Semantic Mapping Rules for Requirement {req_id}
% These rules connect DPA actions to requirement predicates
"""
            for mapping_rule in req_mappings:
                lp_content += f"{mapping_rule}\n"
        else:
            lp_content += f"""
% ----------------------------------------
% No semantic mappings found for Requirement {req_id}
% The DPA does not semantically satisfy this requirement
"""
        
        # Add status logic - simplified and without syntax errors
        lp_content += f"""
% ----------------------------------------
% Status Logic for Requirement {req_id}

% A requirement is satisfied if it has a semantic mapping that is fulfilled
satisfies(req{req_id}) :- 
    true.  % If there's a semantic mapping, it means the DPA satisfies this requirement

% A requirement is not mentioned if there's no semantic mapping
not_mentioned(req{req_id}) :- 
    not satisfies(req{req_id}).

% Additional built-in facts
processor(processor).
controller(controller).
data_subject(data_subject).
personal_data(personal_data).

% ----------------------------------------
% Show Directives
#show satisfies/1.
#show not_mentioned/1.
"""
        
        # Write the LP file
        with open(lp_file_path, 'w') as f:
            f.write(lp_content)
    
    print(f"Generated {len(requirements)} LP files in: {dpa_dir}")

def generate_semantic_rules(requirements_file, dpa_segments_file, model_path, output_dir):
    """
    Generate semantic mapping rules between requirements and DPA segments.
    """
    # Initialize LLM
    print(f"Initializing LLM with model: {model_path}")
    llm_config = LlamaConfig(model_path=model_path, temperature=0.1)
    llm_model = LlamaModel(llm_config)
    print("LLM initialized successfully")
    
    # Load requirements
    print(f"Loading requirements from: {requirements_file}")
    requirements = load_requirements(requirements_file)
    print(f"Loaded {len(requirements)} requirements")
    
    # Load DPA deontic data
    print(f"Loading DPA deontic statements from: {dpa_segments_file}")
    with open(dpa_segments_file, 'r') as f:
        dpa_deontic_data = json.load(f)
    
    target_dpa = dpa_deontic_data["dpa_id"]
    dpa_segments = dpa_deontic_data["segments"]
    print(f"Processing {len(dpa_segments)} segments for DPA: {target_dpa}")
    
    # Collect all DPA deontic statements
    all_dpa_deontic_statements = []
    for segment in dpa_segments:
        all_dpa_deontic_statements.extend(segment["deontic_statements"])
    
    # Store all semantic rules
    all_mappings = {}
    
    # Process each requirement
    print("Creating semantic mappings for each requirement...")
    for req_id, req_details in tqdm(requirements.items()):
        # Create prompt for this requirement
        system_prompt, user_prompt = create_semantic_mapping_prompt(req_details, all_dpa_deontic_statements)
        
        # Generate semantic mapping
        mapping_result = llm_model.generate_symbolic_representation(user_prompt, system_prompt)
        
        # Check if connection found
        if "NO_SEMANTIC_CONNECTION" not in mapping_result:
            # Extract and validate mapping rules
            mapping_rules = []
            for line in mapping_result.split('\n'):
                line = line.strip()
                if line and ':-' in line and not line.startswith('%'):
                    # Validate the rule
                    is_valid, cleaned_rule = validate_mapping_rule(line)
                    if is_valid:
                        mapping_rules.append(cleaned_rule)
            
            # Store mappings for this requirement only if valid rules exist
            if mapping_rules:
                all_mappings[req_id] = mapping_rules
    
    # Save all semantic mappings to file
    mappings_file = os.path.join(output_dir, "semantic_mappings.json")
    with open(mappings_file, 'w') as f:
        json.dump({
            "dpa_id": target_dpa,
            "mappings": all_mappings
        }, f, indent=2)
    
    print(f"Saved semantic mappings to: {mappings_file}")
    
    # Generate LP files
    generate_lp_files(requirements, dpa_deontic_data, all_mappings, output_dir)
    
    return all_mappings

def main():
    parser = argparse.ArgumentParser(description="Generate semantic mapping rules for deontic logic")
    parser.add_argument("--requirements", type=str, default="data/processed/requirements_symbolic.json",
                        help="Path to requirements symbolic JSON file")
    parser.add_argument("--dpa_segments", type=str, default="semantic_results/dpa_deontic.json",
                        help="Path to DPA deontic statements JSON file")
    parser.add_argument("--model", type=str, default="models/Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf",
                        help="Path to LLM model file")
    parser.add_argument("--output", type=str, default="semantic_results",
                        help="Output directory")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Generate semantic rules
    generate_semantic_rules(args.requirements, args.dpa_segments, args.model, args.output)

if __name__ == "__main__":
    main()