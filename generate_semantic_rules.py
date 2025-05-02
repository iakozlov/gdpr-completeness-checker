# generate_semantic_rules.py
import os
import json
import argparse
import re
from tqdm import tqdm
from models.llm import LlamaModel
from config.llm_config import LlamaConfig

def load_requirements(requirements_file):
    """
    Load requirements from JSON file with symbolic representations.
    
    Args:
        requirements_file: Path to requirements symbolic JSON file
        
    Returns:
        Dictionary mapping requirement IDs to their details
    """
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
        
        # Extract deontic operator
        modality = "obligatory"  # Default
        if "&obligatory" in symbolic:
            modality = "obligatory"
        elif "&permitted" in symbolic:
            modality = "permitted"
        elif "&forbidden" in symbolic:
            modality = "forbidden"
        
        # Extract action
        action = None
        action_pattern = r'&(?:obligatory|permitted|forbidden){([^}]+)}'
        action_matches = re.findall(action_pattern, symbolic)
        if action_matches:
            action = action_matches[0].strip()
        
        # Store processed requirement
        processed_requirements[req_id] = {
            "text": req_text,
            "symbolic": symbolic,
            "modality": modality,
            "action": action if action else "generic_action(processor)"
        }
    
    return processed_requirements

def create_batch_semantic_mapping_prompt(requirement_details, all_segments):
    """
    Create a prompt for batch generating semantic mapping rules for a requirement 
    against multiple DPA segments.
    
    Args:
        requirement_details: Dictionary with requirement details
        all_segments: Dictionary with all DPA segment info
        
    Returns:
        Prompt for the LLM to generate semantic mapping rules in batch
    """
    # Extract key information about the requirement
    req_text = requirement_details["text"]
    req_modality = requirement_details["modality"]
    req_action = requirement_details["action"]
    
    # Format all segments for the batch processing
    segments_text = ""
    for segment_id, segment_info in all_segments.items():
        segment_text = segment_info["text"]
        segment_actions = segment_info["actions"]
        formatted_actions = "\n".join(segment_actions)
        
        segments_text += f"SEGMENT ID: {segment_id}\n"
        segments_text += f"TEXT: {segment_text}\n"
        segments_text += f"ACTIONS:\n{formatted_actions}\n\n"
    
    system_prompt = """
You are a specialized AI for legal analysis. Your task is to identify semantic connections between a regulatory requirement (GDPR) and multiple Data Processing Agreement (DPA) segments.

Rather than looking for exact vocabulary matches, you need to determine if any of the DPA segments semantically satisfy the requirement, even if different terminology is used.

You will analyze all provided segments and create semantic mapping rules only for those that have a genuine connection to the requirement. For each segment that matches:

1. Analyze if the segment semantically covers any aspect of the requirement's intent
2. Generate a logical rule that connects the actions in that segment to the requirement
3. Only create connection rules for segments with a genuine semantic relationship
4. When a segment has no connection, you should exclude it from your rules

Use Answer Set Programming (ASP) syntax for your rules.
"""

    user_prompt = f"""
REQUIREMENT TO ANALYZE:
Text: {req_text}
Deontic Modality: {req_modality}
Action: {req_action}

DPA SEGMENTS TO CHECK FOR SEMANTIC MATCHES:
{segments_text}

Generate semantic connection rules ONLY for segments that genuinely match the requirement semantically. For each matching segment, provide a rule in this format:

```
% Rule for Segment [ID]: [brief explanation of the semantic connection]
{req_action} :- 
    [dpa_action1],
    [dpa_action2].
```

IMPORTANT INSTRUCTIONS:
- Only create rules for segments with a true semantic connection to the requirement
- Skip segments with no semantic relationship to the requirement
- Be precise about which actions from a segment contribute to satisfying the requirement
- If you find multiple segments that together satisfy the requirement, you may create a rule that combines their actions
- If NO segments have a semantic match with the requirement, respond with just: "No semantic connections found"

Your response should contain ONLY the semantic connection rules or the "No semantic connections found" message.
"""
    
    return system_prompt, user_prompt

def extract_semantic_rules_from_response(response_text, requirement_action):
    """
    Extract and validate semantic rules from the LLM response.
    
    Args:
        response_text: Text response from the LLM
        requirement_action: The action from the requirement
        
    Returns:
        List of valid semantic rules
    """
    # Clean up the response
    cleaned_response = response_text.strip()
    
    # Check if no connections were found
    if "No semantic connections found" in cleaned_response:
        return []
    
    # Extract rules (each rule should start with %)
    rules = []
    current_rule = ""
    
    for line in cleaned_response.split('\n'):
        line = line.strip()
        
        # Skip empty lines and code block markers
        if not line or line == '```':
            continue
            
        # Start of a new rule
        if line.startswith('%'):
            if current_rule:  # Save previous rule if exists
                rules.append(current_rule)
            current_rule = line
        elif current_rule:  # Continue current rule
            current_rule += '\n' + line
    
    # Add the last rule if exists
    if current_rule:
        rules.append(current_rule)
    
    # Validate each rule to ensure it has the requirement action
    valid_rules = []
    for rule in rules:
        # Check if the rule references the requirement action
        if requirement_action in rule:
            valid_rules.append(rule)
    
    return valid_rules

def generate_semantic_rules(requirements_file, dpa_segments_file, model_path, output_dir):
    """
    Generate semantic mapping rules between requirements and DPA segments.
    Optimized to use only one LLM call per requirement.
    
    Args:
        requirements_file: Path to requirements symbolic JSON file
        dpa_segments_file: Path to DPA segments with actions JSON file
        model_path: Path to LLM model file
        output_dir: Directory to store output LP files
    """
    # Initialize LLM with optimized settings for fewer calls
    print(f"Initializing LLM with model: {model_path}")
    llm_config = LlamaConfig(
        model_path=model_path, 
        temperature=0.1,
        n_ctx=50000  # Increase context window to handle more segments at once
    )
    llm_model = LlamaModel(llm_config)
    print("LLM initialized successfully")
    
    # Load requirements
    print(f"Loading requirements from: {requirements_file}")
    requirements = load_requirements(requirements_file)
    print(f"Loaded {len(requirements)} requirements")
    
    # Load DPA segments
    print(f"Loading DPA segments from: {dpa_segments_file}")
    with open(dpa_segments_file, 'r') as f:
        dpa_data = json.load(f)
    
    # Extract target DPA information
    target_dpa = dpa_data["dpa_id"]
    segments = dpa_data["segments"]
    print(f"Processing {len(segments)} segments for DPA: {target_dpa}")
    
    # Create output directory for this DPA
    dpa_dir = os.path.join(output_dir, f"dpa_{target_dpa.replace(' ', '_')}")
    os.makedirs(dpa_dir, exist_ok=True)
    
    # Store semantic rules for each requirement
    semantic_rules_by_req = {}
    
    # Process each requirement in a single batch operation
    print("Creating semantic mappings for each requirement (one LLM call per requirement)...")
    for req_id, req_details in tqdm(requirements.items(), total=len(requirements)):
        # Create batch prompt for all segments
        system_prompt, user_prompt = create_batch_semantic_mapping_prompt(req_details, segments)
        
        # Generate semantic mappings for all segments at once
        mapping_result = llm_model.generate_symbolic_representation(user_prompt, system_prompt)
        
        # Extract and validate the semantic rules
        req_semantic_rules = extract_semantic_rules_from_response(
            mapping_result, req_details["action"]
        )
        
        # Store rules for this requirement
        semantic_rules_by_req[req_id] = req_semantic_rules
    
    # Save semantic rules to file
    rules_file = os.path.join(output_dir, "semantic_rules.json")
    with open(rules_file, 'w') as f:
        json.dump({
            "dpa_id": target_dpa,
            "semantic_rules": semantic_rules_by_req
        }, f, indent=2)
    
    print(f"Saved semantic rules to: {rules_file}")
    return semantic_rules_by_req

def main():
    parser = argparse.ArgumentParser(description="Generate semantic mapping rules (optimized)")
    parser.add_argument("--requirements", type=str, default="data/processed/requirements_symbolic.json",
                        help="Path to requirements symbolic JSON file")
    parser.add_argument("--dpa_segments", type=str, default="semantic_results/dpa_symbolic.json",
                        help="Path to DPA segments with actions JSON file")
    parser.add_argument("--model", type=str, default="models/Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf",
                        help="Path to LLM model file")
    parser.add_argument("--output", type=str, default="semantic_results",
                        help="Output directory")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Generate semantic rules with optimized approach
    generate_semantic_rules(args.requirements, args.dpa_segments, args.model, args.output)

if __name__ == "__main__":
    main()