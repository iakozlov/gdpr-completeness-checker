# generate_lp_files_semantic.py
import os
import json
import argparse
import re

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

def extract_action_from_predicate(predicate):
    """
    Extract action and modality from a dpa_states predicate.
    
    Args:
        predicate: dpa_states predicate string
        
    Returns:
        Tuple of (id, modality, action)
    """
    # Default values
    dpa_id = "dpa1"
    modality = "obligatory"
    action = "generic_action(processor)"
    
    # Try to parse the predicate
    pattern = r'dpa_states\(([^,]+),\s*([^,]+),\s*([^)]+)\)'
    match = re.match(pattern, predicate)
    
    if match:
        dpa_id = match.group(1).strip()
        modality = match.group(2).strip()
        action = match.group(3).strip()
    
    return dpa_id, modality, action

def generate_lp_files(requirements_file, dpa_segments_file, semantic_rules_file, output_dir):
    """
    Generate LP files with semantic rules for each requirement.
    
    Args:
        requirements_file: Path to requirements symbolic JSON file
        dpa_segments_file: Path to DPA segments with actions JSON file
        semantic_rules_file: Path to semantic rules JSON file
        output_dir: Directory to store output LP files
    """
    # Load requirements
    print(f"Loading requirements from: {requirements_file}")
    requirements = load_requirements(requirements_file)
    print(f"Loaded {len(requirements)} requirements")
    
    # Load DPA segments
    print(f"Loading DPA segments from: {dpa_segments_file}")
    with open(dpa_segments_file, 'r') as f:
        dpa_data = json.load(f)
    
    # Load semantic rules
    print(f"Loading semantic rules from: {semantic_rules_file}")
    with open(semantic_rules_file, 'r') as f:
        rules_data = json.load(f)
    
    # Extract target DPA information
    target_dpa = dpa_data["dpa_id"]
    segments = dpa_data["segments"]
    semantic_rules = rules_data["semantic_rules"]
    
    print(f"Processing {len(requirements)} requirements for DPA: {target_dpa}")
    
    # Create output directory for this DPA
    dpa_dir = os.path.join(output_dir, f"dpa_{target_dpa.replace(' ', '_')}")
    os.makedirs(dpa_dir, exist_ok=True)
    
    # Process each requirement
    for req_id, req_details in requirements.items():
        # Create LP file for this requirement
        lp_file_path = os.path.join(dpa_dir, f"req_{req_id}.lp")
        
        # Start building the LP file content
        lp_content = f"""% Deolingo program to check DPA text against Requirement {req_id}
% ==========================================================

% --- Requirement Definitions ---
% Assign IDs and represent the core content of each requirement rule.

requirement(req{req_id}). % Requirement {req_id} ID
req_modality(req{req_id}, {req_details['modality']}).
req_action(req{req_id}, {req_details['action']}).

% --- DPA Segment Content Representation ---
% Represent what the DPA explicitly states using these facts.

"""
        
        # Add all DPA actions
        processed_actions = set()  # To avoid duplicates
        
        for segment_id, segment_info in segments.items():
            segment_text = segment_info["text"]
            segment_actions = segment_info["actions"]
            
            lp_content += f"% DPA Segment {segment_id}: {segment_text[:80]}...\n"
            
            for action_pred in segment_actions:
                # Extract action components
                dpa_id, modality, action = extract_action_from_predicate(action_pred)
                
                # Create a unique identifier for this action
                action_id = f"{dpa_id}_{modality}_{action}"
                
                # Only add if not already processed
                if action_id not in processed_actions:
                    lp_content += f"{action_pred}.\n"
                    processed_actions.add(action_id)
            
            lp_content += "\n"
        
        # Add default facts for testing
        lp_content += """% --- Additional DPA Context ---
% Default facts for testing purposes
processor(processor).
sub_processor(sub_processor).
controller(controller).
data_subject(data_subject).
personal_data(personal_data).

"""
        
        # Add semantic mapping rules
        lp_content += "% --- Semantic Connection Rules ---\n"
        lp_content += "% These rules help establish semantic connections between different actions\n\n"
        
        # Get rules specific to this requirement
        req_rules = semantic_rules.get(req_id, [])
        
        if req_rules:
            for rule in req_rules:
                lp_content += f"{rule}\n\n"
        else:
            lp_content += "% No semantic connections found between this requirement and any DPA segment\n\n"
        
        # Add matching logic
        lp_content += """% --- Matching and Status Logic ---

% Helper: Does the DPA mention a specific action with any modality?
dpa_mentions_action(Action) :- dpa_states(_, _, Action).

% Status Rule 1: Satisfaction
% A requirement R is satisfied if the DPA states the same modality M for the same action A,
% or if the action is true based on semantic rules.
satisfies(R) :-
    requirement(R),
    req_modality(R, M),
    req_action(R, A),
    A.  % Check if the action is true based on the rules

% Alternative satisfaction through direct action match
satisfies(R) :-
    requirement(R),
    req_modality(R, M),
    req_action(R, A),
    dpa_states(_, M, A).  % Direct match in DPA statements

% Status Rule 2: Violation
% A requirement R is violated if the DPA states a conflicting modality for the same action A.
violates(R) :-
    requirement(R),
    req_modality(R, obligatory), % Requirement is Obligatory
    req_action(R, A),
    dpa_states(_, forbidden, A).   % DPA Forbids it

% Status Rule 3: Non-Mention
% A requirement R is not mentioned if it's neither satisfied nor violated by the DPA text.
not_mentioned(R) :-
    requirement(R),
    not satisfies(R),
    not violates(R).

% --- Show Directives ---
% Display the status for each Requirement ID.
#show satisfies/1.
#show violates/1.
#show not_mentioned/1.
"""
        
        # Write the LP file
        with open(lp_file_path, 'w') as f:
            f.write(lp_content)
        
    print(f"Generated LP files for {len(requirements)} requirements in: {dpa_dir}")

def main():
    parser = argparse.ArgumentParser(description="Generate LP files with semantic rules")
    parser.add_argument("--requirements", type=str, default="data/processed/requirements_symbolic.json",
                        help="Path to requirements symbolic JSON file")
    parser.add_argument("--dpa_segments", type=str, default="semantic_results/dpa_symbolic.json",
                        help="Path to DPA segments with actions JSON file")
    parser.add_argument("--semantic_rules", type=str, default="semantic_results/semantic_rules.json",
                        help="Path to semantic rules JSON file")
    parser.add_argument("--output", type=str, default="semantic_results",
                        help="Output directory for LP files")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Generate LP files
    generate_lp_files(args.requirements, args.dpa_segments, args.semantic_rules, args.output)

if __name__ == "__main__":
    main()