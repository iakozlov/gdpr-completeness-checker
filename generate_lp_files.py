# generate_lp_files.py
import os
import json
import argparse
import re

def extract_modality_and_action(symbolic_repr, original_text):
    """Extract modality and action from symbolic representation and original text."""
    # Default values
    modality = "obligatory"
    action = "generic_action"
    
    # Try to extract from symbolic representation
    if "&obligatory" in symbolic_repr or "O(" in symbolic_repr or "o(" in symbolic_repr:
        modality = "obligatory"
    elif "&permitted" in symbolic_repr or "P(" in symbolic_repr or "p(" in symbolic_repr:
        modality = "permitted"
    elif "&forbidden" in symbolic_repr or "F(" in symbolic_repr or "f(" in symbolic_repr:
        modality = "forbidden"
    
    # Extract action name - look for patterns in both symbolic repr and original text
    combined_text = (symbolic_repr + " " + original_text).lower()
    
    # Add appropriate arguments based on the action
    if "(" not in action and ")" not in action:
        action = f"{action}(processor)"
    
    return modality, action

def convert_to_predicate_format(symbolic_repr, entity_id, original_text, entity_type="requirement"):
    """Convert to predicate-based format used in the ASP template."""
    modality, action = extract_modality_and_action(symbolic_repr, original_text)
    
    if entity_type == "requirement":
        statements = [
            f"requirement({entity_id}). % Requirement {entity_id} ID",
            f"req_modality({entity_id}, {modality}).",
            f"req_action({entity_id}, {action})."
        ]
    else:  # dpa
        statements = [
            f"dpa_states({entity_id}, {modality}, {action}). % DPA statement about {action}"
        ]
    
    return statements

def main():
    parser = argparse.ArgumentParser(description="Generate LP files from symbolic representations")
    parser.add_argument("--requirements", type=str, default="data/processed/requirements_symbolic.json",
                        help="Path to requirements symbolic JSON file")
    parser.add_argument("--dpa", type=str, default="data/processed/dpa_segments_symbolic.json",
                        help="Path to DPA segments symbolic JSON file")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory for LP files")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load requirements
    print(f"Loading requirements from: {args.requirements}")
    with open(args.requirements, 'r') as f:
        requirements_data = json.load(f)
    
    # Load DPA segments
    print(f"Loading DPA segments from: {args.dpa}")
    with open(args.dpa, 'r') as f:
        dpa_data = json.load(f)
    
    # Extract information
    dpa_id = dpa_data["dpa_id"]
    segments = dpa_data["segments"]
    
    # Create directory for this DPA
    dpa_dir = os.path.join(args.output, f"dpa_{dpa_id.replace(' ', '_')}")
    os.makedirs(dpa_dir, exist_ok=True)
    
    # Identify numeric IDs in the requirements text
    req_id_map = {}
    for req_text in requirements_data.keys():
        match = re.match(r'.*?(\d+).*', req_text)
        if match:
            req_id = match.group(1)
            req_id_map[req_text] = req_id
        else:
            # If no number found, assign sequential ID
            req_id_map[req_text] = str(len(req_id_map) + 1)
    
    # Process each requirement
    print(f"Generating LP files for DPA: {dpa_id}")
    for req_text, req_symbolic in requirements_data.items():
        req_id = req_id_map[req_text]
        
        # Create LP file
        lp_file_path = os.path.join(dpa_dir, f"req_{req_id}.lp")
        
        # Generate requirement predicates
        req_predicates = convert_to_predicate_format(req_symbolic, f"req{req_id}", req_text, "requirement")
        
        with open(lp_file_path, "w") as f:
            # Add header
            f.write(f"""% Deolingo program to check DPA against Requirement {req_id}
% ==========================================================

% --- Requirement Definition ---
% Original text: {req_text}
% Symbolic representation: {req_symbolic}

{os.linesep.join(req_predicates)}

% --- DPA Segment Content Representation ---
% Represents what the DPA explicitly states using these facts.

""")
            
            # Add DPA segments
            for segment_text, segment_info in segments.items():
                segment_id = segment_info["id"]
                segment_symbolic = segment_info["symbolic"]
                
                dpa_predicates = convert_to_predicate_format(
                    segment_symbolic, f"dpa{segment_id}", 
                    segment_text, "dpa")
                
                f.write(f"% DPA Segment {segment_id}: {segment_text[:50]}...\n")
                f.write(os.linesep.join(dpa_predicates) + "\n\n")
            
            # Add matching logic
            f.write("""
% --- Matching and Status Logic ---

% Helper: Does the DPA mention a specific action with any modality?
dpa_mentions_action(Action) :- dpa_states(_, _, Action).

% Status Rule 1: Satisfaction
% A requirement R is satisfied if the DPA states the same modality M for the same action A.
satisfies(R) :-
    requirement(R),
    req_modality(R, M),
    req_action(R, A),
    dpa_states(_, M, A). % Check if DPA states the exact same modality and action

% Status Rule 2: Violation
% A requirement R is violated if the DPA states a conflicting modality for the same action A.
violates(R) :-
    requirement(R),
    req_modality(R, obligatory), % Requirement is Obligatory
    req_action(R, A),
    dpa_states(_, forbidden, A).   % DPA Forbids it

violates(R) :-
    requirement(R),
    req_modality(R, permitted),  % Requirement is Permitted
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
""")
    
    print(f"Generated {len(requirements_data)} LP files in: {dpa_dir}")

if __name__ == "__main__":
    main()