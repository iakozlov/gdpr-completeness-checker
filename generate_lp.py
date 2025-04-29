# generate_lp.py
import os
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
    
    # Key action patterns to look for
    action_patterns = {
        "impose_obligations": ["impose", "obligation", "same obligation"],
        "engage_sub_processor": ["engage", "sub-processor", "subprocessor"],
        "process_personal_data": ["process", "personal data"],
        "obtain_authorization": ["obtain", "authorization", "authorisation"],
        "make_available": ["make available", "information", "provide information"],
        "allow_audit": ["audit", "inspection", "contribute to audit"],
        "assist_controller": ["assist", "controller", "respond to request"],
        "ensure_security": ["security", "secure", "measures"],
        "notify_breach": ["notify", "breach", "personal data breach"],
        "return_delete_data": ["return", "delete", "personal data"]
    }
    
    # Find the best matching action
    for act_name, keywords in action_patterns.items():
        if any(keyword in combined_text for keyword in keywords):
            action = act_name
            break
    
    # Add appropriate arguments based on the action
    if "(" not in action and ")" not in action:
        if action == "impose_obligations":
            action = "impose_obligations(processor, sub_processor)"
        elif action == "engage_sub_processor":
            action = "engage_sub_processor(processor, sub_processor)"
        elif action == "process_personal_data":
            action = "process_personal_data(processor)"
        elif action == "make_available":
            action = "make_available(processor, information)"
        elif action == "assist_controller":
            action = "assist_controller(processor, controller)"
        elif action == "allow_audit":
            action = "allow_audit(processor, controller)"
        else:
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

def generate_lp_files(requirements, dpa_translations, output_dir):
    """Generate .lp files for each DPA-requirement pair."""
    
    # Create output directories for each DPA
    for dpa_id in dpa_translations.keys():
        dpa_dir = os.path.join(output_dir, f"dpa_{dpa_id}")
        os.makedirs(dpa_dir, exist_ok=True)
        
        # Get DPA segments
        segments = dpa_translations[dpa_id]
        
        # Process each requirement separately
        for req_id, req_info in requirements.items():
            # Create .lp file for this requirement
            lp_file_path = os.path.join(dpa_dir, f"req_{req_id}.lp")
            
            with open(lp_file_path, "w") as f:
                # Convert requirement to predicate format
                req_predicates = convert_to_predicate_format(
                    req_info["symbolic"], f"req{req_id}", req_info["text"], "requirement")
                
                # Add header
                f.write(f"""% Deolingo program to check DPA against Requirement {req_id}
% ==========================================================

% --- Requirement Definition ---
% Represents the core content of requirement {req_id}.

{os.linesep.join(req_predicates)}

% --- DPA Segment Content Representation ---
% Represents what the DPA explicitly states using these facts.

""")
                
                # Add DPA segments
                for segment in segments:
                    dpa_predicates = convert_to_predicate_format(
                        segment["symbolic"], f"dpa{segment['segment_id']}", 
                        segment["text"], "dpa")
                    
                    f.write(f"% DPA Segment {segment['segment_id']}: {segment['text'][:50]}...\n")
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
    dpa_states(_, M, A) % Check if DPA states the exact same modality and action

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