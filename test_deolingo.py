# test_deolingo.py
import os
import random
import pandas as pd
import re
from models.llm import LlamaModel
from solver.deontic_translator import DeonticTranslator
from config.llm_config import LlamaConfig

def load_requirements(file_path="data/requirements/ground_truth_requirements.txt"):
    """Load requirements from the ground truth file."""
    if not os.path.exists(file_path):
        print(f"Error: Requirements file not found at {file_path}")
        return []
    
    requirements = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Extract requirement text from numbered format like "1. The processor shall..."
            match = re.match(r'^\d+\.\s*(.+)$', line)
            if match:
                requirements.append(match.group(1))
            else:
                requirements.append(line)
    
    return requirements

def load_dpa_segments(file_path="data/train_set.csv"):
    """Load DPA segments from the training set CSV file."""
    if not os.path.exists(file_path):
        print(f"Error: DPA segments file not found at {file_path}")
        return []
    
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Check if 'Sentence' column exists
        if 'Sentence' not in df.columns:
            print(f"Error: 'Sentence' column not found in {file_path}")
            return []
        
        # Extract DPA segments
        dpa_segments = df['Sentence'].dropna().tolist()
        return dpa_segments
        
    except Exception as e:
        print(f"Error loading DPA segments: {e}")
        return []

def extract_modality_and_action(symbolic_repr, original_text):
    """
    Extract modality and action from symbolic representation and original text.
    
    Args:
        symbolic_repr: Symbolic representation from LLM
        original_text: Original text for context
        
    Returns:
        Tuple of (modality, action)
    """
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
    """
    Convert to predicate-based format used in the ASP template.
    
    Args:
        symbolic_repr: Symbolic representation from LLM
        entity_id: ID for the entity (e.g., "req1")
        original_text: Original text for context
        entity_type: Either "requirement" or "dpa"
        
    Returns:
        List of predicate statements
    """
    modality, action = extract_modality_and_action(symbolic_repr, original_text)
    
    if entity_type == "requirement":
        statements = [
            f"requirement({entity_id}). % Requirement {entity_id[3:]} ID",
            f"req_modality({entity_id}, {modality}).",
            f"req_action({entity_id}, {action})."
        ]
    else:  # dpa
        statements = [
            f"dpa_states({entity_id}, {modality}, {action}). % DPA statement about {action}"
        ]
    
    return statements

def main():
    """
    Generate symbolic representations for a random requirement and DPA segment,
    and convert them to the structured ASP format.
    """
    print("=== Generating Structured ASP Program with Random Requirement and DPA Segment ===")
    
    # Load requirements and DPA segments
    requirements = load_requirements()
    dpa_segments = load_dpa_segments()
    
    if not requirements:
        print("No requirements found. Using default requirement.")
        requirements = ["The processor shall not engage a sub-processor without a prior specific or general written authorization of the controller."]
    
    if not dpa_segments:
        print("No DPA segments found. Using default DPA segment.")
        dpa_segments = ["The Processor shall obtain prior written authorization from the Controller before engaging any sub-processor."]
    
    # Select random requirement and DPA segment
    requirement = random.choice(requirements)
    dpa_segment = random.choice(dpa_segments)
    
    print(f"\nSelected Requirement: {requirement}")
    print(f"\nSelected DPA Segment: {dpa_segment}")
    
    # Get model path from environment or use default
    model_path = os.environ.get("LLAMA_MODEL_PATH", "models/Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf")
    
    # Initialize LLM and translator
    try:
        llm_config = LlamaConfig(model_path=model_path, temperature=0.1)
        llm_model = LlamaModel(llm_config)
        print("LLM model initialized successfully")
        translator = DeonticTranslator()
    except Exception as e:
        print(f"Failed to initialize LLM model: {e}")
        return
    
    # Translate requirement
    print("\n--- Translating Requirement ---")
    req_output = llm_model.generate_symbolic_representation(requirement)
    print(f"LLM symbolic representation: {req_output}")
    req_symbolic = translator.translate(req_output)
    print("Deolingo format:")
    print(req_symbolic)
    
    # Translate DPA segment
    print("\n--- Translating DPA Segment ---")
    dpa_output = llm_model.generate_symbolic_representation(dpa_segment)
    print(f"LLM symbolic representation: {dpa_output}")
    dpa_symbolic = translator.translate(dpa_output)
    print("Deolingo format:")
    print(dpa_symbolic)
    
    # Convert to structured ASP format
    req_predicates = convert_to_predicate_format(req_symbolic, "req1", requirement, "requirement")
    dpa_predicates = convert_to_predicate_format(dpa_symbolic, "dpa1", dpa_segment, "dpa")
    
    # Create the ASP program using the template
    output_file = "dpa_compliance.lp"
    with open(output_file, 'w') as f:
        f.write(f"""% Deolingo program to check DPA text against Requirement rules
% ==========================================================

% --- Requirement Definitions ---
% Assign IDs and represent the core content of each requirement rule.

{os.linesep.join(req_predicates)}

% --- DPA Segment Content Representation ---
% Represent what the DPA explicitly states using these facts.

{os.linesep.join(dpa_predicates)}

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
% Conflict examples: req=obligatory vs dpa=forbidden, req=permitted vs dpa=forbidden
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
    
    print(f"\nSaved program to: {output_file}")
    print(f"You can run it manually with: clingo {output_file}")

if __name__ == "__main__":
    main()