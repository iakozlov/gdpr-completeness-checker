# example_dpa_check.py
import os
import subprocess
import argparse

from models.llm import LlamaModel
from solver.deontic_translator import DeonticTranslator
from config.llm_config import LlamaConfig

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
    
    # Extract action name from symbolic representation and text
    combined_text = (symbolic_repr + " " + original_text).lower()
    
    # Check for specific actions
    if "assist" in combined_text and ("data subject" in combined_text or "request" in combined_text):
        action = "assist_controller(processor, respond_to_data_subject_requests)"
    elif "redirect" in combined_text and "request" in combined_text:
        action = "redirect_requests(processor, controller)"
    elif "notify" in combined_text and "controller" in combined_text:
        action = "notify_controller(processor, request)"
    elif "impose" in combined_text and "obligation" in combined_text:
        action = "impose_obligations(processor, sub_processor)"
    elif "process" in combined_text and "personal data" in combined_text:
        action = "process_personal_data(processor)"
    
    return modality, action

def convert_to_predicate_format(symbolic_repr, entity_id, original_text, entity_type="requirement"):
    """Convert to predicate-based format used in the ASP template."""
    modality, action = extract_modality_and_action(symbolic_repr, original_text)
    
    if entity_type == "requirement":
        statements = [
            f"requirement({entity_id}).",
            f"req_modality({entity_id}, {modality}).",
            f"req_action({entity_id}, {action})."
        ]
    else:  # dpa
        statements = [
            f"dpa_states({entity_id}, {modality}, {action})."
        ]
    
    return statements

def run_dpa_example(model_path=None):
    """Run a complete example of DPA compliance checking with deolingo using our framework."""
    
    # Example texts
    requirement = "The processor shall assist the controller in fulfilling its obligation to respond to requests for exercising the data subject's rights."
    dpa_segment = """In case of such a request, processor or the Sub-Processor will (i) redirect such requesting entity to request data directly from controller and may provide controller's basic contact information, and (ii) promptly notify controller and provide a copy of the request, unless processor is prevented from doing so by applicable laws or governmental order."""
    
    print("=== DPA Compliance Example Using Framework ===")
    print(f"\nRequirement: {requirement}")
    print(f"\nDPA Segment: {dpa_segment}")
    
    # Initialize LLM and translator
    if not model_path:
        model_path = os.environ.get("LLAMA_MODEL_PATH", "models/Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf")
    
    print(f"\nInitializing LLM with model: {model_path}")
    llm_config = LlamaConfig(model_path=model_path, temperature=0.1)
    llm_model = LlamaModel(llm_config)
    translator = DeonticTranslator()
    print("LLM initialized successfully")
    
    # Translate requirement to symbolic representation
    print("\n--- Translating Requirement ---")
    req_output = llm_model.generate_symbolic_representation(requirement)
    print(f"LLM output: {req_output}")
    req_symbolic = translator.translate(req_output)
    print("Symbolic representation:")
    print(req_symbolic)
    
    # Translate DPA segment to symbolic representation
    print("\n--- Translating DPA Segment ---")
    dpa_output = llm_model.generate_symbolic_representation(dpa_segment)
    print(f"LLM output: {dpa_output}")
    dpa_symbolic = translator.translate(dpa_output)
    print("Symbolic representation:")
    print(dpa_symbolic)
    
    # Convert to predicate format
    req_predicates = convert_to_predicate_format(req_symbolic, "req7", requirement, "requirement")
    
    # Create multiple DPA predicates for the different actions in the segment
    combined_text_parts = [
        "redirect requesting entity to request data directly from controller",
        "notify controller and provide a copy of the request"
    ]
    
    dpa_predicates = []
    for i, text_part in enumerate(combined_text_parts, 1):
        # Generate sub-symbolic representation for parts of the DPA segment
        sub_output = llm_model.generate_symbolic_representation(text_part)
        sub_symbolic = translator.translate(sub_output)
        sub_predicates = convert_to_predicate_format(sub_symbolic, f"dpa{i}", text_part, "dpa")
        dpa_predicates.extend(sub_predicates)
    
    # Create the .lp file content - SIMPLIFIED FOR DEOLINGO COMPATIBILITY
    lp_content = f"""% Deolingo program to check DPA text against Requirement rules
% ==========================================================

% --- Requirement Definitions ---
{os.linesep.join(req_predicates)}

% --- DPA Segment Content Representation ---
{os.linesep.join(dpa_predicates)}

% --- Facts ---
% Default facts for testing
processor(processor).
sub_processor(sub_processor).
controller(controller).
request(request).
data_subject_request.
-prevented_by_law(processor).

% --- Action relationships ---
assist_controller(processor, respond_to_data_subject_requests) :- 
    redirect_requests(processor, controller),
    notify_controller(processor, request).

% --- Matching and Status Logic ---
dpa_mentions_action(Action) :- dpa_states(_, _, Action).

% Status Rule 1: Satisfaction
satisfies(R) :-
    requirement(R),
    req_modality(R, M),
    req_action(R, A),
    dpa_states(_, M, A)..

% Status Rule 2: Violation
violates(R) :-
    requirement(R),
    req_modality(R, obligatory),
    req_action(R, A),
    dpa_states(_, forbidden, A).

% Status Rule 3: Non-Mention
not_mentioned(R) :-
    requirement(R),
    not satisfies(R),
    not violates(R).

% --- Show Directives ---
#show satisfies/1.
#show violates/1.
#show not_mentioned/1.
"""
    
    # Save the .lp file
    lp_file = "dpa_example.lp"
    with open(lp_file, "w") as f:
        f.write(lp_content)
    
    print(f"\nSaved .lp file to: {lp_file}")
    
    # Run deolingo
    print("\n=== Running Deolingo Solver ===")
    try:
        # Try to run deolingo command
        result = subprocess.run(
            ["deolingo", lp_file],
            capture_output=True,
            text=True,
            check=False
        )
        
        # Display output
        print("\nDeolingo Output:")
        print("===============")
        print(result.stdout)
        
        if result.stderr:
            print("Errors:")
            print(result.stderr)
            
        # Interpret the result
        if "satisfies(req7)" in result.stdout:
            print("\nCONCLUSION: The DPA segment SATISFIES the requirement")
        elif "violates(req7)" in result.stdout:
            print("\nCONCLUSION: The DPA segment VIOLATES the requirement")
        elif "not_mentioned(req7)" in result.stdout:
            print("\nCONCLUSION: The DPA segment does NOT MENTION the requirement")
        else:
            print("\nCONCLUSION: Unable to determine compliance status")
            
    except Exception as e:
        print(f"Error running deolingo: {e}")
        
    print("\n=== Example Complete ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DPA compliance example")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to LLM model file")
    args = parser.parse_args()
    
    run_dpa_example(args.model)