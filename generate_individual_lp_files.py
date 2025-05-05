# generate_individual_lp_files.py - Generate semantic connection based LP files
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
    
    # Process requirements into a more structured format - ONLY Article 32 security requirement
    processed_requirements = {}
    
    # Look for the specific requirement about Article 32 security
    target_requirement = "The processor shall take all measures required pursuant to Article 32 or to ensure the security of processing."
    
    for req_text, symbolic in requirements_data.items():
        # Match the target requirement
        if req_text.strip() == target_requirement:
            # This is requirement 6 (the Article 32 security requirement)
            req_id = '6'
            
            # Store processed requirement with corrected modality
            processed_requirements[req_id] = {
                "text": req_text,
                "symbolic": symbolic,
                "modality": 'obligatory',  # This requirement is obligatory (must take measures)
                "action": 'take_measures_security_of_processing',
                "conditions": []
            }
            break  # We found our target requirement
    
    return processed_requirements

def create_semantic_mapping_prompt(requirement_details, dpa_segment):
    """Create a prompt for generating semantic connection rules."""
    req_text = requirement_details["text"]
    req_symbolic = requirement_details["symbolic"]
    
    # Extract requirement's deontic statements
    req_deontic_parts = []
    for line in req_symbolic.split('\n'):
        if any(op in line for op in ['&obligatory', '&permitted', '&forbidden']) and line.strip():
            req_deontic_parts.append(line.strip())
    
    req_deontic_text = "\n".join(req_deontic_parts) if req_deontic_parts else req_symbolic
    
    system_prompt = """
You are a specialized AI for legal analysis. Your task is to identify semantic connections 
between predicates in regulatory requirements and DPA segments.

Create semantic connection rules when the textual forms of statements mean the same thing in context.

EXAMPLE SEMANTIC CONNECTION:
Requirement: The processor shall assist the controller in fulfilling its obligation to respond to requests for exercising the data subject's rights
DPA: will redirect such requesting entity to request data directly from controller and promptly notify controller

SEMANTIC CONNECTION RULES:
assist_controller(processor, respond_to_data_subject_requests) :- redirect_requests(processor, controller), notify_controller(processor, request), provide_request_copy(processor, controller).

RULES FOR CREATING MAPPINGS:
1. Use standard ASP syntax: head :- body1, body2.
2. Head should be a predicate from the requirement symbolic representation
3. Body should contain predicates that express the same meaning as the DPA segment
4. When using 'not', do NOT put predicates in parentheses and use comma (,) for conjunction
5. For disjunction, use semicolon (;) with proper spacing
6. Extract meaningful actions from both requirement and DPA text
7. Do NOT number the rules (no 1., 2., 3., etc.)
8. Create connection rules that show how DPA actions fulfill requirement actions
"""

    user_prompt = f"""
REQUIREMENT TO ANALYZE:
Text: {req_text}
Deontic Logic: {req_deontic_text}

DPA SEGMENT TO CHECK:
Text: {dpa_segment}

Analyze if the DPA segment provides actions that semantically satisfy the requirement.
Create connection rules that link DPA actions to requirement actions.

Rules format:
requirement_action :- dpa_action1, dpa_action2.

If no semantic connection exists, respond with: "NO_SEMANTIC_CONNECTION"
"""
    
    return system_prompt, user_prompt

def clean_semantic_mapping_rule(rule):
    """Clean a semantic mapping rule to ensure valid ASP syntax."""
    rule = rule.strip()
    
    # Remove numbering from rules (e.g., "1. " or "2. ")
    rule = re.sub(r'^\d+\.\s*', '', rule)
    
    # Fix negation syntax - remove parentheses after not and fix conjunction
    if ':-' in rule:
        parts = rule.split(':-')
        head = parts[0].strip()
        body = parts[1].strip()
        
        # Fix negation with disjunction: "not A | B" should be "not A, not B"
        if 'not ' in body and '|' in body:
            # Replace | with , for conjunction when used with not
            body = body.replace('|', ',')
        
        # Remove parentheses from negated predicates
        body = re.sub(r'not\s*\(([^)]+)\)', r'not \1', body)
        
        rule = f"{head} :- {body}"
    
    # Ensure proper ending with period
    if not rule.endswith('.'):
        rule += '.'
    
    return rule

def generate_individual_lp_files(requirements_file, dpa_segments_file, model_path, output_dir):
    """Generate individual LP files for each requirement-DPA pair."""
    
    # Initialize LLM
    print(f"Initializing LLM with model: {model_path}")
    llm_config = LlamaConfig(model_path=model_path, temperature=0.1)
    llm_model = LlamaModel(llm_config)
    print("LLM initialized successfully")
    
    # Load requirements - ONLY Article 32 security requirement
    print(f"Loading requirements from: {requirements_file}")
    requirements = load_requirements(requirements_file)
    print(f"Loaded {len(requirements)} requirements")
    
    # Verify we found our target requirement
    if '6' not in requirements:
        print("Error: Article 32 security requirement not found in requirements file")
        return
    
    # Load DPA deontic data
    print(f"Loading DPA deontic statements from: {dpa_segments_file}")
    with open(dpa_segments_file, 'r') as f:
        dpa_deontic_data = json.load(f)
    
    target_dpa = dpa_deontic_data["dpa_id"]
    dpa_segments = dpa_deontic_data["segments"]
    print(f"Processing {len(dpa_segments)} segments for DPA: {target_dpa}")
    
    # Create output directory for each requirement-segment pair
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate LP files for requirement 6 only
    print("Generating LP files for Article 32 security requirement only...")
    
    req_id = '6'
    req_details = requirements[req_id]
    
    # Get requirement details
    req_action = req_details.get('action', 'take_measures_security_of_processing')
    req_modality = req_details.get('modality', 'obligatory')
    
    # Create directory for this requirement
    req_dir = os.path.join(output_dir, f"req_{req_id}")
    os.makedirs(req_dir, exist_ok=True)
    
    for segment in tqdm(dpa_segments):
        segment_id = segment["id"]
        segment_text = segment["text"]
        segment_deontics = segment["deontic_statements"]
        
        # Create LP file for this requirement-segment pair
        lp_file_path = os.path.join(req_dir, f"dpa_segment_{segment_id}.lp")
        
        # Generate semantic mapping for this specific pair
        system_prompt, user_prompt = create_semantic_mapping_prompt(req_details, segment_text)
        
        try:
            mapping_result = llm_model.generate_symbolic_representation(user_prompt, system_prompt)
            semantic_mapping = None
            
            if "NO_SEMANTIC_CONNECTION" not in mapping_result:
                # Extract mapping rules
                mapping_rules = []
                for line in mapping_result.split('\n'):
                    line = line.strip()
                    if line and ':-' in line and not line.startswith('%'):
                        # Clean the mapping rule (remove numbering and fix syntax)
                        cleaned_rule = clean_semantic_mapping_rule(line)
                        mapping_rules.append(cleaned_rule)
                
                if mapping_rules:
                    semantic_mapping = mapping_rules
        except Exception as e:
            print(f"Error generating semantic mapping for Req {req_id}, Segment {segment_id}: {e}")
            semantic_mapping = None
        
        # Build LP file content
        lp_content = f"""% ========================================================================
% LP file for Requirement {req_id} and DPA Segment {segment_id}
% ========================================================================

% --- Requirement Definitions ---
% Assign IDs and represent the core content of requirement {req_id}.

requirement(req{req_id}). % Requirement {req_id} ID
req_modality(req{req_id}, {req_modality}).
req_action(req{req_id}, {req_action}).

% --- DPA Segment Content Representation ---
% Represent what the DPA explicitly states using these facts.

"""
        
        # Add DPA actions extracted from deontic statements
        dpa_action_id = 1
        for statement in segment_deontics:
            # Extract action from deontic statement
            if '&obligatory' in statement or '&permitted' in statement or '&forbidden' in statement:
                action_match = re.search(r'{([^}]+)}', statement)
                if action_match:
                    action = action_match.group(1)
                    modality = 'obligatory' if '&obligatory' in statement else 'permitted' if '&permitted' in statement else 'forbidden'
                    lp_content += f"dpa_states(dpa{dpa_action_id}, {modality}, {action}).\n"
                    dpa_action_id += 1
        
        lp_content += f"""
% --- Additional DPA Context ---
processor(processor).
controller(controller).
data_subject(data_subject).
personal_data(personal_data).

% --- Semantic Connection Rules ---
% These rules help establish semantic connections between different actions
"""
        
        # Add semantic mapping if exists
        if semantic_mapping:
            for mapping_rule in semantic_mapping:
                lp_content += f"{mapping_rule}\n"
        else:
            lp_content += f"% No semantic connections found for Requirement {req_id} and Segment {segment_id}\n"
        
        lp_content += f"""
% --- Matching and Status Logic ---

% Helper: Does the DPA mention a specific action with any modality?
dpa_mentions_action(Action) :- dpa_states(_, _, Action).

% Status Rule 1: Satisfaction
% A requirement R is satisfied if the action is true based on semantic connections
% First check if action is satisfied through DPA states with same modality
satisfies(R) :-
    requirement(R),
    req_modality(R, M),
    req_action(R, A),
    dpa_states(_, M, A).

% Or if action is true through semantic connection rules
satisfies(R) :-
    requirement(R),
    req_modality(R, obligatory),
    req_action(R, take_measures_security_of_processing),
    take_measures_security_of_processing.

% For other requirements with different actions, we need similar rules
% This is a simplification for this specific requirement

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
    
    print(f"Generated LP files for Article 32 security requirement only")
    return {req_id: req_details}, dpa_segments

def main():
    parser = argparse.ArgumentParser(description="Generate individual LP files for Article 32 security requirement only")
    parser.add_argument("--requirements", type=str, default="data/processed/requirements_symbolic.json",
                        help="Path to requirements symbolic JSON file")
    parser.add_argument("--dpa_segments", type=str, default="semantic_results/dpa_deontic.json",
                        help="Path to DPA deontic statements JSON file")
    parser.add_argument("--model", type=str, default="models/Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf",
                        help="Path to LLM model file")
    parser.add_argument("--output", type=str, default="semantic_results/individual_lp_files",
                        help="Output directory for individual LP files")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Generate individual LP files for Article 32 security requirement only
    generate_individual_lp_files(args.requirements, args.dpa_segments, args.model, args.output)

if __name__ == "__main__":
    main()