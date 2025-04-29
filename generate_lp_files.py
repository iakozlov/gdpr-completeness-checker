# generate_lp_files.py
import os
import json
import argparse
import re
from typing import Tuple, List, Dict

def sanitize_symbolic_repr(symbolic_repr: str) -> str:
    """
    Sanitize a symbolic representation to remove or fix syntax that causes errors.
    
    Args:
        symbolic_repr: The original symbolic representation
        
    Returns:
        A sanitized version with problematic parts removed or fixed
    """
    # Remove the "Facts derived from rule conditions" section entirely
    facts_pattern = r'% Facts derived from rule conditions\s*\n(.*?)(?:\n\n|\n%|\Z)'
    sanitized = re.sub(facts_pattern, '', symbolic_repr, flags=re.DOTALL)
    
    # If the above didn't work, try a more aggressive approach
    if '% Facts derived from rule conditions' in sanitized:
        lines = sanitized.split('\n')
        cleaned_lines = []
        skip_mode = False
        
        for line in lines:
            if '% Facts derived from rule conditions' in line:
                skip_mode = True
                continue
            
            if skip_mode and (line.strip() == '' or line.startswith('%')):
                skip_mode = False
            
            if not skip_mode:
                cleaned_lines.append(line)
        
        sanitized = '\n'.join(cleaned_lines)
    
    return sanitized

def extract_deontic_operator(symbolic_repr: str) -> str:
    """Extract deontic operator from symbolic representation."""
    if "&obligatory" in symbolic_repr or "O(" in symbolic_repr or "o(" in symbolic_repr or "obligation" in symbolic_repr:
        return "obligatory"
    elif "&permitted" in symbolic_repr or "P(" in symbolic_repr or "p(" in symbolic_repr or "permission" in symbolic_repr:
        return "permitted"
    elif "&forbidden" in symbolic_repr or "F(" in symbolic_repr or "f(" in symbolic_repr or "prohibition" in symbolic_repr:
        return "forbidden"
    else:
        # Default to obligatory if no clear indicator is found
        return "obligatory"

def extract_action(symbolic_repr: str, original_text: str) -> str:
    """
    Extract action from symbolic representation and original text
    without relying on hardcoded action patterns.
    """
    # Try to extract from symbolic representation first
    action = None
    
    # Look for action inside deontic operators
    operator_pattern = r'&(?:obligatory|permitted|forbidden){([^}]+)}'
    match = re.search(operator_pattern, symbolic_repr)
    if match:
        action = match.group(1).strip()
    
    # Check for actions in alternative formats
    if not action:
        alt_formats = [
            r'[OoPpFf]\(([^)]+)\)',  # O(), P(), F() format
            r'(?:obligation|permission|prohibition)\(([^)]+)\)'  # obligation(), etc. format
        ]
        
        for pattern in alt_formats:
            match = re.search(pattern, symbolic_repr)
            if match:
                action = match.group(1).strip()
                break
    
    # If still no action, try to extract a meaningful predicate from the text
    if not action:
        # Get the first line of the symbolic repr (typically contains the main rule)
        first_line = symbolic_repr.split('\n')[0] if '\n' in symbolic_repr else symbolic_repr
        
        # Look for words that might be predicates
        words = re.findall(r'[a-zA-Z_]+[a-zA-Z0-9_]*', first_line)
        
        # Filter out common non-action words
        non_actions = {'obligatory', 'permitted', 'forbidden', 'if', 'and', 'or', 'not'}
        potential_actions = [w for w in words if w.lower() not in non_actions and len(w) > 3]
        
        if potential_actions:
            # Use the longest potential action word
            action = max(potential_actions, key=len)
    
    # If we still don't have an action, extract keywords from the original text
    if not action or action == "X":
        # Identify potential action verbs in the original text
        text_lower = original_text.lower()
        
        # Common action verbs in regulatory requirements
        common_verbs = ["process", "assist", "notify", "ensure", "implement", 
                       "provide", "make", "maintain", "protect", "delete", 
                       "return", "engage", "obtain", "allow", "conduct", "impose"]
        
        for verb in common_verbs:
            if verb in text_lower:
                # Get the sentence fragment starting with this verb
                pos = text_lower.find(verb)
                fragment = text_lower[pos:pos+30]  # Take a reasonably sized fragment
                
                # Extract a keyword phrase
                words = re.findall(r'\b\w+\b', fragment)
                if len(words) >= 2:
                    # Use verb + next meaningful word
                    action = f"{words[0]}_{words[1]}"
                    break
                else:
                    action = verb
                    break
    
    # Clean up the action
    if action:
        # Remove spaces and replace with underscores
        action = action.replace(' ', '_').lower()
        
        # Remove any remaining special characters
        action = re.sub(r'[^a-z0-9_]', '', action)
    else:
        # Default action if we couldn't extract anything
        action = "generic_action"
    
    # Add arguments if not present
    if '(' not in action and ')' not in action:
        # Determine appropriate arguments based on context
        if any(term in original_text.lower() for term in ["sub-processor", "subprocessor"]):
            action = f"{action}(processor, sub_processor)"
        elif any(term in original_text.lower() for term in ["controller", "assist controller"]):
            action = f"{action}(processor, controller)"
        elif any(term in original_text.lower() for term in ["personal data", "data"]):
            action = f"{action}(processor, personal_data)"
        else:
            # Default to processor as actor
            action = f"{action}(processor)"
    
    return action

def convert_to_predicate_format(symbolic_repr: str, entity_id: str, original_text: str, entity_type: str = "requirement") -> List[str]:
    """Convert to predicate-based format used in the ASP template."""
    modality = extract_deontic_operator(symbolic_repr)
    action = extract_action(symbolic_repr, original_text)
    
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

def generate_default_facts() -> List[str]:
    """Generate default facts for the ASP program."""
    return [
        "% --- Facts ---",
        "% Default facts for testing",
        "processor(processor).",
        "sub_processor(sub_processor).",
        "controller(controller).",
        "personal_data(personal_data).",
        "data_subject(data_subject)."
    ]

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
        
        # Sanitize symbolic representation to avoid syntax errors
        sanitized_symbolic = sanitize_symbolic_repr(req_symbolic)
        
        # Generate requirement predicates
        req_predicates = convert_to_predicate_format(sanitized_symbolic, f"req{req_id}", req_text, "requirement")
        
        with open(lp_file_path, "w") as f:
            # Add header - use the sanitized symbolic representation
            f.write(f"""% Deolingo program to check DPA against Requirement {req_id}
% ==========================================================

% --- Requirement Definition ---
% Original text: {req_text}

{os.linesep.join(req_predicates)}

% --- DPA Segment Content Representation ---
% Represents what the DPA explicitly states using these facts.

""")
            
            # Add DPA segments
            for segment_text, segment_info in segments.items():
                segment_id = segment_info["id"]
                segment_symbolic = segment_info["symbolic"]
                
                # Sanitize segment symbolic representation as well
                sanitized_segment = sanitize_symbolic_repr(segment_symbolic)
                
                dpa_predicates = convert_to_predicate_format(
                    sanitized_segment, f"dpa{segment_id}", 
                    segment_text, "dpa")
                
                # Truncate the segment text to keep file readable
                truncated_text = segment_text[:50] + "..." if len(segment_text) > 50 else segment_text
                f.write(f"% DPA Segment {segment_id}: {truncated_text}\n")
                f.write(os.linesep.join(dpa_predicates) + "\n\n")
            
            # Add facts - use default facts instead of trying to extract from symbolic
            facts = generate_default_facts()
            f.write(os.linesep.join(facts) + "\n\n")
            
            # Add matching logic
            f.write("""% --- Matching and Status Logic ---

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