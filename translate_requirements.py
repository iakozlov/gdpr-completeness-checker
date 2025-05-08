# translate_requirements.py
import os
import re
import json
import argparse
from models.llm import LlamaModel
from solver.deontic_translator import DeonticTranslator
from config.llm_config import LlamaConfig

def main():
    parser = argparse.ArgumentParser(description="Translate GDPR requirements to deontic logic")
    parser.add_argument("--requirements", type=str, default="data/requirements/ground_truth_requirements.txt",
                        help="Path to requirements file")
    parser.add_argument("--model", type=str, default="models/Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf",
                        help="Path to LLM model file")
    parser.add_argument("--output", type=str, default="data/processed/requirements_deontic.json",
                        help="Output JSON file path")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Initialize LLM and translator
    print(f"Initializing LLM with model: {args.model}")
    llm_config = LlamaConfig(model_path=args.model, temperature=0.1)
    llm_model = LlamaModel(llm_config)
    translator = DeonticTranslator()
    print("LLM initialized successfully")
    
    # Read and parse requirements
    print(f"Reading requirements from: {args.requirements}")
    requirements = {}
    
    with open(args.requirements, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Extract requirement ID and text (format: "1. The processor shall...")
            match = re.match(r'^(\d+)\.\s*(.+)$', line)
            if match:
                req_id = match.group(1)
                req_text = match.group(2)
                
                print(f"Translating requirement {req_id}: {req_text[:50]}...")
                
                # Create system prompt for deontic logic translation
                system_prompt = """
                You are a specialized AI assistant trained to translate GDPR legal requirements 
                into formal Deontic Logic representations via Answer Set Programming (ASP).
                
                Translate the requirement into Deontic Logic format using the following structure:

                &obligatory{action} :- condition1, condition2.
                &permitted{action} :- condition1, condition2.
                &forbidden{action} :- condition1, condition2.
                
                IMPORTANT SYNTAX RULES:
                1. Use only alphanumeric characters and underscores in predicate names
                2. For processor obligations, use the form: &obligatory{action} :- role(processor).
                3. For controller obligations, use the form: &obligatory{action} :- role(controller).
                4. NEVER use "not(predicate)" format - ASP uses "-predicate" for negation
                5. For negated conditions, use "-predicate" format (with a minus sign)
                6. Keep predicates simple and readable
                7. ALWAYS close all parentheses in predicates: role(processor) NOT role(processor
                8. NEVER use commas inside role predicates - use: role(processor) NOT role(processor, controller)
                9. Use separate predicates for different conditions instead of complex expressions

                EXAMPLES:

                Example 1:
                Requirement: The processor shall ensure that persons authorised to process the personal data have committed themselves to confidentiality.
                Correct: &obligatory{ensure_confidentiality} :- role(processor).
                Incorrect: &obligatory{ensure_confidentiality} :- not(role(controller)). (Don't use not() format)
                Incorrect: &obligatory{ensure_confidentiality} :- role(processor. (Unclosed parentheses)

                Example 2:
                Requirement: The processor shall not engage another processor without prior authorization of the controller.
                Correct: &obligatory{not_engage_sub_processor} :- role(processor), -prior_authorization.
                Incorrect: &forbidden{engage_sub_processor} :- role(processor), -authorized_by_controller(controller. (Unclosed parentheses)
                Incorrect: &forbidden{engage_sub_processor} :- role(processor, controller). (Don't use multiple values in role predicate)

                Example 3:
                Requirement: The processor shall assist the controller in ensuring compliance with security obligations.
                Correct: &obligatory{assist_with_security_compliance} :- role(processor).
                
                Include only the symbolic representation in the response without any other comments.
                """
                
                user_prompt = f"Translate this GDPR requirement into symbolic representation in Deontic Logic via ASP: {req_text}"
                
                # Generate symbolic representation
                llm_output = llm_model.generate_symbolic_representation(user_prompt, system_prompt)
                
                # Fix common syntax issues
                fixed_output = fix_symbolic_syntax(llm_output)
                symbolic = translator.translate(fixed_output)
                
                # Extract atoms from symbolic representation
                atoms = extract_atoms(symbolic)
                
                # Store in dictionary
                requirements[req_id] = {
                    "text": req_text,
                    "symbolic": symbolic,
                    "atoms": atoms
                }
    
    # Save to JSON file
    print(f"Saving requirement representations to: {args.output}")
    with open(args.output, 'w') as f:
        json.dump(requirements, f, indent=2)
    
    print(f"Translation complete! Processed {len(requirements)} requirements.")

def fix_symbolic_syntax(symbolic):
    """Fix common syntax errors in the symbolic representation."""
    fixed = symbolic
    
    # Replace not() with minus sign
    fixed = fixed.replace("not(", "-").replace("not (", "-")
    
    # Fix unclosed parentheses in predicates
    parentheses_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\(([^)]*?)(?=[,\.\n])'
    fixed = re.sub(parentheses_pattern, r'\1(\2)', fixed)
    
    # Fix role predicates with multiple arguments
    role_pattern = r'role\(([^,]+),\s*([^)]+)\)'
    role_matches = re.findall(role_pattern, fixed)
    for match in role_matches:
        original = f"role({match[0]}, {match[1]})"
        replacement = f"role({match[0]}), {match[1]}"
        fixed = fixed.replace(original, replacement)
    
    # Ensure parentheses are balanced
    lines = fixed.split('\n')
    balanced_lines = []
    for line in lines:
        # Count opening and closing parentheses
        open_count = line.count('(')
        close_count = line.count(')')
        # Add missing closing parentheses
        if open_count > close_count:
            line += ')' * (open_count - close_count)
        balanced_lines.append(line)
    
    fixed = '\n'.join(balanced_lines)
    
    return fixed

def extract_atoms(symbolic):
    """Extract atoms (predicates) from the symbolic representation."""
    atoms = set()
    
    # Extract predicates from deontic operators
    for op in ['&obligatory', '&permitted', '&forbidden']:
        pattern = rf'{op}{{([^}}]+)}}'
        matches = re.findall(pattern, symbolic)
        for match in matches:
            # Extract predicate without arguments
            if '(' in match:
                predicate = match.split('(')[0].strip()
                atoms.add(predicate)
            else:
                atoms.add(match.strip())
    
    # Extract predicates from conditions
    for line in symbolic.split('\n'):
        if ':-' in line:
            condition = line.split(':-')[1].strip()
            if condition.endswith('.'):
                condition = condition[:-1]
            
            # Split by comma and extract predicates
            for part in condition.split(','):
                part = part.strip()
                if part.startswith('not '):
                    part = part[4:].strip()
                
                # Extract predicate without arguments
                if '(' in part:
                    predicate = part.split('(')[0].strip()
                    atoms.add(predicate)
                else:
                    atoms.add(part)
    
    return list(atoms)

if __name__ == "__main__":
    main()