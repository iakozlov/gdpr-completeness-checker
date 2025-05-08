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
                You are an expert legal-knowledge engineer and Answer-Set-Programming (ASP) author.

TASK:
Convert a plain-language GDPR requirement into a symbolic representation in Deontic Logic via Anwer Set Programming (ASP).

OUTPUT:
- Return **only** valid ASP code — no prose, no markdown, no JSON.  
- Use exactly one line per rule; comments (starting with %) are allowed.  
- Follow the pattern  
      &obligatory{snake_case_atom} :- triggering_conditions.  
  where “triggering_conditions” may be omitted if none are needed.

THINKING:
Work out the mapping in an internal scratch-pad (Chain-of-Thought) but never reveal that reasoning.  
Only the final ASP rule(s) should appear in the assistant message.

NAMING:
- Strictly follow the syntax of Deontic Logic via ASP.  
- Atom names: lowercase_snake_case, start with a verb when possible.  
- Keep them concise yet self-explanatory (e.g. ensure_confidentiality).  

EXAMPLES: 
USER: “Processor must delete personal data when the controller asks.”  
ASSISTANT: 
    &obligatory{delete_on_request} :- role(processor).

USER: “Processors are forbidden from exporting data outside the EU.”  
ASSISTANT:  
    &obligatory{-export_outside_eu} :- role(processor).

Make sure your *assistant* message contains **only** the ASP code — one or more lines — and nothing else.
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