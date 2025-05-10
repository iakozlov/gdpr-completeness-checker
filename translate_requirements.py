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
    parser.add_argument("--output", type=str, default="data/processed/requirements_deontic_experiments.json",
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
You are a specialized AI trained to translate legal requirements into formal Deontic Logic representations using Answer Set Programming (ASP) syntax.

IMPORTANT OUTPUT RULES:
1. ONLY output the ASP code - NO explanations, comments about facts, or additional text
2. Output ONLY the deontic rules - DO NOT include any derived facts
3. DO NOT include anything like "% Facts derived from rule conditions"
4. Ensure all predicates have properly BALANCED parentheses: role(processor) NOT role(processor))
5. DO NOT include "." at the end of conditions that follow ":-"
6. Use only "-predicate" for negation, NEVER use "not(predicate)"

RULES FOR PREDICATES:
- Use snake_case for predicate names (e.g., ensure_security)
- For role predicates, ONLY use either role(processor) or role(controller) - no other variations
- Ensure ALL parentheses are properly balanced and closed
- Keep predicates simple and meaningful
- Avoid periods within predicates

PATTERN TO FOLLOW:
&obligatory{action_predicate} :- condition1, condition2.
&permitted{action_predicate} :- condition1, condition2.
&forbidden{action_predicate} :- condition1, condition2.

EXAMPLES OF CORRECT OUTPUT:

Example 1:
&obligatory{ensure_confidentiality} :- role(processor).

Example 2:
&obligatory{not_engage_subprocessor} :- role(processor), -prior_authorization.

Example 3:
&obligatory{assist_controller} :- role(processor).
&obligatory{ensure_compliance} :- role(processor).

DO NOT include ANY explanatory text or anything other than the ASP rules.
"""
                
                user_prompt = f"Translate this GDPR requirement into symbolic representation in Deontic Logic via ASP: {req_text}"
                
                # Generate symbolic representation
                llm_output = llm_model.generate_symbolic_representation(user_prompt, system_prompt)
                
                # Clean the output - remove any facts or explanatory sections
                cleaned_output = clean_symbolic_output(llm_output)
                
                # Fix common syntax issues
                fixed_output = fix_symbolic_syntax(cleaned_output)
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

def clean_symbolic_output(output):
    """Remove explanatory text, facts sections, and other non-rule content."""
    lines = output.strip().split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue
        
        # Skip comment lines and facts sections
        if line.strip().startswith('%'):
            continue
        
        # Skip lines that don't contain deontic operators
        if not any(op in line for op in ['&obligatory', '&permitted', '&forbidden']):
            continue
            
        # Remove trailing periods from condition lists
        if ':-' in line:
            parts = line.split(':-')
            if len(parts) == 2:
                head = parts[0].strip()
                body = parts[1].strip()
                if body.endswith('.'):
                    body = body[:-1]
                line = f"{head} :- {body}."
        
        cleaned_lines.append(line.strip())
    
    return '\n'.join(cleaned_lines)

def fix_symbolic_syntax(symbolic):
    """Fix common syntax errors in the symbolic representation."""
    fixed = symbolic
    
    # Replace not() with minus sign
    fixed = re.sub(r'not\s*\(([^)]+)\)', r'-\1', fixed)
    
    # Fix unclosed parentheses in predicates
    parentheses_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\(([^)]*?)(?=[,\.\s]|$)'
    fixed = re.sub(parentheses_pattern, r'\1(\2)', fixed)
    
    # Fix extra parentheses in role predicates - role(processor)) -> role(processor)
    fixed = re.sub(r'role\(([a-z_]+)\)\)', r'role(\1)', fixed)
    
    # Fix role predicates with multiple arguments
    role_pattern = r'role\(([^,]+),\s*([^)]+)\)'
    fixed = re.sub(role_pattern, r'role(\1), role(\2)', fixed)
    
    # Remove extra spaces around commas
    fixed = re.sub(r'\s*,\s*', r', ', fixed)
    
    # Ensure each line ends with a period
    lines = fixed.split('\n')
    for i in range(len(lines)):
        line = lines[i].strip()
        if line and not line.endswith('.'):
            lines[i] = line + '.'
    
    fixed = '\n'.join(lines)
    
    # Handle unbalanced parentheses in the entire text
    open_count = fixed.count('(')
    close_count = fixed.count(')')
    
    if open_count > close_count:
        fixed += ')' * (open_count - close_count)
    
    # Ensure all parentheses are properly balanced in each predicate
    # This is a more thorough approach to handle nested parentheses
    def balance_parentheses(text):
        result = ""
        stack = []
        
        for char in text:
            if char == '(':
                stack.append(char)
                result += char
            elif char == ')':
                if stack:
                    stack.pop()
                    result += char
                # Skip extra closing parentheses
            else:
                result += char
                
        # Add missing closing parentheses
        result += ')' * len(stack)
        return result
    
    # Apply to each predicate
    predicate_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*\([^,\.\s]*)'
    fixed = re.sub(predicate_pattern, lambda m: balance_parentheses(m.group(0)), fixed)
    
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
                if part.startswith('-'):
                    part = part[1:].strip()
                
                # Extract predicate without arguments
                if '(' in part:
                    predicate = part.split('(')[0].strip()
                    atoms.add(predicate)
                else:
                    atoms.add(part)
    
    return list(atoms)

if __name__ == "__main__":
    main()