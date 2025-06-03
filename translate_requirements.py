# translate_requirements.py
import os
import re
import json
import argparse
from models.gpt_model import GPTModel
from models.llama_model import LlamaModel
from models.ollama_model import OllamaModel
from config.gpt_config import GPTConfig
from config.llama_config import LlamaConfig
from config.ollama_config import OllamaConfig

def extract_predicates_from_symbolic(symbolic_repr: str) -> list:
    """Extract all predicates/atoms from a symbolic representation.
    
    Args:
        symbolic_repr: The symbolic deontic logic representation
        
    Returns:
        List of unique predicates found in the symbolic representation
    """
    predicates = set()
    
    # Find predicates in deontic operators: &obligatory{...}, &forbidden{...}, &permitted{...}
    deontic_patterns = [
        r'&obligatory\{([^}]+)\}',
        r'&forbidden\{([^}]+)\}', 
        r'&permitted\{([^}]+)\}'
    ]
    
    for pattern in deontic_patterns:
        matches = re.findall(pattern, symbolic_repr)
        for match in matches:
            # Handle negation (remove leading -)
            predicate = match.strip()
            if predicate.startswith('-'):
                predicate = predicate[1:]
            predicates.add(predicate)
    
    # Find predicates in rule bodies (after :-)
    if ':-' in symbolic_repr:
        body = symbolic_repr.split(':-')[1].strip()
        # Remove trailing period
        if body.endswith('.'):
            body = body[:-1]
        
        # Split by commas and extract predicates
        conditions = [cond.strip() for cond in body.split(',')]
        for condition in conditions:
            # Handle negation (not predicate)
            if condition.startswith('not '):
                predicate = condition[4:].strip()
            else:
                predicate = condition.strip()
            
            # Special handling for standard predicates with arguments
            if predicate.startswith('role(') or predicate.startswith('authorization('):
                # Keep the full predicate with arguments for standard predicates
                predicates.add(predicate)
            elif '(' in predicate:
                # For other predicates, extract just the predicate name
                predicate_name = predicate.split('(')[0]
                if predicate_name and predicate_name not in ['not', 'true', 'false']:
                    predicates.add(predicate_name)
            else:
                # Simple predicate without parentheses
                if predicate and predicate not in ['not', 'true', 'false']:
                    predicates.add(predicate)
    
    return sorted(list(predicates))

def create_system_prompt() -> str:
    """Create the system prompt for deontic logic translation."""
    return """You are a specialized AI assistant trained to translate legal requirements into formal Deontic Logic via Answer Set Programming representations.

Your task is to convert natural language legal requirements into symbolic deontic logic expressions using the following format:

DEONTIC OPERATORS:
- &obligatory{predicate} for obligations (things that must be done)
- &forbidden{predicate} for prohibitions (things that must not be done)  
- &permitted{predicate} for permissions (things that may be done)

LOGICAL STRUCTURE:
- Use :- for logical implication (if-then)
- Use role(processor) to indicate the processor role
- Use role(controller) to indicate the controller role
- Use 'not' for negation
- Use commas to separate multiple conditions
- End statements with a period
- Generate ONLY ONE deontic operator per rule
- Do NOT use multiple &obligatory{} statements in a single rule

PREDICATE NAMING:
- Create meaningful predicate names using snake_case
- Use descriptive names that capture the essence of the requirement
- Keep predicates concise but clear
- Do NOT use logical operators like "or", "and" as predicate names
- Do NOT include punctuation or special characters in predicate names

EXAMPLES:
- "The processor must not perform action X without proper authorization" 
  → &obligatory{-perform_action_x} :- role(processor), not proper_authorization.

- "The processor shall notify the authority when incidents occur"
  → &obligatory{notify_authority} :- role(processor), incident_occurs.

- "If special conditions apply, the processor must follow additional procedures"
  → &obligatory{follow_additional_procedures} :- role(processor), special_conditions.

- "The processor is forbidden from sharing data without user consent"
  → &forbidden{share_data} :- role(processor), not user_consent.

IMPORTANT RULES:
1. Generate exactly ONE deontic logic statement per requirement
2. Use only ONE &obligatory{}, &forbidden{}, or &permitted{} per rule
3. Do NOT use words like "or", "and", "not" as standalone predicates
4. Keep predicate names simple and descriptive
5. Provide ONLY the symbolic representation without any additional explanation, comments, or formatting."""

def create_user_prompt(requirement_text: str) -> str:
    """Create the user prompt for a specific requirement.
    
    Args:
        requirement_text: The natural language requirement text
        
    Returns:
        The formatted user prompt
    """
    return f"""Translate the following GDPR requirement into deontic logic:

{requirement_text}

Provide only the symbolic representation in the format: &deontic_operator{{predicate}} :- conditions."""

def main():
    parser = argparse.ArgumentParser(description="Translate GDPR requirements to deontic logic")
    parser.add_argument("--requirements", type=str, default="data/requirements/ground_truth_requirements.txt",
                        help="Path to requirements file")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="Model to use (gpt-4o, meta-llama/Llama-3.3-70B-Instruct, or ollama model like llama3.3:70b)")
    parser.add_argument("--output", type=str, default="results/gpt4o_experiment/requirements_deontic.json",
                        help="Output JSON file path")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Initialize LLM
    print(f"Initializing model: {args.model}")
    
    # Choose the appropriate model based on the model name
    if args.model.startswith("meta-llama"):
        model_config = LlamaConfig(model=args.model, temperature=0.1)
        llm_model = LlamaModel(model_config)
    elif args.model.startswith("gpt-"):
        model_config = GPTConfig(model=args.model, temperature=0.1)
        llm_model = GPTModel(model_config)
    else:
        # Assume it's an Ollama model (e.g., llama3.3:70b, qwen2.5:32b, etc.)
        model_config = OllamaConfig(model=args.model, temperature=0.1)
        llm_model = OllamaModel(model_config)
    
    print("Model initialized successfully")
    
    # Create system prompt
    system_prompt = create_system_prompt()
    
    # Read and parse requirements
    print(f"Reading requirements from: {args.requirements}")
    requirements = {}
    
    with open(args.requirements, 'r') as f:
        current_req_id = None
        current_req_text = []
        
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Check if this is a requirement ID line - support both formats
            # Format 1: "R1: text..." 
            req_match_r = re.match(r'^R(\d+):', line)
            # Format 2: "1. text..."
            req_match_num = re.match(r'^(\d+)\.\s+(.+)', line)
            
            if req_match_r:
                # Save previous requirement if exists
                if current_req_id is not None:
                    requirements[current_req_id] = {
                        'text': ' '.join(current_req_text),
                        'symbolic': None,
                        'atoms': []
                    }
                
                # Start new requirement (R format)
                current_req_id = req_match_r.group(1)
                current_req_text = [line]
            elif req_match_num:
                # Save previous requirement if exists
                if current_req_id is not None:
                    requirements[current_req_id] = {
                        'text': ' '.join(current_req_text),
                        'symbolic': None,
                        'atoms': []
                    }
                
                # Start new requirement (numbered format)
                current_req_id = req_match_num.group(1)
                current_req_text = [req_match_num.group(2)]  # Just the text part, not the number
            else:
                # Add to current requirement text
                if current_req_id is not None:
                    current_req_text.append(line)
        
        # Save last requirement
        if current_req_id is not None:
            requirements[current_req_id] = {
                'text': ' '.join(current_req_text),
                'symbolic': None,
                'atoms': []
            }
    
    # Translate each requirement to deontic logic
    print("Translating requirements to deontic logic...")
    for req_id, req_data in requirements.items():
        print(f"Processing requirement {req_id}...")
        try:
            # Create user prompt for this requirement
            user_prompt = create_user_prompt(req_data['text'])
            
            # Generate symbolic representation
            symbolic_repr = llm_model.generate_symbolic_representation(
                user_prompt=user_prompt,
                system_prompt=system_prompt
            )
            
            # Clean up the response (remove any extra formatting)
            symbolic_repr = symbolic_repr.strip()
            if symbolic_repr.startswith('```'):
                # Remove code block formatting if present
                lines = symbolic_repr.split('\n')
                symbolic_repr = '\n'.join(line for line in lines if not line.startswith('```')).strip()
            
            # Extract predicates from the symbolic representation
            predicates = extract_predicates_from_symbolic(symbolic_repr)
            
            # Store results
            requirements[req_id]['symbolic'] = symbolic_repr
            requirements[req_id]['atoms'] = predicates
            
            print(f"  Generated: {symbolic_repr}")
            print(f"  Predicates: {predicates}")
            
        except Exception as e:
            print(f"Error processing requirement {req_id}: {str(e)}")
            requirements[req_id]['symbolic'] = None
            requirements[req_id]['atoms'] = []
    
    # Save results
    print(f"Saving results to: {args.output}")
    with open(args.output, 'w') as f:
        json.dump(requirements, f, indent=2)
    
    print("Translation completed successfully!")
    
    # Print summary
    successful = sum(1 for req in requirements.values() if req['symbolic'] is not None)
    total = len(requirements)
    print(f"Successfully translated {successful}/{total} requirements")

if __name__ == "__main__":
    main()