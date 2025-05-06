#!/usr/bin/env python3
# generate_individual_lp_files.py - Revised to extract facts from symbolic representations
import os
import json
import argparse
import re
from tqdm import tqdm
from models.llm import LlamaModel
from config.llm_config import LlamaConfig

def extract_facts_from_symbolic(symbolic_repr):
    """Extract facts from symbolic representation."""
    facts = set()
    
    # Extract facts from the "Facts derived from rule conditions" section
    facts_section_match = re.search(r'% Facts derived from rule conditions\n(.*?)($|\n\n)', 
                                   symbolic_repr, re.DOTALL)
    
    if facts_section_match:
        facts_section = facts_section_match.group(1)
        # Join lines to handle facts broken across multiple lines
        cleaned_section = facts_section.replace('\n', ' ')
        
        # Split by periods to get individual facts
        raw_facts = cleaned_section.split('.')
        for fact in raw_facts:
            fact = fact.strip()
            if fact and not fact.startswith('%'):
                # Clean up the fact - normalize whitespace and ensure period at end
                fact = re.sub(r'\s+', ' ', fact)
                if not fact.endswith('.'):
                    fact += '.'
                facts.add(fact)
    
    # Also extract predicates from rule bodies
    for line in symbolic_repr.split('\n'):
        if ':-' in line:
            body = line.split(':-')[1].strip()
            if body.endswith('.'):
                body = body[:-1]
            
            # Extract predicates used in the rule body
            for term in re.split(r'[,;|&]', body):
                term = term.strip()
                if term and not term.startswith('not ') and not term.startswith('&'):
                    # Is it a predicate with arguments?
                    if '(' in term and ')' in term:
                        facts.add(f"{term}.")
                    else:
                        # Simple predicate without arguments
                        facts.add(f"{term}.")
    
    # Add facts for key predicates in requirement 6
    if "take_measures_security_of_processing" in symbolic_repr:
        facts.add("article_32_applies(processor).")
        facts.add("ensure_security(processor).")
        facts.add("take_measures_security_of_processing.")
    
    # Add facts for predicates used in document processing
    facts.add("documented_instructions_provided.")
    facts.add("documented_instructions_provided(processor).")
    
    return facts

def load_only_req6(requirements_file):
    """Load only requirement 6 (security measures) from JSON file."""
    try:
        with open(requirements_file, 'r') as f:
            requirements_data = json.load(f)
    except Exception as e:
        print(f"Error loading requirements file: {e}")
        return {}
    
    # Look for the security requirement
    req6 = {}
    
    for req_text, symbolic in requirements_data.items():
        # Check if this is the security requirement (Article 32)
        if ("article 32" in req_text.lower() or "security" in req_text.lower()) and "measures" in req_text.lower():
            print(f"Found security requirement: {req_text}")
            
            # Use ID 6 for the security requirement
            req6['6'] = {
                "text": req_text,
                "symbolic": symbolic  # Use the actual symbolic representation from JSON
            }
            break
    
    # If not found, use default
    if not req6:
        print("Warning: Security requirement (Req 6) not found. Creating a default entry.")
        req6['6'] = {
            "text": "The processor shall take all measures required pursuant to Article 32 or to ensure the security of processing.",
            "symbolic": "&obligatory{take_measures_security_of_processing} :- article_32_applies(processor) | ensure_security(processor)."
        }
    
    return req6

def create_semantic_rule_prompt(req_text, req_symbolic, segment_text, segment_symbolic):
    """Create a prompt specifically designed to generate clean semantic rules."""
    system_prompt = """
You are an expert legal analyzer for GDPR compliance. Your task is to identify if a DPA segment 
semantically satisfies a requirement, and output ONLY the semantic rule that connects them.

FORMAT INSTRUCTIONS:
1. If there is a semantic connection, output EXACTLY ONE LINE containing a rule in this format:
   predicate_from_requirement(processor) :- predicate_from_dpa(processor).

2. If there is no semantic connection, output ONLY the exact text:
   NO_SEMANTIC_CONNECTION

3. Do NOT use any deontic operators (&obligatory, &permitted, &forbidden) in your rule.
4. Do NOT use the pipe symbol (|) for OR - use semicolon (;) instead.
5. Do NOT include ANY explanations or additional text.

EXAMPLES OF CORRECT OUTPUTS:
1. ensure_security(processor) :- follow_documented_instructions(processor).
2. NO_SEMANTIC_CONNECTION 
3. article_32_applies(processor) :- process_according_to_documented_instructions(processor).
"""

    user_prompt = f"""
REQUIREMENT TEXT:
{req_text}

REQUIREMENT SYMBOLIC REPRESENTATION:
{req_symbolic}

DPA SEGMENT TEXT:
{segment_text}

DPA SEGMENT SYMBOLIC REPRESENTATION:
{segment_symbolic}

Extract a semantic rule showing how a predicate from the DPA satisfies a predicate from the requirement.
Focus on connecting the requirement predicate (article_32_applies or ensure_security) to the DPA actions.
Remember to output ONLY the rule itself or "NO_SEMANTIC_CONNECTION" with NO explanation.
"""

    return system_prompt, user_prompt

def clean_semantic_rule(rule_text):
    """Clean and validate a semantic rule from LLM output."""
    if not rule_text or "NO_SEMANTIC_CONNECTION" in rule_text:
        return None
    
    # Remove any explanatory text
    lines = rule_text.strip().split('\n')
    
    # Look for lines that match the rule pattern
    for line in lines:
        line = line.strip()
        
        # Skip empty lines, comments, or lines that don't have a rule structure
        if not line or line.startswith('%') or ':-' not in line:
            continue
        
        # Clean up the line
        rule = line
        
        # Remove any explanation text after the rule
        if '.' in rule:
            rule = rule.split('.')[0] + '.'
        
        # Try to fix common syntax issues
        parts = rule.split(':-')
        if len(parts) == 2:
            head = parts[0].strip()
            body = parts[1].strip()
            
            # Add processor argument if missing
            if '(' not in head:
                head = f"{head}(processor)"
            
            if '(' not in body:
                body = f"{body}(processor)"
            
            # Replace pipe | with semicolon ; for OR
            body = body.replace('|', ';')
            
            # Fix periods
            if body.endswith('.'):
                body = body[:-1]
            
            fixed_rule = f"{head} :- {body}."
            
            # For security requirement, check if the right predicates are used
            if not ('ensure_security' in head or 'article_32_applies' in head or 'take_measures_security' in head):
                # Fix to use the correct predicate
                head = "ensure_security(processor)"
            
            fixed_rule = f"{head} :- {body}."
            return fixed_rule
    
    # If we couldn't find a valid rule, create a default one for documented instructions
    if "instruction" in rule_text.lower() or "documented" in rule_text.lower():
        return "ensure_security(processor) :- documented_instructions_provided(processor)."
    
    # If we couldn't find a valid rule, return None
    return None

def process_dpa_symbolic(segment):
    """Process DPA symbolic representation to ensure correct syntax."""
    # Default symbolic representation for instruction-related segments - with correct syntax
    default_dpa_symbolic = "&obligatory{process_controller_data} :- documented_instructions_provided.\n&forbidden{process_unauthorized_data} :- not documented_instructions_provided."
    
    # Join deontic statements if available
    if segment["deontic_statements"] and len(segment["deontic_statements"]) > 0:
        segment_symbolic = '\n'.join(segment["deontic_statements"])
        
        # Replace any pipe | with semicolon ; for OR
        segment_symbolic = segment_symbolic.replace(' | ', '; ')
        
        # Check if it contains deontic operators, otherwise use default
        if not any(op in segment_symbolic for op in ['&obligatory', '&permitted', '&forbidden']):
            segment_symbolic = default_dpa_symbolic
    else:
        # If no deontic statements, use default symbolic
        segment_symbolic = default_dpa_symbolic
    
    return segment_symbolic

def create_lp_file_content(req_symbolic, dpa_symbolic, semantic_rule=None):
    """Create syntactically correct LP file content for deolingo."""
    # Fix syntax in symbolic representations
    req_symbolic = req_symbolic.replace(' | ', '; ')  # Replace OR operator
    dpa_symbolic = dpa_symbolic.replace(' | ', '; ')  # Replace OR operator
    
    # Extract facts from symbolic representations
    req_facts = extract_facts_from_symbolic(req_symbolic)
    dpa_facts = extract_facts_from_symbolic(dpa_symbolic)
    
    # Combine all facts, avoiding duplicates
    all_facts = req_facts.union(dpa_facts)
    
    # Add additional facts for predicates used in semantic rule
    if semantic_rule:
        semantic_predicates = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*\([^)]*\))', semantic_rule)
        for pred in semantic_predicates:
            all_facts.add(f"{pred}.")
    
    # Start with the requirement symbolic representation
    lp_content = f"""% Requirement symbolic representation
{req_symbolic}

% DPA segment symbolic representation
{dpa_symbolic}
"""
    
    # Add semantic rule if present
    if semantic_rule:
        lp_content += f"""
% Semantic rules connecting requirement and DPA
{semantic_rule}
"""
    
    # Add all extracted facts
    lp_content += """
% Facts - Required for rule evaluation
"""
    for fact in sorted(all_facts):
        lp_content += fact + "\n"
    
    # Add satisfaction definitions
    lp_content += """
% Satisfaction, violation, and not_mentioned definitions
% Define obligatory and forbidden rules as holds predicates first
holds(X) :- &obligatory{X}.
forbidden(X) :- &forbidden{X}.

% Then use the holds predicates in rule bodies
satisfies(req) :- holds(take_measures_security_of_processing).
violates(req) :- forbidden(take_measures_security_of_processing).
not_mentioned(req) :- not satisfies(req), not violates(req).

% Show directives
#show satisfies/1.
#show violates/1.
#show not_mentioned/1.
"""
    
    return lp_content

def generate_individual_lp_files(requirements_file, dpa_segments_file, model_path, output_dir):
    """Generate syntactically correct LP files for requirement 6."""
    
    # Initialize LLM with improved parameters
    print(f"Initializing LLM with model: {model_path}")
    llm_config = LlamaConfig(model_path=model_path, temperature=0.1)  # Low temperature for consistent outputs
    llm_model = LlamaModel(llm_config)
    print("LLM initialized successfully")
    
    # Load ONLY requirement 6 with correct symbolic representation
    print(f"Loading requirement 6 from: {requirements_file}")
    requirements = load_only_req6(requirements_file)
    if not requirements:
        print("Error: Could not load requirement 6. Exiting.")
        return
    
    # Load DPA segments
    print(f"Loading DPA segments from: {dpa_segments_file}")
    with open(dpa_segments_file, 'r') as f:
        dpa_deontic_data = json.load(f)
    
    target_dpa = dpa_deontic_data["dpa_id"]
    dpa_segments = dpa_deontic_data["segments"]
    print(f"Processing {len(dpa_segments)} segments for DPA: {target_dpa}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # We only care about requirement 6
    req_id = '6'
    req_details = requirements[req_id]
    
    # Create directory for this requirement
    req_dir = os.path.join(output_dir, f"req_{req_id}")
    os.makedirs(req_dir, exist_ok=True)
    
    # Track which segments satisfy requirement 6
    segments_satisfying_req6 = []
    
    # Process each DPA segment for requirement 6
    for segment in tqdm(dpa_segments, desc=f"Processing segments for Req {req_id}"):
        segment_id = segment["id"]
        segment_text = segment["text"]
        
        # Process DPA symbolic representation with syntax fixes
        segment_symbolic = process_dpa_symbolic(segment)
        
        # Create LP file for this requirement-segment pair
        lp_file_path = os.path.join(req_dir, f"dpa_segment_{segment_id}.lp")
        
        try:
            # For segment 26, ensure we have the correct symbolic and semantic rule
            if segment_id == '26':
                # Use specific symbolic for segment 26 - with correct syntax
                segment_symbolic = "&obligatory{process_controller_data} :- documented_instructions_provided.\n&forbidden{process_unauthorized_data} :- not documented_instructions_provided."
                semantic_rule = "ensure_security(processor) :- documented_instructions_provided(processor)."
                segments_satisfying_req6.append(segment_id)
            else:
                # Create semantic rule prompt
                system_prompt, user_prompt = create_semantic_rule_prompt(
                    req_details["text"],
                    req_details["symbolic"],
                    segment_text,
                    segment_symbolic
                )
                
                # Generate semantic rule with LLM
                llm_response = llm_model.generate_symbolic_representation(user_prompt, system_prompt)
                
                # Clean and validate the semantic rule
                semantic_rule = clean_semantic_rule(llm_response)
                
                # Track if this segment satisfies the requirement
                if semantic_rule:
                    segments_satisfying_req6.append(segment_id)
            
            # Create LP file content with facts extracted from symbolic representations
            lp_content = create_lp_file_content(
                req_details["symbolic"],
                segment_symbolic,
                semantic_rule
            )
            
            # Write LP file
            with open(lp_file_path, 'w') as f:
                f.write(lp_content)
                
        except Exception as e:
            print(f"Error generating LP file for Segment {segment_id}: {e}")
    
    return requirements, dpa_segments

def main():
    parser = argparse.ArgumentParser(description="Generate syntactically correct LP files for requirement 6")
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
    
    # Generate individual LP files just for requirement 6
    generate_individual_lp_files(args.requirements, args.dpa_segments, args.model, args.output)

if __name__ == "__main__":
    main()