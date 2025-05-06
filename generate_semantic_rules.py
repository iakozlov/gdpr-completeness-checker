# generate_semantic_rules.py
import os
import json
import argparse
from models.llm import LlamaModel
from config.llm_config import LlamaConfig

def generate_semantic_rule_prompt():
    """Create prompt with examples for semantic rule generation."""
    
    system_prompt = """
You are a legal expert in GDPR compliance. Your task is to identify semantic connections
between regulatory requirements and DPA (Data Processing Agreement) segments.

For each pair, determine if the DPA segment semantically satisfies the requirement,
and if so, generate a logical rule that connects them.

RULES FOR SEMANTIC CONNECTIONS:
1. Each rule should connect a predicate from the requirement to a predicate from the DPA
2. Use the format: requirement_predicate :- dpa_predicate.
3. Do not use deontic operators (&obligatory, &permitted, &forbidden) in your rule
4. Rules must be syntactically valid for Deolingo (an Answer Set Programming system)
5. If there is no semantic connection, respond with: NO_SEMANTIC_CONNECTION

EXAMPLES:

Example 1:
REQUIREMENT: The processor shall ensure the security of processing.
SYMBOLIC: &obligatory{ensure_security_processing} :- process_personal_data(processor).
DPA SEGMENT: The Processor shall implement appropriate technical measures.
SYMBOLIC: &obligatory{implement_technical_measures} :- process_data(processor).
SEMANTIC RULE: ensure_security_processing :- implement_technical_measures.

Example 2:
REQUIREMENT: The processor shall delete all personal data after the end of service.
SYMBOLIC: &obligatory{delete_personal_data} :- end_of_service.
DPA SEGMENT: Data must be permanently destroyed after completion of services.
SYMBOLIC: &obligatory{destroy_data} :- completion_of_services.
SEMANTIC RULE: delete_personal_data :- destroy_data.

Example 3:
REQUIREMENT: The processor shall not engage a sub-processor without authorization.
SYMBOLIC: &forbidden{engage_sub_processor} :- not authorization(controller).
DPA SEGMENT: The Company may hire external consultants to improve their IT systems.
SYMBOLIC: &permitted{hire_consultants} :- improve_it_systems.
SEMANTIC RULE: NO_SEMANTIC_CONNECTION
"""

    return system_prompt

def generate_semantic_rule(llm_model, req_text, req_symbolic, dpa_text, dpa_symbolic):
    """Generate a semantic rule connecting a DPA segment to a requirement."""
    
    system_prompt = generate_semantic_rule_prompt()
    
    user_prompt = f"""
REQUIREMENT TEXT:
{req_text}

REQUIREMENT SYMBOLIC REPRESENTATION:
{req_symbolic}

DPA SEGMENT TEXT:
{dpa_text}

DPA SEGMENT SYMBOLIC REPRESENTATION:
{dpa_symbolic}

Generate a single semantic connection rule that shows how the DPA segment satisfies the requirement.
Extract predicates from within deontic operators if needed.

IMPORTANT: 
- Only output the rule itself with no explanations
- Format: requirement_predicate :- dpa_predicate.
- If no connection exists, just respond with: NO_SEMANTIC_CONNECTION
"""
    
    response = llm_model.generate_symbolic_representation(user_prompt, system_prompt)
    
    # Process response
    response = response.strip()
    if "NO_SEMANTIC_CONNECTION" in response:
        return None
    
    # Extract just the rule
    lines = response.split('\n')
    for line in lines:
        line = line.strip()
        if ':-' in line:
            # Ensure rule ends with a period
            if not line.endswith('.'):
                line += '.'
            return line
    
    return None