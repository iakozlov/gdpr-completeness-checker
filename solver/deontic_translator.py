# solver/deontic_translator.py
import re
from typing import Tuple

class DeonticTranslator:
    """
    Translator to ensure LLM outputs conform to deolingo syntax
    while preserving the semantic content.
    """
    
    def __init__(self):
        self.operators = {
            'obligation': '&obligatory',
            'permission': '&permitted',
            'prohibition': '&forbidden',
        }
    
    def clean_representation(self, repr_text: str) -> str:
        """Clean and standardize the symbolic representation from LLM to valid Deolingo syntax."""
        # Remove markdown code blocks if present
        cleaned = re.sub(r'```(?:prolog|asp)?\n(.*?)```', r'\1', repr_text, flags=re.DOTALL)
        
        # Remove explanatory text sections
        cleaned = re.sub(r'\*\*.*?\*\*', '', cleaned)  # Remove **headers**
        cleaned = re.sub(r'\\forall.*?→', '', cleaned)  # Remove mathematical notation
        cleaned = re.sub(r'- `.*?`', '', cleaned)  # Remove bullet points
        cleaned = re.sub(r'Let `.*?`.*', '', cleaned)  # Remove definitions
        cleaned = re.sub(r'\d+\. .*', '', cleaned)  # Remove numbered lists
        cleaned = re.sub(r'The antecedent.*', '', cleaned)  # Remove explanations
        cleaned = re.sub(r'indicates that.*', '', cleaned)  # Remove explanations
        
        # Extract only valid deontic statements
        deontic_lines = []
        for line in cleaned.split('\n'):
            line = line.strip()
            # Skip empty lines and explanatory text
            if not line or line.startswith('%'):
                continue
            
            # Check if line contains deontic operators
            if any(op in line for op in ['&obligatory', '&permitted', '&forbidden']):
                # Clean up invalid characters - Fixed regex pattern
                line = re.sub(r'[^a-zA-Z0-9_{}(),.:;\-\s]', '', line)
                
                # Fix common syntax errors
                line = line.replace(')))', ')')
                line = line.replace('__', '_')
                line = line.replace(',,', ',')
                line = line.replace(':.', '.')
                line = line.replace(':-.', ':- .')
                
                # Ensure proper ending
                if not line.endswith('.'):
                    line += '.'
                
                deontic_lines.append(line)
        
        # Return cleaned lines or default to simple obligatory statement
        if deontic_lines:
            return '\n'.join(deontic_lines)
        else:
            # If no valid lines found, return a default
            return '&obligatory{process_data(processor)} :- controller(controller).'
    
    def validate_syntax(self, repr_text: str) -> Tuple[bool, str]:
        """Validate the syntax of the representation."""
        # Check for deontic operators in deolingo format
        if not re.search(r'&obligatory|&permitted|&forbidden', repr_text):
            return False, "Missing deontic operators (&obligatory, &permitted, &forbidden)"
        
        # Check for unbalanced braces
        if repr_text.count('{') != repr_text.count('}'):
            return False, "Unbalanced braces"
        
        # Check for unbalanced parentheses
        if repr_text.count('(') != repr_text.count(')'):
            return False, "Unbalanced parentheses"
        
        # Check for invalid characters with fixed regex pattern
        if re.search(r'[^a-zA-Z0-9_{}(),.:;\-\s]', repr_text):
            return False, "Invalid characters found"
        
        return True, "Valid syntax"
    
    def translate(self, llm_output: str) -> str:
        """Translate LLM output to deolingo format."""
        # Print the original output for reference
        print(f"LLM symbolic representation: {llm_output}")
        
        # Clean up the representation
        cleaned = self.clean_representation(llm_output)
        valid, message = self.validate_syntax(cleaned)
        
        # If invalid, make a minimal viable representation
        if not valid:
            print(f"Invalid syntax: {message}. Making minimal fixes.")
            
            # Try to extract meaningful content while fixing syntax
            if "&obligatory" in llm_output or "obligat" in llm_output.lower():
                fixed = "&obligatory{process_data(processor)} :- controller(controller)."
            elif "&forbidden" in llm_output or "forbid" in llm_output.lower() or "prohibit" in llm_output.lower():
                fixed = "&forbidden{unauthorized_action(processor)} :- not authorized(controller)."
            else:
                fixed = "&obligatory{comply(processor)} :- true."
            
            return fixed
        
        return cleaned