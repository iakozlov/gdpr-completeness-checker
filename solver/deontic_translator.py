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
        cleaned = re.sub(r'```(?:asp)?\n(.+?)\n```', r'\1', repr_text, flags=re.DOTALL)
        
        # ---- Step 1: Fix deontic operators to proper format ----
        
        # Replace O/P/F formats with &obligatory/&permitted/&forbidden
        cleaned = re.sub(r'O\(([^)]+)\)', r'&obligatory{\1}', cleaned)
        cleaned = re.sub(r'P\(([^)]+)\)', r'&permitted{\1}', cleaned)
        cleaned = re.sub(r'F\(([^)]+)\)', r'&forbidden{\1}', cleaned)
        
        cleaned = re.sub(r'o\(([^)]+)\)', r'&obligatory{\1}', cleaned)
        cleaned = re.sub(r'p\(([^)]+)\)', r'&permitted{\1}', cleaned)
        cleaned = re.sub(r'f\(([^)]+)\)', r'&forbidden{\1}', cleaned)
        
        # Replace full word versions
        cleaned = re.sub(r'obligation\(([^)]+)\)', r'&obligatory{\1}', cleaned)
        cleaned = re.sub(r'permission\(([^)]+)\)', r'&permitted{\1}', cleaned)
        cleaned = re.sub(r'prohibition\(([^)]+)\)', r'&forbidden{\1}', cleaned)
        
        # ---- Step 2: Fix the contents inside deontic operators ----
        
        # Function to fix the content inside deontic operators
        def fix_deontic_content(match):
            op = match.group(1)  # The operator (&obligatory, &permitted, &forbidden)
            content = match.group(2)  # The content inside braces
            
            # Remove dots/periods inside the braces (convert to underscores)
            content = content.replace('.', '_')
            
            # Remove complex predicate structures (simplify to just the predicate name)
            if '(' in content and ')' in content:
                # Extract the predicate name before arguments
                predicate = content.split('(')[0].strip()
                
                # Extract any arguments and convert them to simpler form
                args_match = re.search(r'\(([^)]+)\)', content)
                if args_match:
                    args = args_match.group(1).split(',')
                    # Create a simple predicate with args as suffix
                    simple_pred = predicate
                    for arg in args:
                        # Clean up arg and add as suffix
                        arg_clean = arg.strip().lower().replace(' ', '_')
                        if arg_clean and arg_clean != 'processor' and arg_clean != 'controller':
                            simple_pred += f"_{arg_clean}"
                    
                    # Return fixed deontic expression
                    return f"{op}{{{simple_pred}}}"
                else:
                    return f"{op}{{{predicate}}}"
            else:
                # No arguments, just clean up spaces
                content = content.strip().replace(' ', '_')
                return f"{op}{{{content}}}"
        
        # Apply the fix to all deontic operators
        cleaned = re.sub(r'(&obligatory|&permitted|&forbidden){([^}]+)}', fix_deontic_content, cleaned)
        
        # ---- Step 3: Ensure each rule ends with a period ----
        lines = cleaned.split('\n')
        formatted_lines = []
        for line in lines:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('%'):
                formatted_lines.append(line)
                continue
            
            # Add period if missing
            if line and not line.endswith('.'):
                line += '.'
            
            formatted_lines.append(line)
        
        cleaned = '\n'.join(formatted_lines)
        
        # ---- Step 4: Add facts for predicates used in rule bodies ----
        # Extract predicates from rule bodies (after ":-")
        facts = []
        fact_set = set()
        
        for line in formatted_lines:
            if ':-' in line:
                body = line.split(':-')[1].strip(' .')
                
                # Handle negation
                for term in body.split(','):
                    term = term.strip()
                    if term.startswith('not '):
                        # For negated terms, we'll add them as commented-out facts
                        pred = term[4:].strip()
                        if pred and pred not in fact_set and not pred.startswith('&'):
                            facts.append(f"% Uncomment to test: {pred}.")
                            fact_set.add(pred)
                    else:
                        # For positive terms, add as facts
                        if term and term not in fact_set and not term.startswith('&'):
                            facts.append(f"{term}.")
                            fact_set.add(term)
        
        # Add the facts after the rules
        if facts:
            if not cleaned.endswith('\n'):
                cleaned += '\n'
            cleaned += '\n% Facts derived from rule conditions\n' + '\n'.join(facts)
        
        return cleaned
    
    def validate_syntax(self, repr_text: str) -> Tuple[bool, str]:
        """Validate the syntax of the representation."""
        # Check for deontic operators in deolingo format
        if not re.search(r'&obligatory|&permitted|&forbidden', repr_text):
            return False, "Missing deontic operators (&obligatory, &permitted, &forbidden)"
        
        # Check for syntax errors with periods inside braces
        if re.search(r'{[^}]*\.[^}]*}', repr_text):
            return False, "Periods (.) inside deontic operator braces are not allowed"
        
        # Check for missing closing braces
        if repr_text.count('{') != repr_text.count('}'):
            return False, "Unbalanced braces"
        
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
                fixed = re.sub(r'&obligatory{[^}]+}', "&obligatory{comply}", llm_output)
                fixed = re.sub(r'o\([^)]+\)', "&obligatory{comply}", fixed)
                fixed = "&obligatory{comply} :- relevant_condition.\nrelevant_condition."
            elif "&forbidden" in llm_output or "forbid" in llm_output.lower() or "prohibit" in llm_output.lower():
                fixed = re.sub(r'&forbidden{[^}]+}', "&forbidden{violate}", llm_output)
                fixed = re.sub(r'f\([^)]+\)', "&forbidden{violate}", fixed)
                fixed = "&forbidden{violate} :- not compliance_condition.\ncompliance_condition."
            else:
                fixed = "&obligatory{comply} :- true."
            
            return fixed
        
        return cleaned