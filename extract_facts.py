# extract_facts.py
import re

def extract_facts_from_symbolic(symbolic_repr):
    """Extract facts from symbolic representation."""
    facts = set()
    
    # Extract predicates from rule bodies
    for line in symbolic_repr.split('\n'):
        if ':-' in line:
            body = line.split(':-')[1].strip()
            if body.endswith('.'):
                body = body[:-1]
            
            # Split by common operators
            terms = re.split(r'[,;|&]', body)
            for term in terms:
                term = term.strip()
                if not term or term.startswith('not ') or term.startswith('&'):
                    continue
                
                # Clean up the term and add as fact
                fact = term.strip()
                if not fact.endswith('.'):
                    fact += '.'
                facts.add(fact)
    
    # Extract predicates from deontic operators
    for op in ['&obligatory', '&permitted', '&forbidden']:
        pattern = rf'{op}{{([^}}]+)}}'
        matches = re.findall(pattern, symbolic_repr)
        for match in matches:
            pred = match.strip()
            if not pred.endswith('.'):
                pred += '.'
            facts.add(pred)
    
    # Extract standard predicates with arguments
    arg_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\(([^)]+)\)'
    for match in re.finditer(arg_pattern, symbolic_repr):
        pred_name = match.group(1)
        args = match.group(2)
        
        # Skip if part of deontic operator
        if any(op in symbolic_repr[:match.start()] for op in ['&obligatory{', '&permitted{', '&forbidden{']):
            continue
        
        fact = f"{pred_name}({args})."
        facts.add(fact)
    
    return facts