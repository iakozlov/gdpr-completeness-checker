# Requirements Completeness Verification (RCV) Approach

## Overview

The RCV approach is a hybrid symbolic-neural method for automatically verifying whether Data Processing Agreements (DPAs) satisfy GDPR processor obligations. It combines Large Language Models (LLMs) for natural language understanding with Answer Set Programming (ASP) for formal logical reasoning.

## Core Algorithm

### Phase 1: Requirement Formalization
1. **Symbolic Representation**: GDPR processor obligations are encoded as deontic logic rules using obligation operators (`&obligatory{}`)
2. **Predicate Extraction**: Each requirement is decomposed into atomic predicates representing roles, conditions, and actions
3. **Logic Programming Rules**: Requirements are expressed as ASP rules with triggering conditions and obligated outcomes

### Phase 2: Document Processing
1. **Segmentation**: DPA documents are divided into coherent segments (paragraphs or logical units)
2. **Classification**: Each segment is classified against all requirements to determine potential relevance
3. **Fact Extraction**: For relevant segments, atomic facts are extracted that correspond to requirement predicates
4. **Validation**: Extracted facts are filtered to ensure only processor-related obligations are considered

### Phase 3: Logical Verification
1. **Program Construction**: For each segment-requirement pair:
   - Combine the symbolic requirement rule
   - Add extracted facts as ground truth
   - Include deontic status mapping rules
2. **ASP Solving**: Execute the combined program using a deontic logic solver
3. **Status Determination**: Map solver output to requirement satisfaction status:
   - `SATISFIED`: Obligation is fulfilled (action performed)
   - `VIOLATED`: Obligation exists but is breached
   - `NOT_MENTIONED`: No relevant obligation found

### Phase 4: Completeness Assessment
1. **Requirement Aggregation**: Determine which requirements are satisfied across all document segments
2. **Coverage Analysis**: Compare satisfied requirements against the complete set of GDPR obligations
3. **Gap Identification**: Identify missing or unsatisfied requirements
4. **Completeness Scoring**: Calculate overall compliance metrics

## Data Flow

```
DPA Document → Segments → Classification → Fact Extraction → ASP Programs → Deontic Solver → Status Results → Completeness Analysis
```

### Key Transformations

1. **Natural Language → Symbolic Logic**: LLM converts segment text into logical predicates
2. **Predicates → ASP Facts**: Extracted predicates become ground facts in logic programs
3. **Deontic Rules → Satisfaction Status**: ASP solver determines obligation fulfillment
4. **Status Collection → Completeness Verdict**: Aggregate individual results into overall assessment

## Logical Foundation

The approach leverages **deontic logic** principles where:
- Obligations are triggered by contextual conditions (e.g., processor role)
- Fulfillment requires actual performance of obligated actions
- Violations occur when obligations exist but actions are not performed
- Undetermined states indicate insufficient information

### Satisfaction Criteria

An obligation is considered **satisfied** when:
1. The obligation rule is triggered (conditions met)
2. The obligated action is actually performed (evidenced in text)

This ensures that segments demonstrating actual compliance behavior are correctly recognized, regardless of explicit termination conditions or contextual triggers.

## Advantages

- **Formal Verification**: Uses logical reasoning rather than pattern matching
- **Interpretability**: Clear mapping from text to logical facts to conclusions
- **Precision**: Distinguishes between different types of non-compliance
- **Scalability**: Systematic processing of large document collections
- **Consistency**: Uniform application of GDPR requirements across documents

## Applications

- **Compliance Auditing**: Automated verification of DPA completeness
- **Gap Analysis**: Identification of missing contractual provisions
- **Quality Assurance**: Systematic review of legal document adequacy
- **Regulatory Assessment**: Support for GDPR compliance evaluation 