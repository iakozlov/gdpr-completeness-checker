% Requirement symbolic representation
&obligatory{take_measures_security_of_processing} :- article_32_applies(processor); ensure_security(processor).
&forbidden{fail_to_take_measures_security_of_processing} :- not (article_32_applies(processor); ensure_security(processor)).

% Facts derived from rule conditions
article_32_applies(processor); ensure_security(processor).
% Uncomment to test: (article_32_applies(processor); ensure_security(processor)).

% DPA segment symbolic representation
&obligatory{apply_standard_contractual_clauses} .
&forbidden{vary_standard_contractual_clauses} :- apply_standard_contractual_clauses.
&forbidden{modify_standard_contractual_clauses} :- apply_standard_contractual_clauses.

% Semantic rules connecting requirement and DPA
ensure_security(processor) :- apply_standard_contractual_clauses.(processor).

% Facts - Required for rule evaluation
apply_standard_contractual_clauses.
article_32_applies(processor).
article_32_applies(processor); ensure_security(processor).
documented_instructions_provided(processor).
documented_instructions_provided.
ensure_security(processor)).
ensure_security(processor).
take_measures_security_of_processing.

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
