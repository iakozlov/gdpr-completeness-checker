% Requirement symbolic representation
&obligatory{take_measures_security_of_processing} :- article_32_applies(processor); ensure_security(processor).
&forbidden{fail_to_take_measures_security_of_processing} :- not (article_32_applies(processor); ensure_security(processor)).

% Facts derived from rule conditions
article_32_applies(processor); ensure_security(processor).
% Uncomment to test: (article_32_applies(processor); ensure_security(processor)).

% DPA segment symbolic representation
&obligatory{controller_agree_additional_fees} :- controller_instruct_processor.
&permitted{processor_charge_additional_fees} :- controller_agree_to_pay.

% Semantic rules connecting requirement and DPA
ensure_security(processor) :- controller_instruct_processor.(processor).

% Facts - Required for rule evaluation
article_32_applies(processor).
article_32_applies(processor); ensure_security(processor).
controller_agree_to_pay.
controller_instruct_processor.
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
