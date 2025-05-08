% Requirement symbolic representation
&obligatory{take_measures_security_of_processing} :- article_32_applies(processor); ensure_security(processor).
&forbidden{fail_to_take_measures_security_of_processing} :- not (article_32_applies(processor); ensure_security(processor)).

% Facts derived from rule conditions
article_32_applies(processor); ensure_security(processor).
% Uncomment to test: (article_32_applies(processor); ensure_security(processor)).

% DPA segment symbolic representation
&permitted{post_processor_website} :- processor_posted_at_https_processor_amazon_com_compliance_sub_processors.

% Semantic rules connecting requirement and DPA
ensure_security(processor) :- processor_posted_at_https_processor_amazon_com_compliance_sub_processors.(processor).

% Facts - Required for rule evaluation
article_32_applies(processor).
article_32_applies(processor); ensure_security(processor).
documented_instructions_provided(processor).
documented_instructions_provided.
ensure_security(processor)).
ensure_security(processor).
processor_posted_at_https_processor_amazon_com_compliance_sub_processors.
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
