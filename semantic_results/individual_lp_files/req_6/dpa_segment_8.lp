% Requirement symbolic representation
&obligatory{take_measures_security_of_processing} :- article_32_applies(processor); ensure_security(processor).
&forbidden{fail_to_take_measures_security_of_processing} :- not (article_32_applies(processor); ensure_security(processor)).

% Facts derived from rule conditions
article_32_applies(processor); ensure_security(processor).
% Uncomment to test: (article_32_applies(processor); ensure_security(processor)).

% DPA segment symbolic representation
&permitted{act_as_processor(processor, controller)}.
&permitted{act_as_controller(controller)} :- not act_as_processor(processor, controller).
&forbidden{act_as_both(processor, controller)} :- act_as_processor(processor, controller), act_as_controller(controller).

% Semantic rules connecting requirement and DPA
ensure_security(processor) :- act_as_processor(processor, controller).

% Facts - Required for rule evaluation
act_as_controller(controller).
act_as_processor(processor, controller).
act_as_processor(processor.
article_32_applies(processor).
article_32_applies(processor); ensure_security(processor).
controller).
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
