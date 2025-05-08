% Requirement symbolic representation
&obligatory{take_measures_security_of_processing} :- article_32_applies(processor); ensure_security(processor).
&forbidden{fail_to_take_measures_security_of_processing} :- not (article_32_applies(processor); ensure_security(processor)).

% Facts derived from rule conditions
article_32_applies(processor); ensure_security(processor).
% Uncomment to test: (article_32_applies(processor); ensure_security(processor)).

% DPA segment symbolic representation
&obligatory{give_controller_reasonable_notice(processor, controller)} :- compelled_to_disclose(controller_data, governmental_body), not legally_prohibited(processor).
&permitted{seek_protective_order(controller)} :- compelled_to_disclose(controller_data, governmental_body), give_controller_reasonable_notice(processor, controller).
&forbidden{disclose_controller_data(processor, governmental_body)} :- compelled_to_disclose(controller_data, governmental_body), not give_controller_reasonable_notice(processor, controller).

% Semantic rules connecting requirement and DPA
ensure_security(processor) :- give_controller_reasonable_notice(processor, controller).

% Facts - Required for rule evaluation
article_32_applies(processor).
article_32_applies(processor); ensure_security(processor).
compelled_to_disclose(controller_data.
controller).
documented_instructions_provided(processor).
documented_instructions_provided.
ensure_security(processor)).
ensure_security(processor).
give_controller_reasonable_notice(processor, controller).
give_controller_reasonable_notice(processor.
governmental_body).
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
