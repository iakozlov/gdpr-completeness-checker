% Requirement definition
&obligatory{protect_data} :- processor(x).

% DPA segment exactly matches the requirement
&obligatory{protect_data} :- processor(x).

% Facts - Define all entities and make sure all predicates in rule bodies exist
processor(x).
protect_data.  % Explicitly state that protect_data is true

% Satisfaction definition - directly checks if the obligation exists and is fulfilled
satisfies(req) :- &obligatory{protect_data}, protect_data.

% Violation definition
violates(req) :- &forbidden{protect_data}.

% Not-mentioned definition
not_mentioned(req) :- not satisfies(req), not violates(req).

% Show directives
#show satisfies/1.
#show violates/1.
#show not_mentioned/1.