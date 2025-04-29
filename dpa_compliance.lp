% Deolingo program to check DPA text against Requirement rules
% ==========================================================

% --- Requirement Definitions ---
requirement(req1).
req_modality(req1, obligatory).
req_action(req1, impose_obligations(processor, sub_processor)).

% --- DPA Segment Content Representation ---
dpa_states(dpa1, obligatory, process_personal_data(processor)).

% --- Facts ---
% Default facts for testing
processor(processor).
sub_processor(sub_processor).
controller(controller).

% --- Matching and Status Logic ---
% Helper: Does the DPA mention a specific action with any modality?
dpa_mentions_action(Action) :- dpa_states(_, _, Action).

% Status Rule 1: Satisfaction
satisfies(R) :-
    requirement(R),
    req_modality(R, M),
    req_action(R, A),
    dpa_states(_, M, A).

% Status Rule 2: Violation
violates(R) :-
    requirement(R),
    req_modality(R, obligatory),
    req_action(R, A),
    dpa_states(_, forbidden, A).

violates(R) :-
    requirement(R),
    req_modality(R, permitted),
    req_action(R, A),
    dpa_states(_, forbidden, A).

% Status Rule 3: Non-Mention
not_mentioned(R) :-
    requirement(R),
    not satisfies(R),
    not violates(R).

% --- Show Directives ---
#show satisfies/1.
#show violates/1.
#show not_mentioned/1.