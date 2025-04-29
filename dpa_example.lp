% Deolingo program to check DPA text against Requirement rules
% ==========================================================

% --- Requirement Definitions ---
requirement(req7).
req_modality(req7, obligatory).
req_action(req7, assist_controller(processor, respond_to_data_subject_requests)).

% --- DPA Segment Content Representation ---
dpa_states(dpa1, obligatory, redirect_requests(processor, controller)).
dpa_states(dpa2, obligatory, notify_controller(processor, request)).

% --- Facts ---
% Default facts for testing
processor(processor).
sub_processor(sub_processor).
controller(controller).
request(request).
data_subject_request.
-prevented_by_law(processor).

% --- Action relationships ---
assist_controller(processor, respond_to_data_subject_requests) :- 
    redirect_requests(processor, controller),
    notify_controller(processor, request).

% --- Matching and Status Logic ---
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

% Status Rule 3: Non-Mention
not_mentioned(R) :-
    requirement(R),
    not satisfies(R),
    not violates(R).

% --- Show Directives ---
#show satisfies/1.
#show violates/1.
#show not_mentioned/1.
