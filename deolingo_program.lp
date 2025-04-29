% Deolingo program for compliance checking
% ===================================================

% --- Requirement Description ---
% The processor shall make available to the controller information necessary to demonstrate compliance with the obligations Article 28 in GDPR.

% --- Requirement Symbolic Representation ---
&obligatory{make_available_information} :- necessary_for_compliance(processor, controller, Article_28_in_GDPR).
&permitted{access_information} :- make_available(processor, information).

% Facts derived from rule conditions
necessary_for_compliance(processor.
controller.
Article_28_in_GDPR).
make_available(processor.
information).

% --- DPA Segment Description ---
% In such a case the Controller shall be entitled to take the measures foreseen in Clause 5 (b).

% --- DPA Segment Symbolic Representation ---
&obligatory{take_measures_clause_5_b} :- condition.

% Facts derived from rule conditions
condition.

% --- Violation Check ---
violation :- &forbidden{X}, X.
compliant :- not violation.

% --- Show Directives ---
#show violation/0.
#show compliant/0.
