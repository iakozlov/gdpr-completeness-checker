
% --- Example Deolingo Program ---
% Requirement: The processor shall not engage a sub-processor without a prior specific or general written authorization of the controller.
&forbidden{engage_sub_processor(processor, sub_processor)} :- not prior_written_authorization(controller).

% DPA Segment: The Processor shall obtain prior written authorization from the Controller before engaging any sub-processor.
&obligatory{processor.obtain(prior written authorization from controller)} :- processor.engaging(sub-processor).
&forbidden{processor.engaging(sub-processor)} :- not processor.obtain(prior written authorization from controller).

% Violation Check
violation :- &forbidden{X}, X.
compliant :- not violation.

% Show Directives
#show violation/0.
#show compliant/0.
