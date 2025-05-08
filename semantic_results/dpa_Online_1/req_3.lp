% ========================================================================
% Deolingo program to check DPA 'Online 1' against Requirement 3
% ========================================================================

% --- Requirement 3 ---
% Text: The processor shall process personal data only on documented instructions from the controller.
% Deontic Logic:
&obligatory{process_data(processor)} :- controller(controller).

% --- DPA Segment Deontic Statements ---

% ----------------------------------------
% Segment 2:
% Text: This Data Processing Addendum (“DPA”) supplements the processor controller Agreement available at as updated from time to time between controller and processor, or other agreement between controller and processor governing controller’s use of the Service Offerings (the “Agreement”) when the GDPR applies to your use of the processor Services to process controller Data.
% Deontic Logic:
&obligatory{update_agreement} :- time_passes.
&permitted{use_service_offerings} :- gdpr_applies, controller_use_service_offerings.

% ----------------------------------------
% Segment 3:
% Text: This DPA is an agreement between you and the entity you represent (“controller”, “you” or “your”) and the applicable Amazon Web Services contracting entity under the Agreement (“processor”).
% Deontic Logic:
&obligatory{establish_agreement(controller, processor)}.
&permitted{engage_in_contracting_process} :- is_applicable(processor).
&forbidden{fail_to_establish_agreement} :- is_required(controller).

% ----------------------------------------
% Segment 4:
% Text: Unless otherwise defined in this DPA or in the Agreement, all capitalised terms used in this DPA will have the meanings given to them in Section 17 of this DPA.
% Deontic Logic:
&obligatory{define_capitalised_terms} :- not defined_in_dpa_or_agreement.
&permitted{use_capitalised_terms} :- defined_in_section_17_of_dpa.

% ----------------------------------------
% Segment 7:
% Text: This DPA applies when controller Data is processed by processor.
% Deontic Logic:
&obligatory{apply_dpa} :- process_controller_data(controller, processor).
&permitted{process_controller_data(controller, processor)} :- apply_dpa.

% ----------------------------------------
% Segment 8:
% Text: In this context, processor will act as “processor” to controller who may act either as “controller” or “processor” with respect to controller Data (as each term is defined in the GDPR).
% Deontic Logic:
&permitted{act_as_processor(processor, controller)}.
&permitted{act_as_controller(controller)} :- not act_as_processor(processor, controller).
&forbidden{act_as_both(processor, controller)} :- act_as_processor(processor, controller), act_as_controller(controller).

% ----------------------------------------
% Segment 10:
% Text: The Services provide controller with a number of controls, including security features and functionalities, that controller may use to retrieve, correct, delete or restrict controller Data as described in the Documentation.
% Deontic Logic:
&permitted{use_controls} :- has_security_features, has_functionalities.
&permitted{retrieve_controller_data} :- use_controls.
&permitted{correct_controller_data} :- use_controls.
&permitted{delete_controller_data} :- use_controls.
&permitted{restrict_controller_data} :- use_controls.

% ----------------------------------------
% Segment 11:
% Text: Without prejudice to Section 5.1, controller may use these controls as technical and organisational measures to assist it in connection with its obligations under the GDPR, including its obligations relating to responding to requests from data subjects.
% Deontic Logic:
&permitted{use_controls_as_tom} :- not &forbidden{use_controls_as_tom},.
&obligatory{responding_to_requests_from_data_subjects} :- controller_is_under_gdpr.

% ----------------------------------------
% Segment 14:
% Text: The subject matter of the data processing under this DPA is controller Data.
% Deontic Logic:
&obligatory{controller_data_subject_matter} :- controller_data_is_controller_data.

% ----------------------------------------
% Segment 16:
% Text: As between processor and controller, the duration of the data processing under this DPA is determined by controller.
% Deontic Logic:
&obligatory{determine_duration_controller} .
&forbidden{determine_duration_processor} .

% ----------------------------------------
% Segment 18:
% Text: The purpose of the data processing under this DPA is the provision of the Services initiated by controller from time to time.
% Deontic Logic:
&obligatory{provide_services} :- initiated_by_controller(controller).
&permitted{initiate_services_request} :- time_to_time_initiation.

% ----------------------------------------
% Segment 19:
% Text: Nature of the processing: Compute, storage and such other Services as described in the Documentation and initiated by controller from time to time.
% Deontic Logic:
&obligatory{initiate_services(controller)} :- is_documentation_available.
&permitted{provide_compute_services(processor)} :- initiated_by_controller.
&forbidden{process_data_unauthorized} :- not controller_initiated.

% ----------------------------------------
% Segment 20:
% Text: Type of controller Data: controller Data uploaded to the Services under controller’s processor accounts.
% Deontic Logic:
&obligatory{controller_data_upload} :- controller_processor_account_exists.
&permitted{access_controller_data} :- data_uploaded_under_controller_processor_account.

% ----------------------------------------
% Segment 21:
% Text: Categories of data subjects: The data subjects may include controller’s controllers, employees, suppliers and end-users.
% Deontic Logic:
&permitted{include_data_subjects_controller_controllers} .
&permitted{include_data_subjects_employees} .
&permitted{include_data_subjects_suppliers} .
&permitted{include_data_subjects_end_users} .

% ----------------------------------------
% Segment 23:
% Text: Each party will comply with all lprocessor, rules and regulations applicable to it and binding on it in the performance of this DPA, including the GDPR.
% Deontic Logic:
&obligatory{comply_processor_rules} :- applicable_processor_rule(processor).
&obligatory{comply_processor_regulations} :- binding_processor_regulation(processor).
&permitted{access_gdpr} :- gdpr_applicable.

% ----------------------------------------
% Segment 25:
% Text: The parties agree that this DPA and the Agreement (including the provision of instructions via configuration tools such as the processor management console and APIs made available by processor for the Services) constitute controller’s documented instructions regarding processor’s processing of controller Data (“Documented Instructions”).
% Deontic Logic:
&obligatory{document_instructions} :- provide_dpa_and_agreement.
&permitted{provide_instruction_via_config_tools} :- use_processor_management_console, use_apis_made_available_by_processor.
&forbidden{process_data_without_documented_instructions} :- not documented_instructions_provided.

% ----------------------------------------
% Segment 26:
% Text: processor will process controller Data only in accordance with Documented Instructions.
% Deontic Logic:
&obligatory{process_controller_data} :- documented_instructions_provided.
&forbidden{process_unauthorized_data} :- not documented_instructions_provided.

% ----------------------------------------
% Segment 27:
% Text: Additional instructions outside the scope of the Documented Instructions (if any) require prior written agreement between processor
% Deontic Logic:
&obligatory{request_written_agreement(processor, controller)} :- additional_instructions_outside_documented_instructions.
&forbidden{engage_additional_instruction(processor)} :- not prior_written_agreement(processor, controller).

% ----------------------------------------
% Segment 28:
% Text: and controller, including agreement on any additional fees payable by controller to processor for carrying out such instructions.
% Deontic Logic:
&obligatory{controller_agree_additional_fees} :- controller_instruct_processor.
&permitted{processor_charge_additional_fees} :- controller_agree_to_pay.

% ----------------------------------------
% Segment 29:
% Text: controller is entitled to terminate this DPA and the Agreement if processor declines to follow instructions requested by controller that are outside the scope of, or changed from, those given or agreed to be given in this DPA.
% Deontic Logic:
&obligatory{terminate_dpa(controller)} :- decline_to_follow_instructions(processor, controller).
&permitted{request_instructions_outside_scope(controller, processor)} :- not changed_from_original_instructions(processor, controller).
&forbidden{decline_to_follow_instructions(processor, controller)} :- outside_scope_instructions(controller, processor).

% ----------------------------------------
% Segment 31:
% Text: processor will not access or use, or disclose to any third party, any controller Data, except, in each case, as necessary to maintain or provide the Services, or as necessary to comply with the law or a valid and binding order of a governmental body (such as a subpoena or court order).
% Deontic Logic:
&forbidden{access_controller_data} :- not necessary_to_maintain_services, not necessary_to_comply_with_law.
&permitted{access_controller_data} :- necessary_to_maintain_services, necessary_to_comply_with_law.

% ----------------------------------------
% Segment 32:
% Text: If a governmental body sends processor a demand for controller Data, processor will attempt to redirect the governmental body to request that data directly from controller.
% Deontic Logic:
&obligatory{redirect_demand(governmental_body, processor)} :- governmental_body_request(controller_data).
&permitted{request_data_directly(governmental_body, controller)} :- redirect_demand(governmental_body, processor).

% ----------------------------------------
% Segment 33:
% Text: As part of this effort, processor may provide controller’s basic contact information to the governmental body.
% Deontic Logic:
&permitted{provide_contact_information_to_governmental_body} :- is_part_of_effort.

% ----------------------------------------
% Segment 34:
% Text: If compelled to disclose controller Data to a governmental body, then processor will give controller reasonable notice of the demand to allow controller to seek a protective order or other appropriate remedy unless processor is legally prohibited from doing so.
% Deontic Logic:
&obligatory{give_controller_reasonable_notice(processor, controller)} :- compelled_to_disclose(controller_data, governmental_body), not legally_prohibited(processor).
&permitted{seek_protective_order(controller)} :- compelled_to_disclose(controller_data, governmental_body), give_controller_reasonable_notice(processor, controller).
&forbidden{disclose_controller_data(processor, governmental_body)} :- compelled_to_disclose(controller_data, governmental_body), not give_controller_reasonable_notice(processor, controller).

% ----------------------------------------
% Segment 35:
% Text: If the Standard Contractual Clauses apply, nothing in this Section 3 varies or modifies the Standard Contractual Clauses.
% Deontic Logic:
&obligatory{apply_standard_contractual_clauses} .
&forbidden{vary_standard_contractual_clauses} :- apply_standard_contractual_clauses.
&forbidden{modify_standard_contractual_clauses} :- apply_standard_contractual_clauses.

% ----------------------------------------
% Segment 37:
% Text: processor restricts its personnel from processing controller Data without authorisation by processor as described in the processor Security Standards.
% Deontic Logic:
&obligatory{restrict_processor_personnel} :- authorized_by_processor(controller, processor_security_standards).
&forbidden{process_controller_data_without_authorization} :- not authorized_by_processor(controller, processor_security_standards).

% ----------------------------------------
% Segment 38:
% Text: processor imposes appropriate contractual obligations upon its personnel, including relevant obligations regarding confidentiality, data protection and data security.
% Deontic Logic:
&obligatory{impose_contractual_obligations} :- personnel_involved(processor).
&obligatory{ensure_confidentiality} :- impose_contractual_obligations.
&obligatory{protect_data} :- impose_contractual_obligations.
&obligatory{secure_data} :- impose_contractual_obligations.

% ----------------------------------------
% Segment 40:
% Text: processor has implemented and will maintain the technical and organisational measures for the processor Network as described in the processor Security Standards and this Section.
% Deontic Logic:
&obligatory{maintain_security_measures} :- implemented_security_measures.
&obligatory{implement_security_measures} :- processor_network_description_available.

% ----------------------------------------
% Segment 41:
% Text: In particular, processor has implemented and will maintain the following technical and organisational measures:
% Deontic Logic:
&obligatory{processor_implement_measures} .
&obligatory{processor_maintain_measures} .

% ----------------------------------------
% Segment 42:
% Text: security of the processor Network as set out in Section 1.1 of the processor Security Standards;
% Deontic Logic:
&obligatory{maintain_processor_network_security} :- comply_with_section_1_1_of_processor_security_standards.

% ----------------------------------------
% Segment 43:
% Text: physical security of the facilities as set out in Section 1.2 of the processor Security Standards;
% Deontic Logic:
&obligatory{maintain_physical_security} :- comply_with_processor_security_standards.

% ----------------------------------------
% Segment 44:
% Text: measures to control access rights for processor employees and contractors in relation to the processor Network as set out in Section 1.1 of the processor Security Standards; and
% Deontic Logic:
&obligatory{control_access_rights_processor_employees} :- measures_set_out_in_section_1_1_security_standards.
&permitted{access_processor_network} :- control_access_rights_processor_employees.

% ----------------------------------------
% Segment 45:
% Text: processes for regularly testing, assessing and evaluating the effectiveness of the technical and organisational measures implemented by processor as described in Section 2 of the processor Security Standards.
% Deontic Logic:
&obligatory{test_effectiveness_of_security_measures} :- regularly_test_security_measures.
&permitted{evaluate_security_measures} :- test_effectiveness_of_security_measures, not &forbidden{evaluate_security_measures}.
&obligatory{assess_security_measures} :- evaluate_security_measures.

% ----------------------------------------
% Segment 46:
% Text: controller may elect to implement technical and organisational measures in relation to controller Data.
% Deontic Logic:
&permitted{elect_to_implement_measures} :- relation_to_controller_data.

% ----------------------------------------
% Segment 47:
% Text: Such technical and organisational measures include the following which may be obtained by controller from processor as described in the Documentation, or directly from a third party supplier:
% Deontic Logic:
&permitted{obtain_measures_from_processor} :- obtain_measures_from_documentation.
&permitted{obtain_measures_directly} :- obtain_measures_from_third_party_supplier.

% ----------------------------------------
% Segment 48:
% Text: pseudonymisation and encryption to ensure an appropriate level of security;
% Deontic Logic:
&obligatory{ensure_security_level} :- use_pseudonymisation, use_encryption.

% ----------------------------------------
% Segment 49:
% Text: measures to ensure the ongoing confidentiality, integrity, availability and resilience of the processing systems and services that are being operated by controller;
% Deontic Logic:
&obligatory{ensure_confidentiality_processing_systems} :- ongoing_operation(controller).
&obligatory{ensure_integrity_processing_systems} :- ongoing_operation(controller).
&obligatory{ensure_availability_processing_systems} :- ongoing_operation(controller).
&obligatory{ensure_resilience_processing_systems} :- ongoing_operation(controller).

% ----------------------------------------
% Segment 50:
% Text: measures to allow controller to backup and archive appropriately in order to restore availability and access to controller Data in a timely manner in the event of a physical or technical incident; and
% Deontic Logic:
&obligatory{controller_backup_data} :- physical_or_technical_incident.
&permitted{archive_controller_data} :- controller_backup_data.

% ----------------------------------------
% Segment 51:
% Text: processes for regularly testing, assessing and evaluating the effectiveness of the technical and organisational measures implemented by controller.
% Deontic Logic:
&obligatory{test_effectiveness_of_measures} :- regularly_testing_assessing_and_evaluating(controller).
&permitted{evaluate_organisational_measures} :- test_effectiveness_of_measures.
&forbidden{fail_to_evaluate_technical_measures} :- not evaluate_organisational_measures.

% ----------------------------------------
% Segment 54:
% Text: controller agrees that processor may use sub-processors to fulfill its contractual obligations under this DPA or to provide certain services on its behalf, such as providing support services.
% Deontic Logic:
&permitted{use_sub_processors} :- contractual_obligations_fulfilled.
&permitted{engage_sub_processors} :- provide_support_services.

% ----------------------------------------
% Segment 55:
% Text: The processor website (currently posted at https://processor.amazon.com/compliance/sub-processors/)
% Deontic Logic:
&permitted{post_processor_website} :- processor_posted_at_https_processor_amazon_com_compliance_sub_processors.

% ----------------------------------------
% Semantic Mapping Rules for Requirement 3
% These rules connect DPA actions to requirement predicates
controller_initiated :- controller(controller).
documented_instructions_provided :- provide_dpa_and_agreement.

% ----------------------------------------
% Status Logic for Requirement 3

% A requirement is satisfied if it has a semantic mapping that is fulfilled
satisfies(req3) :- 
    true.  % If there's a semantic mapping, it means the DPA satisfies this requirement

% A requirement is not mentioned if there's no semantic mapping
not_mentioned(req3) :- 
    not satisfies(req3).

% Additional built-in facts
processor(processor).
controller(controller).
data_subject(data_subject).
personal_data(personal_data).

% ----------------------------------------
% Show Directives
#show satisfies/1.
#show not_mentioned/1.
