% Deolingo program to check DPA against Requirement 19
% ==========================================================

% --- Requirement Definition ---
% Original text: When assessing the level of security, the processor shall take into account the risk of accidental or unlawful destruction, loss, alternation, unauthorized disclosure of or access to the personal data transmitted, stored or processed.

requirement(req19). % Requirement req19 ID
req_modality(req19, obligatory).
req_action(req19, take_into_account_risk(processor, personal_data)).

% --- DPA Segment Content Representation ---
% Represents what the DPA explicitly states using these facts.

% DPA Segment 2: This Data Processing Addendum (“DPA”) supplements ...
dpa_states(dpa2, obligatory, comply(processor, controller)). % DPA statement about comply(processor, controller)

% DPA Segment 3: This DPA is an agreement between you and the entit...
dpa_states(dpa3, obligatory, comply(processor, controller)). % DPA statement about comply(processor, controller)

% DPA Segment 4: Unless otherwise defined in this DPA or in the Agr...
dpa_states(dpa4, obligatory, comply(processor)). % DPA statement about comply(processor)

% DPA Segment 7: This DPA applies when controller Data is processed...
dpa_states(dpa7, obligatory, comply(processor, controller)). % DPA statement about comply(processor, controller)

% DPA Segment 8: In this context, processor will act as “processor”...
dpa_states(dpa8, obligatory, comply(processor, controller)). % DPA statement about comply(processor, controller)

% DPA Segment 10: The Services provide controller with a number of c...
dpa_states(dpa10, obligatory, make_available_information_to_controller_data(processor, controller)). % DPA statement about make_available_information_to_controller_data(processor, controller)

% DPA Segment 11: Without prejudice to Section 5.1, controller may u...
dpa_states(dpa11, obligatory, assist(processor, controller)). % DPA statement about assist(processor, controller)

% DPA Segment 14: The subject matter of the data processing under th...
dpa_states(dpa14, obligatory, process_personal_data_controller_data(processor, controller)). % DPA statement about process_personal_data_controller_data(processor, controller)

% DPA Segment 16: As between processor and controller, the duration ...
dpa_states(dpa16, obligatory, make_available_information_to_controller(processor, controller)). % DPA statement about make_available_information_to_controller(processor, controller)

% DPA Segment 18: The purpose of the data processing under this DPA ...
dpa_states(dpa18, obligatory, process_personal_data_personal_data(processor, controller)). % DPA statement about process_personal_data_personal_data(processor, controller)

% DPA Segment 19: Nature of the processing: Compute, storage and suc...
dpa_states(dpa19, obligatory, process_personal_data_personal_data(processor, controller)). % DPA statement about process_personal_data_personal_data(processor, controller)

% DPA Segment 20: Type of controller Data: controller Data uploaded ...
dpa_states(dpa20, obligatory, make_available_information_to_controller_data(processor, controller)). % DPA statement about make_available_information_to_controller_data(processor, controller)

% DPA Segment 21: Categories of data subjects: The data subjects may...
dpa_states(dpa21, permitted, process_personal_data_data_subjects(processor, controller)). % DPA statement about process_personal_data_data_subjects(processor, controller)

% DPA Segment 23: Each party will comply with all lprocessor, rules ...
dpa_states(dpa23, obligatory, comply(processor)). % DPA statement about comply(processor)

% DPA Segment 25: The parties agree that this DPA and the Agreement ...
dpa_states(dpa25, obligatory, comply(processor, controller)). % DPA statement about comply(processor, controller)

% DPA Segment 26: processor will process controller Data only in acc...
dpa_states(dpa26, obligatory, process_personal_data_controller_data(processor, controller)). % DPA statement about process_personal_data_controller_data(processor, controller)

% DPA Segment 27: Additional instructions outside the scope of the D...
dpa_states(dpa27, obligatory, prior_written_agreement(processor)). % DPA statement about prior_written_agreement(processor)

% DPA Segment 28: and controller, including agreement on any additio...
dpa_states(dpa28, permitted, make_available_information_to_controller(processor, controller)). % DPA statement about make_available_information_to_controller(processor, controller)

% DPA Segment 29: controller is entitled to terminate this DPA and t...
dpa_states(dpa29, permitted, terminate_dpa_and_agreement(processor, controller)). % DPA statement about terminate_dpa_and_agreement(processor, controller)

% DPA Segment 31: processor will not access or use, or disclose to a...
dpa_states(dpa31, forbidden, access_unauthorized_personal_data_controller_data(processor, controller)). % DPA statement about access_unauthorized_personal_data_controller_data(processor, controller)

% DPA Segment 32: If a governmental body sends processor a demand fo...
dpa_states(dpa32, permitted, inform_legal_requirement_governmental_body(processor, controller)). % DPA statement about inform_legal_requirement_governmental_body(processor, controller)

% DPA Segment 33: As part of this effort, processor may provide cont...
dpa_states(dpa33, permitted, make_available_information_to_controller_governmental_body(processor, controller)). % DPA statement about make_available_information_to_controller_governmental_body(processor, controller)

% DPA Segment 34: If compelled to disclose controller Data to a gove...
dpa_states(dpa34, obligatory, inform(processor, controller)). % DPA statement about inform(processor, controller)

% DPA Segment 35: If the Standard Contractual Clauses apply, nothing...
dpa_states(dpa35, obligatory, alter_personal_data_standard_contractual_clauses(processor)). % DPA statement about alter_personal_data_standard_contractual_clauses(processor)

% DPA Segment 37: processor restricts its personnel from processing ...
dpa_states(dpa37, obligatory, process_personal_data_controller_data(processor, controller)). % DPA statement about process_personal_data_controller_data(processor, controller)

% DPA Segment 38: processor imposes appropriate contractual obligati...
dpa_states(dpa38, obligatory, impose_obligations_subprocessors_personnel(processor, personal_data)). % DPA statement about impose_obligations_subprocessors_personnel(processor, personal_data)

% DPA Segment 40: processor has implemented and will maintain the te...
dpa_states(dpa40, obligatory, comply(processor)). % DPA statement about comply(processor)

% DPA Segment 41: In particular, processor has implemented and will ...
dpa_states(dpa41, obligatory, retain_personal_data_personal_data(processor)). % DPA statement about retain_personal_data_personal_data(processor)

% DPA Segment 42: security of the processor Network as set out in Se...
dpa_states(dpa42, obligatory, ensure_security_of_processing(processor)). % DPA statement about ensure_security_of_processing(processor)

% DPA Segment 43: physical security of the facilities as set out in ...
dpa_states(dpa43, obligatory, comply(processor)). % DPA statement about comply(processor)

% DPA Segment 44: measures to control access rights for processor em...
dpa_states(dpa44, obligatory, take_into_account_risk_processor_network(processor)). % DPA statement about take_into_account_risk_processor_network(processor)

% DPA Segment 45: processes for regularly testing, assessing and eva...
dpa_states(dpa45, obligatory, take_into_account_risk_risk(processor)). % DPA statement about take_into_account_risk_risk(processor)

% DPA Segment 46: controller may elect to implement technical and or...
dpa_states(dpa46, permitted, take_into_account_risk_controller_data(processor, controller)). % DPA statement about take_into_account_risk_controller_data(processor, controller)

% DPA Segment 47: Such technical and organisational measures include...
dpa_states(dpa47, obligatory, make_available_information_to_controller(processor, controller)). % DPA statement about make_available_information_to_controller(processor, controller)

% DPA Segment 48: pseudonymisation and encryption to ensure an appro...
dpa_states(dpa48, obligatory, take_into_account_risk_security_of_processing(processor)). % DPA statement about take_into_account_risk_security_of_processing(processor)

% DPA Segment 49: measures to ensure the ongoing confidentiality, in...
dpa_states(dpa49, obligatory, take_into_account_risk_risk(processor, controller)). % DPA statement about take_into_account_risk_risk(processor, controller)

% DPA Segment 50: measures to allow controller to backup and archive...
dpa_states(dpa50, obligatory, make_available_information_to_controller_data(processor, controller)). % DPA statement about make_available_information_to_controller_data(processor, controller)

% DPA Segment 51: processes for regularly testing, assessing and eva...
dpa_states(dpa51, obligatory, take_into_account_risk(processor, controller)). % DPA statement about take_into_account_risk(processor, controller)

% DPA Segment 54: controller agrees that processor may use sub-proce...
dpa_states(dpa54, permitted, engage_sub_processor_sub_processor_subprocessor(processor, sub_processor)). % DPA statement about engage_sub_processor_sub_processor_subprocessor(processor, sub_processor)

% DPA Segment 55: The processor website (currently posted at https:/...
dpa_states(dpa55, permitted, make_available_information_to_controller_httpsprocessor_amazon_comcompliancesubprocessors(processor, sub_processor)). % DPA statement about make_available_information_to_controller_httpsprocessor_amazon_comcompliancesubprocessors(processor, sub_processor)

% DPA Segment 56: lists sub-processors that are currently engaged by...
dpa_states(dpa56, obligatory, make_available_information_to_controller(processor, sub_processor)). % DPA statement about make_available_information_to_controller(processor, sub_processor)

% DPA Segment 57: At least 30 days before processor engages any new ...
dpa_states(dpa57, obligatory, update_website(processor, sub_processor)). % DPA statement about update_website(processor, sub_processor)

% DPA Segment 58: If controller objects to a new sub-processor, then...
dpa_states(dpa58, permitted, move_data(processor, sub_processor)). % DPA statement about move_data(processor, sub_processor)

% DPA Segment 59: controller consents to processor’s use of sub-proc...
dpa_states(dpa59, permitted, engage_sub_processor_sub_processor_subprocessors(processor, sub_processor)). % DPA statement about engage_sub_processor_sub_processor_subprocessors(processor, sub_processor)

% DPA Segment 60: Except as set forth in this Section, or as control...
dpa_states(dpa60, obligatory, engage_sub_processor_sub_processor(processor, sub_processor)). % DPA statement about engage_sub_processor_sub_processor(processor, sub_processor)

% DPA Segment 62: Where processor authorises any sub-processor as de...
dpa_states(dpa62, obligatory, impose_obligations_subprocessors_subprocessor(processor, sub_processor)). % DPA statement about impose_obligations_subprocessors_subprocessor(processor, sub_processor)

% DPA Segment 63: processor will restrict the sub-processor’s access...
dpa_states(dpa63, obligatory, impose_obligations_subprocessors_subprocessor(processor, sub_processor)). % DPA statement about impose_obligations_subprocessors_subprocessor(processor, sub_processor)

% DPA Segment 64: processor will enter into a written agreement with...
dpa_states(dpa64, obligatory, enter_into_written_agreement_subprocessor(processor, sub_processor)). % DPA statement about enter_into_written_agreement_subprocessor(processor, sub_processor)

% DPA Segment 65: processor will remain responsible for its complian...
dpa_states(dpa65, obligatory, remain_fully_liable(processor, sub_processor)). % DPA statement about remain_fully_liable(processor, sub_processor)

% DPA Segment 67: Taking into account the nature of the Services, pr...
dpa_states(dpa67, permitted, make_available_information_to_controller_information(processor, controller)). % DPA statement about make_available_information_to_controller_information(processor, controller)

% DPA Segment 68: Should a data subject contact processor with regar...
dpa_states(dpa68, obligatory, make_available_information_to_controller(processor, controller)). % DPA statement about make_available_information_to_controller(processor, controller)

% DPA Segment 70: processor makes available a number of security fea...
dpa_states(dpa70, permitted, make_available_information_to_controller_security_features(processor, controller)). % DPA statement about make_available_information_to_controller_security_features(processor, controller)

% DPA Segment 71: controller is responsible for (a) implementing the...
dpa_states(dpa71, obligatory, implement_measures_measures(processor, controller)). % DPA statement about implement_measures_measures(processor, controller)

% DPA Segment 72: the controls available in connection with the Serv...
dpa_states(dpa72, obligatory, make_available_information_to_controller_data(processor, controller)). % DPA statement about make_available_information_to_controller_data(processor, controller)

% DPA Segment 73: backups and routine archiving of controller Data),...
dpa_states(dpa73, obligatory, retain_personal_data_controller_data(processor, controller)). % DPA statement about retain_personal_data_controller_data(processor, controller)

% DPA Segment 76: processor will (a) notify controller of a Security...
dpa_states(dpa76, obligatory, notify_security_incident(processor, controller)). % DPA statement about notify_security_incident(processor, controller)

% DPA Segment 78: To assist controller in relation to any personal d...
dpa_states(dpa78, obligatory, assist_data_subject(processor, controller)). % DPA statement about assist_data_subject(processor, controller)

% DPA Segment 81: an unsuccessful Security Incident will not be subj...
dpa_states(dpa81, forbidden, remain_fully_liable(processor)). % DPA statement about remain_fully_liable(processor)

% DPA Segment 82: An unsuccessful Security Incident is one that resu...
dpa_states(dpa82, permitted, access_unauthorized_personal_data_data(processor, controller)). % DPA statement about access_unauthorized_personal_data_data(processor, controller)

% DPA Segment 83: processor’s obligation to report or respond to a S...
dpa_states(dpa83, obligatory, fulfill_respond_request(processor)). % DPA statement about fulfill_respond_request(processor)

% DPA Segment 85: Notification(s) of Security Incidents, if any, wil...
dpa_states(dpa85, obligatory, inform_legal_requirement(processor, controller)). % DPA statement about inform_legal_requirement(processor, controller)

% DPA Segment 86: It is controller’s sole responsibility to ensure c...
dpa_states(dpa86, obligatory, make_available_information_to_controller(processor, controller)). % DPA statement about make_available_information_to_controller(processor, controller)

% DPA Segment 89: In addition to the information contained in this D...
dpa_states(dpa89, permitted, make_available_information_to_controller(processor, controller)). % DPA statement about make_available_information_to_controller(processor, controller)

% DPA Segment 90: the certificates issued in relation to the ISO 270...
dpa_states(dpa90, obligatory, make_available_information_to_controller_certificates(processor)). % DPA statement about make_available_information_to_controller_certificates(processor)

% DPA Segment 91: the System and Organization Controls (SOC) 1 Repor...
dpa_states(dpa91, obligatory, make_available_information_to_controller_soc_1_report(processor)). % DPA statement about make_available_information_to_controller_soc_1_report(processor)

% DPA Segment 92: implemented by processor that replace or are subst...
dpa_states(dpa92, obligatory, comply(processor)). % DPA statement about comply(processor)

% DPA Segment 94: processor uses external auditors to verify the ade...
dpa_states(dpa94, obligatory, conduct_inspection_auditor_site_data_centers(processor, personal_data)). % DPA statement about conduct_inspection_auditor_site_data_centers(processor, personal_data)

% DPA Segment 95: This audit: (a) will be performed at least annuall...
dpa_states(dpa95, obligatory, conduct_inspection_auditor_site_auditor(processor)). % DPA statement about conduct_inspection_auditor_site_auditor(processor)

% DPA Segment 97: At controller’s written request, and provided that...
dpa_states(dpa97, obligatory, make_available_information_to_controller(processor, controller)). % DPA statement about make_available_information_to_controller(processor, controller)

% DPA Segment 98: Privacy Impact Assessment and Prior Consultation.
dpa_states(dpa98, obligatory, conduct_inspection_auditor_site(processor)). % DPA statement about conduct_inspection_auditor_site(processor)

% DPA Segment 99: Taking into account the nature of the Services and...
dpa_states(dpa99, obligatory, assist_complyprocessor(processor, controller)). % DPA statement about assist_complyprocessor(processor, controller)

% DPA Segment 101: controller agrees to exercise any right it may hav...
dpa_states(dpa101, obligatory, inform_legal_requirement(processor, controller)). % DPA statement about inform_legal_requirement(processor, controller)

% DPA Segment 102: If controller wishes to change this instruction re...
dpa_states(dpa102, permitted, request_change_instruction(processor, controller)). % DPA statement about request_change_instruction(processor, controller)

% DPA Segment 103: If processor declines to follow any instruction re...
dpa_states(dpa103, obligatory, fail_assist_supervisory_authority(processor, controller)). % DPA statement about fail_assist_supervisory_authority(processor, controller)

% DPA Segment 104: If the Standard Contractual Clauses apply, nothing...
dpa_states(dpa104, forbidden, alter_personal_data_standard_contractual_clauses(processor, personal_data)). % DPA statement about alter_personal_data_standard_contractual_clauses(processor, personal_data)

% DPA Segment 107: controller may specify the location(s) where contr...
dpa_states(dpa107, permitted, make_available_information_to_controller_region(processor, controller)). % DPA statement about make_available_information_to_controller_region(processor, controller)

% DPA Segment 108: Once controller has made its choice, processor wil...
dpa_states(dpa108, obligatory, transfer_data(processor, controller)). % DPA statement about transfer_data(processor, controller)

% DPA Segment 109: If the Standard Contractual Clauses apply, nothing...
dpa_states(dpa109, forbidden, alter_personal_data_standard_contractual_clauses(processor)). % DPA statement about alter_personal_data_standard_contractual_clauses(processor)

% DPA Segment 111: The Standard Contractual Clauses will apply to con...
dpa_states(dpa111, obligatory, comply(processor, controller)). % DPA statement about comply(processor, controller)

% DPA Segment 112: The Standard Contractual Clauses will not apply to...
dpa_states(dpa112, forbidden, apply_standard_contractual_clauses_data(processor, controller)). % DPA statement about apply_standard_contractual_clauses_data(processor, controller)

% DPA Segment 113: Notwithstanding the foregoing, the Standard Contra...
dpa_states(dpa113, forbidden, apply_standard_contractual_clauses(processor, personal_data)). % DPA statement about apply_standard_contractual_clauses(processor, personal_data)

% DPA Segment 115: This DPA shall continue in force until the termina...
dpa_states(dpa115, obligatory, remain_fully_liable(processor)). % DPA statement about remain_fully_liable(processor)

% DPA Segment 116: Return or Deletion of controller Data.
dpa_states(dpa116, obligatory, retain_personal_data_data(processor, controller)). % DPA statement about retain_personal_data_data(processor, controller)

% DPA Segment 117: The Services provide controller with controls that...
dpa_states(dpa117, obligatory, make_available_information_to_controller_controller_data(processor, controller)). % DPA statement about make_available_information_to_controller_controller_data(processor, controller)

% DPA Segment 118: Up to the Termination Date, controller will contin...
dpa_states(dpa118, obligatory, make_available_information_to_controller_controller_data(processor, controller)). % DPA statement about make_available_information_to_controller_controller_data(processor, controller)

% DPA Segment 119: For 90 days following the Termination Date, contro...
dpa_states(dpa119, permitted, make_available_information_to_controller_data(processor, controller)). % DPA statement about make_available_information_to_controller_data(processor, controller)

% DPA Segment 120: No later than the end of this 90 day period, contr...
dpa_states(dpa120, obligatory, close_account(processor, controller)). % DPA statement about close_account(processor, controller)

% DPA Segment 121: processor will delete controller Data when request...
dpa_states(dpa121, obligatory, destroy_personal_data_controller_data(processor, controller)). % DPA statement about destroy_personal_data_controller_data(processor, controller)

% DPA Segment 123: Where controller Data becomes subject to confiscat...
dpa_states(dpa123, obligatory, inform(processor, controller)). % DPA statement about inform(processor, controller)

% DPA Segment 124: processor will, without undue delay, notify all re...
dpa_states(dpa124, obligatory, notify_relevant_parties(processor)). % DPA statement about notify_relevant_parties(processor)

% DPA Segment 125: creditors, bankruptcy trustee) that any controller...
dpa_states(dpa125, obligatory, retain_personal_data_data(processor, controller)). % DPA statement about retain_personal_data_data(processor, controller)

% DPA Segment 127: Except as amended by this DPA, the Agreement will ...
dpa_states(dpa127, obligatory, remain_fully_liable(processor)). % DPA statement about remain_fully_liable(processor)

% DPA Segment 128: If there is a conflict between any other agreement...
dpa_states(dpa128, obligatory, alter_agreement(processor)). % DPA statement about alter_agreement(processor)

% DPA Segment 130: Unless otherwise defined in the Agreement, all cap...
dpa_states(dpa130, obligatory, make_available_information_to_controller_information(processor)). % DPA statement about make_available_information_to_controller_information(processor)

% DPA Segment 131: “processor Network” means processor’s data center ...
dpa_states(dpa131, obligatory, retain_personal_data_data_center_facilities(processor, personal_data)). % DPA statement about retain_personal_data_data_center_facilities(processor, personal_data)

% DPA Segment 132: “processor Security Standards” means the security ...
dpa_states(dpa132, obligatory, comply(processor)). % DPA statement about comply(processor)

% DPA Segment 133: “controller” means you or the entity you represent...
dpa_states(dpa133, obligatory, comply(processor, controller)). % DPA statement about comply(processor, controller)

% DPA Segment 134: “controller Data” means the “personal data” (as de...
dpa_states(dpa134, permitted, make_available_information_to_controller_personal_data(processor, controller)). % DPA statement about make_available_information_to_controller_personal_data(processor, controller)

% DPA Segment 135: “EEA” means the European Economic Area.
dpa_states(dpa135, obligatory, inform_definition_of_eea(processor)). % DPA statement about inform_definition_of_eea(processor)

% DPA Segment 136: “GDPR” means Regulation 2016/679 of the European P...
dpa_states(dpa136, obligatory, inform(processor, personal_data)). % DPA statement about inform(processor, personal_data)

% DPA Segment 137: “processing” has the meaning given to it in the GD...
dpa_states(dpa137, obligatory, comply(processor)). % DPA statement about comply(processor)

% DPA Segment 138: “Security Incident” means a breach of processor’s ...
dpa_states(dpa138, forbidden, lose_personal_data_controller_data(processor, controller)). % DPA statement about lose_personal_data_controller_data(processor, controller)

% DPA Segment 139: “Standard Contractual Clauses” means Annex 2, atta...
dpa_states(dpa139, obligatory, alter_personal_data(processor, personal_data)). % DPA statement about alter_personal_data(processor, personal_data)

% DPA Segment 142: Capitalised terms not otherwise defined in this do...
dpa_states(dpa142, obligatory, comply(processor)). % DPA statement about comply(processor)

% DPA Segment 144: processor will maintain an information security pr...
dpa_states(dpa144, obligatory, retain_personal_data_controller_data(processor, controller)). % DPA statement about retain_personal_data_controller_data(processor, controller)

% DPA Segment 145: processor will designate one or more employees to ...
dpa_states(dpa145, obligatory, designate_employees_information_security_program(processor)). % DPA statement about designate_employees_information_security_program(processor)

% DPA Segment 146: The information security program will include the ...
dpa_states(dpa146, obligatory, make_available_information_to_controller_information_security_program(processor)). % DPA statement about make_available_information_to_controller_information_security_program(processor)

% DPA Segment 148: The processor Network will be electronically acces...
dpa_states(dpa148, permitted, make_available_information_to_controller_employees(processor)). % DPA statement about make_available_information_to_controller_employees(processor)

% DPA Segment 149: processor will maintain access controls and polici...
dpa_states(dpa149, obligatory, maintain_access_controls_processor_network(processor)). % DPA statement about maintain_access_controls_processor_network(processor)

% DPA Segment 150: processor will maintain corrective action and inci...
dpa_states(dpa150, obligatory, retain_personal_data_security_threats(processor)). % DPA statement about retain_personal_data_security_threats(processor)

% DPA Segment 153: Physical components of the processor Network are h...
dpa_states(dpa153, forbidden, lose_personal_data(processor)). % DPA statement about lose_personal_data(processor)

% DPA Segment 154: Physical barrier controls are used to prevent unau...
dpa_states(dpa154, obligatory, conduct_inspection_auditor_site_facilities(processor)). % DPA statement about conduct_inspection_auditor_site_facilities(processor)

% DPA Segment 155: Passage through the physical barriers at the Facil...
dpa_states(dpa155, obligatory, access_unauthorized_personal_data_facilities(processor)). % DPA statement about access_unauthorized_personal_data_facilities(processor)

% DPA Segment 156: or validation by human security personnel (e.g., c...
dpa_states(dpa156, forbidden, access_unauthorized_personal_data(processor)). % DPA statement about access_unauthorized_personal_data(processor)

% DPA Segment 157: Employees and contractors are assigned photo-ID ba...
dpa_states(dpa157, obligatory, make_available_information_to_controller_photoid_badges(processor)). % DPA statement about make_available_information_to_controller_photoid_badges(processor)

% DPA Segment 158: Visitors are required to sign-in with designated p...
dpa_states(dpa158, obligatory, signin_visitor_designated_personnel(processor)). % DPA statement about signin_visitor_designated_personnel(processor)

% DPA Segment 160: processor provides access to the Facilities to tho...
dpa_states(dpa160, obligatory, make_available_information_to_controller_facilities(processor)). % DPA statement about make_available_information_to_controller_facilities(processor)

% DPA Segment 161: When an employee or contractor no longer has a bus...
dpa_states(dpa161, obligatory, alter_personal_data_access_privileges(processor)). % DPA statement about alter_personal_data_access_privileges(processor)

% DPA Segment 163: All access points (other than main entry doors) ar...
dpa_states(dpa163, obligatory, retain_personal_data_access_points(processor)). % DPA statement about retain_personal_data_access_points(processor)

% DPA Segment 164: Access points to the Facilities are monitored by v...
dpa_states(dpa164, forbidden, access_unauthorized_personal_data(processor)). % DPA statement about access_unauthorized_personal_data(processor)

% DPA Segment 165: processor also maintains electronic intrusion dete...
dpa_states(dpa165, obligatory, take_into_account_risk_facilities(processor)). % DPA statement about take_into_account_risk_facilities(processor)

% DPA Segment 166: with door contacts, glass breakage devices, interi...
dpa_states(dpa166, forbidden, access_unauthorized_personal_data(processor)). % DPA statement about access_unauthorized_personal_data(processor)

% DPA Segment 168: physical access to the Facilities by employees and...
dpa_states(dpa168, obligatory, retain_personal_data_access_logs(processor)). % DPA statement about retain_personal_data_access_logs(processor)

% DPA Segment 170: processor will conduct periodic reviews of the sec...
dpa_states(dpa170, obligatory, conduct_inspection_auditor_site_processor_network(processor)). % DPA statement about conduct_inspection_auditor_site_processor_network(processor)

% DPA Segment 171: processor will continually evaluate the security o...
dpa_states(dpa171, obligatory, take_into_account_risk_risk(processor)). % DPA statement about take_into_account_risk_risk(processor)

% DPA Segment 174: For the purposes of Article 26(2) of Directive 95/...
dpa_states(dpa174, forbidden, process_personal_data_personal_data(processor, personal_data)). % DPA statement about process_personal_data_personal_data(processor, personal_data)

% DPA Segment 175: The entity identified as “controller” in the DPA (...
dpa_states(dpa175, obligatory, comply(processor, controller)). % DPA statement about comply(processor, controller)

% DPA Segment 178: 410 Terry Avenue North, Seattle, WA 98109-5210, US...
dpa_states(dpa178, obligatory, make_available_information_to_controller_address(processor)). % DPA statement about make_available_information_to_controller_address(processor)

% DPA Segment 180: each a “party”; together “the parties”,
dpa_states(dpa180, obligatory, comply(processor)). % DPA statement about comply(processor)

% DPA Segment 181: HAVE AGREED on the following Contractual Clauses (...
dpa_states(dpa181, obligatory, comply(processor, controller)). % DPA statement about comply(processor, controller)

% DPA Segment 184: For the purposes of the Clauses:
dpa_states(dpa184, obligatory, comply(processor)). % DPA statement about comply(processor)

% DPA Segment 185: 'personal data', 'special categories of data', 'pr...
dpa_states(dpa185, obligatory, comply(processor, controller)). % DPA statement about comply(processor, controller)

% DPA Segment 186: the controller' means the controller who transfers...
dpa_states(dpa186, obligatory, comply(processor, controller)). % DPA statement about comply(processor, controller)

% DPA Segment 187: the processor' means the processor who agrees to r...
dpa_states(dpa187, obligatory, comply(processor, controller)). % DPA statement about comply(processor, controller)

% DPA Segment 188: the subprocessor' means any processor engaged by t...
dpa_states(dpa188, permitted, engage_sub_processor_sub_processor(processor, sub_processor)). % DPA statement about engage_sub_processor_sub_processor(processor, sub_processor)

% DPA Segment 189: the applicable data protection law' means the legi...
dpa_states(dpa189, obligatory, comply(processor, controller)). % DPA statement about comply(processor, controller)

% DPA Segment 190: 'technical and organisational security measures' m...
dpa_states(dpa190, obligatory, take_into_account_risk_risk(processor, personal_data)). % DPA statement about take_into_account_risk_risk(processor, personal_data)

% DPA Segment 193: The details of the transfer and in particular the ...
dpa_states(dpa193, obligatory, make_available_information_to_controller_appendix_1(processor, personal_data)). % DPA statement about make_available_information_to_controller_appendix_1(processor, personal_data)

% DPA Segment 196: The data subject can enforce against the controlle...
dpa_states(dpa196, permitted, communicate_data_subject_data_subject(processor, controller)). % DPA statement about communicate_data_subject_data_subject(processor, controller)

% DPA Segment 197: The data subject can enforce against the processor...
dpa_states(dpa197, permitted, communicate_data_subject_data_subject(processor, controller)). % DPA statement about communicate_data_subject_data_subject(processor, controller)

% DPA Segment 198: the rights and obligations of the controller, in w...
dpa_states(dpa198, obligatory, inform_data_subject(processor, controller)). % DPA statement about inform_data_subject(processor, controller)

% DPA Segment 199: The data subject can enforce against the subproces...
dpa_states(dpa199, permitted, communicate_data_subject_data_subject_sub_processor(processor, sub_processor)). % DPA statement about communicate_data_subject_data_subject_sub_processor(processor, sub_processor)

% DPA Segment 266: Such third-party liability of the subprocessor sha...
dpa_states(dpa266, forbidden, remain_fully_liable_subprocessor_other(processor, sub_processor)). % DPA statement about remain_fully_liable_subprocessor_other(processor, sub_processor)

% DPA Segment 201: The parties do not object to a data subject being ...
dpa_states(dpa201, permitted, communicate_data_subject_data_subject(processor, personal_data)). % DPA statement about communicate_data_subject_data_subject(processor, personal_data)

% DPA Segment 205: that the processing, including the transfer itself...
dpa_states(dpa205, obligatory, comply(processor, controller)). % DPA statement about comply(processor, controller)

% DPA Segment 206: that it has instructed and throughout the duration...
dpa_states(dpa206, obligatory, inform_applicable_data_protection_law(processor, controller)). % DPA statement about inform_applicable_data_protection_law(processor, controller)

% DPA Segment 207: that the processor will provide sufficient guarant...
dpa_states(dpa207, obligatory, comply(processor)). % DPA statement about comply(processor)

% DPA Segment 208: that after assessment of the requirements of the a...
dpa_states(dpa208, obligatory, take_into_account_risk_data_to_be_protected(processor, personal_data)). % DPA statement about take_into_account_risk_data_to_be_protected(processor, personal_data)

% DPA Segment 209: that it will ensure compliance with the security m...
dpa_states(dpa209, obligatory, ensure_security_of_processing(processor)). % DPA statement about ensure_security_of_processing(processor)

% DPA Segment 210: that, if the transfer involves special categories ...
dpa_states(dpa210, obligatory, inform_data_subject(processor, personal_data)). % DPA statement about inform_data_subject(processor, personal_data)

% DPA Segment 211: to forward any notification received from the proc...
dpa_states(dpa211, obligatory, inform_legal_requirement_supervisory_authority(processor, sub_processor)). % DPA statement about inform_legal_requirement_supervisory_authority(processor, sub_processor)

% DPA Segment 212: to make available to the data subjects upon reques...
dpa_states(dpa212, obligatory, make_available_information_to_controller_data_subject_copy_of_clauses(processor, personal_data)). % DPA statement about make_available_information_to_controller_data_subject_copy_of_clauses(processor, personal_data)

% DPA Segment 213: Clauses or the contract contain commercial informa...
dpa_states(dpa213, permitted, remove_commercial_information_commercial_information(processor)). % DPA statement about remove_commercial_information_commercial_information(processor)

% DPA Segment 214: that, in the event of subprocessing, the processin...
dpa_states(dpa214, obligatory, process_personal_data_personal_data(processor, sub_processor)). % DPA statement about process_personal_data_personal_data(processor, sub_processor)

% DPA Segment 215: that it will ensure compliance with Clause 4(a) to...
dpa_states(dpa215, obligatory, ensure_compliancecontroller(processor)). % DPA statement about ensure_compliancecontroller(processor)

% DPA Segment 219: to process the personal data only on behalf of the...
dpa_states(dpa219, obligatory, process_personal_data_personal_data(processor, controller)). % DPA statement about process_personal_data_personal_data(processor, controller)

% DPA Segment 220: that it has no reason to believe that the legislat...
dpa_states(dpa220, obligatory, comply(processor, controller)). % DPA statement about comply(processor, controller)

% DPA Segment 221: that it has implemented the technical and organisa...
dpa_states(dpa221, obligatory, processor_take_measures_required_byarticle_32(processor, personal_data)). % DPA statement about processor_take_measures_required_byarticle_32(processor, personal_data)

% DPA Segment 222: that it will promptly notify the controller about:
dpa_states(dpa222, obligatory, notify(processor, controller)). % DPA statement about notify(processor, controller)

% DPA Segment 223: any legally binding request for disclosure of the ...
dpa_states(dpa223, permitted, disclose_personal_data_law_enforcement_authority(processor, personal_data)). % DPA statement about disclose_personal_data_law_enforcement_authority(processor, personal_data)

% DPA Segment 224: any accidental or unauthorised access, and
dpa_states(dpa224, obligatory, remain_fully_liable(processor)). % DPA statement about remain_fully_liable(processor)

% DPA Segment 225: any request received directly from the data subjec...
dpa_states(dpa225, forbidden, fail_assist_supervisory_authority(processor, personal_data)). % DPA statement about fail_assist_supervisory_authority(processor, personal_data)

% DPA Segment 226: to deal promptly and properly with all inquiries f...
dpa_states(dpa226, obligatory, fulfill_respond_inquiries(processor, controller)). % DPA statement about fulfill_respond_inquiries(processor, controller)

% DPA Segment 227: 1 Mandatory requirements of the national legislati...
dpa_states(dpa227, obligatory, comply(processor, personal_data)). % DPA statement about comply(processor, personal_data)

% DPA Segment 228: Some examples of such mandatory requirements which...
dpa_states(dpa228, obligatory, disclose_personal_data_third_parties(processor)). % DPA statement about disclose_personal_data_third_parties(processor)

% DPA Segment 229: at the request of the controller to submit its dat...
dpa_states(dpa229, obligatory, conduct_inspection_auditor_site(processor, controller)). % DPA statement about conduct_inspection_auditor_site(processor, controller)

% DPA Segment 230: to make available to the data subject upon request...
dpa_states(dpa230, obligatory, make_available_information_to_controller_data_subject_copy_of_clauses(processor, controller)). % DPA statement about make_available_information_to_controller_data_subject_copy_of_clauses(processor, controller)

% DPA Segment 231: that, in the event of subprocessing, it has previo...
dpa_states(dpa231, obligatory, inform(processor, controller)). % DPA statement about inform(processor, controller)

% DPA Segment 232: that the processing services by the subprocessor w...
dpa_states(dpa232, obligatory, impose_obligations_subprocessors_subprocessor(processor, sub_processor)). % DPA statement about impose_obligations_subprocessors_subprocessor(processor, sub_processor)

% DPA Segment 233: to send promptly a copy of any subprocessor agreem...
dpa_states(dpa233, obligatory, make_available_information_to_controller_subprocessor_agreement(processor, sub_processor)). % DPA statement about make_available_information_to_controller_subprocessor_agreement(processor, sub_processor)

% DPA Segment 236: The parties agree that any data subject, who has s...
dpa_states(dpa236, obligatory, make_available_information_to_controller_compensation(processor, sub_processor)). % DPA statement about make_available_information_to_controller_compensation(processor, sub_processor)

% DPA Segment 237: If a data subject is not able to bring a claim for...
dpa_states(dpa237, obligatory, comply(processor, sub_processor)). % DPA statement about comply(processor, sub_processor)

% DPA Segment 238: The processor may not rely on a breach by a subpro...
dpa_states(dpa238, forbidden, fail_assist(processor, sub_processor)). % DPA statement about fail_assist(processor, sub_processor)

% DPA Segment 239: If a data subject is not able to bring a claim aga...
dpa_states(dpa239, obligatory, fail_assist_supervisory_authority(processor, sub_processor)). % DPA statement about fail_assist_supervisory_authority(processor, sub_processor)

% DPA Segment 240: The liability of the subprocessor shall be limited...
dpa_states(dpa240, forbidden, remain_fully_liable_subprocessor_other(processor, sub_processor)). % DPA statement about remain_fully_liable_subprocessor_other(processor, sub_processor)

% DPA Segment 243: The processor agrees that if the data subject invo...
dpa_states(dpa243, obligatory, accept_decision_data_subject(processor, personal_data)). % DPA statement about accept_decision_data_subject(processor, personal_data)

% DPA Segment 244: to refer the dispute to mediation, by an independe...
dpa_states(dpa244, obligatory, refer_dispute_to_mediation_independent_person_dispute(processor)). % DPA statement about refer_dispute_to_mediation_independent_person_dispute(processor)

% DPA Segment 245: to refer the dispute to the courts in the Member S...
dpa_states(dpa245, obligatory, refer_dispute_courts(processor, controller)). % DPA statement about refer_dispute_courts(processor, controller)

% DPA Segment 246: The parties agree that the choice made by the data...
dpa_states(dpa246, forbidden, alter_substantive_procedural_rights(processor, personal_data)). % DPA statement about alter_substantive_procedural_rights(processor, personal_data)

% DPA Segment 249: The controller agrees to deposit a copy of this co...
dpa_states(dpa249, obligatory, make_available_information_to_controller_supervisory_authority(processor, controller)). % DPA statement about make_available_information_to_controller_supervisory_authority(processor, controller)

% DPA Segment 250: The parties agree that the supervisory authority h...
dpa_states(dpa250, obligatory, conduct_inspection_auditor_site_supervisory_authority(processor, sub_processor)). % DPA statement about conduct_inspection_auditor_site_supervisory_authority(processor, sub_processor)

% DPA Segment 251: The processor shall promptly inform the controller...
dpa_states(dpa251, obligatory, inform_legislation(processor, sub_processor)). % DPA statement about inform_legislation(processor, sub_processor)

% DPA Segment 252: In such a case the controller shall be entitled to...
dpa_states(dpa252, obligatory, take_measures_clause_5_b(processor, controller)). % DPA statement about take_measures_clause_5_b(processor, controller)

% DPA Segment 255: The Clauses shall be governed by the law of the Me...
dpa_states(dpa255, obligatory, comply(processor, controller)). % DPA statement about comply(processor, controller)

% DPA Segment 258: The parties undertake not to vary or modify the Cl...
dpa_states(dpa258, forbidden, alter_personal_data(processor)). % DPA statement about alter_personal_data(processor)

% DPA Segment 259: This does not preclude the parties from adding cla...
dpa_states(dpa259, permitted, add_clauses_parties_business_related_issues(processor)). % DPA statement about add_clauses_parties_business_related_issues(processor)

% DPA Segment 262: The processor shall not subcontract any of its pro...
dpa_states(dpa262, forbidden, engage_sub_processor_sub_processor(processor, controller)). % DPA statement about engage_sub_processor_sub_processor(processor, controller)

% DPA Segment 263: Where the processor subcontracts its obligations u...
dpa_states(dpa263, obligatory, impose_obligations_subprocessors_subprocessor(processor, sub_processor)). % DPA statement about impose_obligations_subprocessors_subprocessor(processor, sub_processor)

% DPA Segment 264: Where the subprocessor fails to fulfil its data pr...
dpa_states(dpa264, obligatory, remain_fully_liable(processor, sub_processor)). % DPA statement about remain_fully_liable(processor, sub_processor)

% DPA Segment 265: The prior written contract between the processor a...
dpa_states(dpa265, obligatory, documented_instruction(processor, sub_processor)). % DPA statement about documented_instruction(processor, sub_processor)

% DPA Segment 267: The provisions relating to data protection aspects...
dpa_states(dpa267, obligatory, comply(processor, controller)). % DPA statement about comply(processor, controller)

% DPA Segment 268: The controller shall keep a list of subprocessing ...
dpa_states(dpa268, obligatory, retain_personal_data_subprocessing_agreements(processor, controller)). % DPA statement about retain_personal_data_subprocessing_agreements(processor, controller)

% DPA Segment 269: The list shall be available to the controller's da...
dpa_states(dpa269, obligatory, make_available_information_to_controller_data_protection_supervisory_authority(processor, controller)). % DPA statement about make_available_information_to_controller_data_protection_supervisory_authority(processor, controller)

% DPA Segment 271: Obligation after the termination of personal data ...
dpa_states(dpa271, obligatory, retain_personal_data_personal_data(processor, personal_data)). % DPA statement about retain_personal_data_personal_data(processor, personal_data)

% DPA Segment 272: The parties agree that on the termination of the p...
dpa_states(dpa272, obligatory, retain_personal_data_personal_data(processor, sub_processor)). % DPA statement about retain_personal_data_personal_data(processor, sub_processor)

% DPA Segment 273: In that case, the processor warrants that it will ...
dpa_states(dpa273, obligatory, ensure_confidentiality_others(processor, personal_data)). % DPA statement about ensure_confidentiality_others(processor, personal_data)

% DPA Segment 274: The processor and the subprocessor warrant that up...
dpa_states(dpa274, obligatory, conduct_inspection_auditor_site_supervisory_authority(processor, sub_processor)). % DPA statement about conduct_inspection_auditor_site_supervisory_authority(processor, sub_processor)

% DPA Segment 275: APPENDIX 1 TO THE STANDARD CONTRACTUAL CLAUSES
dpa_states(dpa275, obligatory, comply(processor)). % DPA statement about comply(processor)

% DPA Segment 277: The controller is the entity identified as “contro...
dpa_states(dpa277, obligatory, comply(processor, controller)). % DPA statement about comply(processor, controller)

% DPA Segment 279: The processor is Amazon Web Services, Inc., a prov...
dpa_states(dpa279, obligatory, comply(processor)). % DPA statement about comply(processor)

% DPA Segment 281: Data subjects are defined in Section 1.3 of the DP...
dpa_states(dpa281, obligatory, make_available_information_to_controller_dpa(processor, personal_data)). % DPA statement about make_available_information_to_controller_dpa(processor, personal_data)

% DPA Segment 283: The personal data is defined in Section 1.3 of the...
dpa_states(dpa283, obligatory, make_available_information_to_controller_dpa(processor, personal_data)). % DPA statement about make_available_information_to_controller_dpa(processor, personal_data)

% DPA Segment 285: The personal data transferred will be subject to t...
dpa_states(dpa285, obligatory, make_available_information_to_controller(processor, personal_data)). % DPA statement about make_available_information_to_controller(processor, personal_data)

% DPA Segment 286: APPENDIX 2 TO THE STANDARD CONTRACTUAL CLAUSES
dpa_states(dpa286, obligatory, comply(processor)). % DPA statement about comply(processor)

% DPA Segment 287: This Appendix forms part of the Clauses and must b...
dpa_states(dpa287, obligatory, make_available_information_to_controller_clauses(processor)). % DPA statement about make_available_information_to_controller_clauses(processor)

% DPA Segment 288: Description of the technical and organisational se...
dpa_states(dpa288, obligatory, comply(processor)). % DPA statement about comply(processor)

% DPA Segment 289: The technical and organisational security measures...
dpa_states(dpa289, obligatory, comply(processor)). % DPA statement about comply(processor)

% --- Facts ---
% Default facts for testing
processor(processor).
sub_processor(sub_processor).
controller(controller).
personal_data(personal_data).
data_subject(data_subject).

% --- Matching and Status Logic ---

% Helper: Does the DPA mention a specific action with any modality?
dpa_mentions_action(Action) :- dpa_states(_, _, Action).

% Status Rule 1: Satisfaction
% A requirement R is satisfied if the DPA states the same modality M for the same action A.
satisfies(R) :-
    requirement(R),
    req_modality(R, M),
    req_action(R, A),
    dpa_states(_, M, A). % Check if DPA states the exact same modality and action

% Status Rule 2: Violation
% A requirement R is violated if the DPA states a conflicting modality for the same action A.
violates(R) :-
    requirement(R),
    req_modality(R, obligatory), % Requirement is Obligatory
    req_action(R, A),
    dpa_states(_, forbidden, A).   % DPA Forbids it

violates(R) :-
    requirement(R),
    req_modality(R, permitted),  % Requirement is Permitted
    req_action(R, A),
    dpa_states(_, forbidden, A).   % DPA Forbids it

% Status Rule 3: Non-Mention
% A requirement R is not mentioned if it's neither satisfied nor violated by the DPA text.
not_mentioned(R) :-
    requirement(R),
    not satisfies(R),
    not violates(R).

% --- Show Directives ---
% Display the status for each Requirement ID.
#show satisfies/1.
#show violates/1.
#show not_mentioned/1.
