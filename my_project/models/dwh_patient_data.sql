-- INSERT INTO patient_data (
--     person_id, member_id, subscriber_id, gender, race, birth_date, death_date, death_flag, 
--     enrollment_start_date, enrollment_end_date, payer, payer_type, plan, original_reason_entitlement_code, 
--     dual_status_code, medicare_status_code, GROUP_ID, GROUP_NAME, first_name, last_name, 
--     social_security_number, subscriber_relation, address, city, state, zip_code, phone, 
--     data_source, file_name, FILE_DATE, ingest_datetime
-- )
SELECT

'P001' AS person_id, 'M001' AS member_id, 'S001' AS subscriber_id, 'M' AS gender, 'White' AS race, '1980-05-15' AS birth_date, NULL AS death_date, FALSE AS death_flag, 
 '2021-01-01' AS enrollment_start_date, '2023-12-31' AS enrollment_end_date, 'Aetna' AS payer, 'Commercial' AS payer_type, 'Silver Plan' AS plan, 'CodeX' AS original_reason_entitlement_code, 
 'Dual' AS dual_status_code, 'Initial Enrollment' AS medicare_status_code, 'G001' AS GROUP_ID, 'Group A' AS GROUP_NAME, 'John' AS first_name, 'Doe' AS last_name, 
 '123-45-6789' AS social_security_number, 'Self' AS subscriber_relation, '1234 Elm St' AS address, 'Chicago' AS city, 'IL' AS state, '60601' AS zip_code, '312-555-1234' AS phone, 
 'CSV Upload' AS data_source, 'insurance_data_2021.csv' AS file_name, '2021-01-01' AS FILE_DATE, '2021-01-01 09:00:00' AS ingest_datetime)
UNION
SELECT
    'P002' AS person_id, 'M002' AS member_id, 'S002' AS subscriber_id, 'F' AS gender, 'Black' AS race, '1975-08-22' AS birth_date, NULL AS death_date, FALSE AS death_flag, 
    '2022-02-01' AS enrollment_start_date, '2024-01-31' AS enrollment_end_date, 'BlueCross' AS payer, 'Medicare' AS payer_type, 'Gold Plan' AS plan, 'CodeY' AS original_reason_entitlement_code, 
    'Not Dual' AS dual_status_code, 'Qualifying Disability' AS medicare_status_code, 'G002' AS GROUP_ID, 'Group B' AS GROUP_NAME, 'Jane' AS first_name, 'Smith' AS last_name, 
    '987-65-4321' AS social_security_number, 'Spouse' AS subscriber_relation, '5678 Oak St' AS address, 'New York' AS city, 'NY' AS state, '10001' AS zip_code, '212-555-5678' AS phone, 
    'Manual Entry' AS data_source, 'patient_data_2022.csv' AS file_name, '2022-02-01' AS FILE_DATE, '2022-02-01 14:15:00' AS ingest_datetime
UNION
SELECT
    'P003' AS person_id, 'M003' AS member_id, 'S003' AS subscriber_id, 'M' AS gender, 'Asian' AS race, '1992-11-30' AS birth_date, '2023-04-01' AS death_date, TRUE AS death_flag, 
    '2020-06-01' AS enrollment_start_date, '2023-05-01' AS enrollment_end_date, 'Cigna' AS payer, 'Commercial' AS payer_type, 'Platinum Plan' AS plan, 'CodeZ' AS original_reason_entitlement_code, 
    'Dual' AS dual_status_code, 'End-Stage Renal Disease' AS medicare_status_code, 'G003' AS GROUP_ID, 'Group C' AS GROUP_NAME, 'Chris' AS first_name, 'Lee' AS last_name, 
    '234-56-7890' AS social_security_number, 'Self' AS subscriber_relation, '4321 Pine St' AS address, 'San Francisco' AS city, 'CA' AS state, '94102' AS zip_code, '415-555-6789' AS phone, 
    'Automated Feed' AS data_source, 'medicare_data_2020.csv' AS file_name, '2020-06-01' AS FILE_DATE, '2020-06-01 10:30:00' AS ingest_datetime
UNION
SELECT
    'P004' AS person_id, 'M004' AS member_id, 'S004' AS subscriber_id, 'F' AS gender, 'Hispanic' AS race, '1999-02-10' AS birth_date, NULL AS death_date, FALSE AS death_flag, 
    '2023-03-15' AS enrollment_start_date, '2025-03-14' AS enrollment_end_date, 'UnitedHealthcare' AS payer, 'Medicaid' AS payer_type, 'Bronze Plan' AS plan, 'CodeW' AS original_reason_entitlement_code, 
    'Not Dual' AS dual_status_code, 'Medicare Due to Age' AS medicare_status_code, 'G004' AS GROUP_ID, 'Group D' AS GROUP_NAME, 'Maria' AS first_name, 'Gonzalez' AS last_name, 
    '345-67-8901' AS social_security_number, 'Dependent' AS subscriber_relation, '8765 Maple St' AS address, 'Miami' AS city, 'FL' AS state, '33101' AS zip_code, '305-555-7890' AS phone, 
    'Manual Entry' AS data_source, 'health_data_2023.csv' AS file_name, '2023-03-15' AS FILE_DATE, '2023-03-15 11:45:00' AS ingest_datetime
UNION
SELECT
    'P005' AS person_id, 'M005' AS member_id, 'S005' AS subscriber_id, 'M' AS gender, 'White' AS race, '1985-07-22' AS birth_date, NULL AS death_date, FALSE AS death_flag, 
    '2022-05-01' AS enrollment_start_date, '2023-05-01' AS enrollment_end_date, 'Humana' AS payer, 'Medicare' AS payer_type, 'Bronze Plan' AS plan, 'CodeX' AS original_reason_entitlement_code, 
    'Dual' AS dual_status_code, 'Initial Enrollment' AS medicare_status_code, 'G005' AS GROUP_ID, 'Group E' AS GROUP_NAME, 'Mike' AS first_name, 'Johnson' AS last_name, 
    '123-12-1234' AS social_security_number, 'Self' AS subscriber_relation, '2345 Birch St' AS address, 'Austin' AS city, 'TX' AS state, '73301' AS zip_code, '512-555-2345' AS phone, 
    'CSV Upload' AS data_source, 'insurance_data_2022.csv' AS file_name, '2022-05-01' AS FILE_DATE, '2022-05-01 16:00:00' AS ingest_datetime
UNION
SELECT
    'P006' AS person_id, 'M006' AS member_id, 'S006' AS subscriber_id, 'F' AS gender, 'Black' AS race, '1988-03-12' AS birth_date, '2023-02-01' AS death_date, TRUE AS death_flag, 
    '2019-09-01' AS enrollment_start_date, '2023-02-01' AS enrollment_end_date, 'Aetna' AS payer, 'Commercial' AS payer_type, 'Silver Plan' AS plan, 'CodeY' AS original_reason_entitlement_code, 
    'Not Dual' AS dual_status_code, 'Disability' AS medicare_status_code, 'G006' AS GROUP_ID, 'Group F' AS GROUP_NAME, 'Lisa' AS first_name, 'Brown' AS last_name, 
    '987-12-3456' AS social_security_number, 'Self' AS subscriber_relation, '6789 Cedar St' AS address, 'Los Angeles' AS city, 'CA' AS state, '90001' AS zip_code, '323-555-6789' AS phone, 
    'Automated Feed' AS data_source, 'medicare_data_2019.csv' AS file_name, '2019-09-01' AS FILE_DATE, '2019-09-01 09:45:00' AS ingest_datetime
UNION
SELECT
    'P007' AS person_id, 'M007' AS member_id, 'S007' AS subscriber_id, 'M' AS gender, 'Asian' AS race, '1972-11-20' AS birth_date, '2022-08-01' AS death_date, FALSE AS death_flag, 
    '2018-04-01' AS enrollment_start_date, '2022-04-01' AS enrollment_end_date, 'BlueCross' AS payer, 'Medicaid' AS payer_type, 'Gold Plan' AS plan, 'CodeZ' AS original_reason_entitlement_code, 
    'Dual' AS dual_status_code, 'End-Stage Renal Disease' AS medicare_status_code, 'G007' AS GROUP_ID, 'Group G' AS GROUP_NAME, 'David' AS first_name, 'Nguyen' AS last_name, 
    '234-56-8901' AS social_security_number, 'Self' AS subscriber_relation, '3456 Maple Ave' AS address, 'Boston' AS city, 'MA' AS state, '02108' AS zip_code, '617-555-9876' AS phone, 
    'Manual Entry' AS data_source, 'health_data_2018.csv' AS file_name, '2018-04-01' AS FILE_DATE, '2018-04-01 14:00:00' AS ingest_datetime;
