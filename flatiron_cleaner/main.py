import pandas as pd
from breast import DataProcessorBreast
from merge_utils import merge_dataframes
from IPython import embed     

if __name__ == "__main__":
    processor = DataProcessorBreast()

    #enhanced_file_path = "data_crc/Enhanced_MetastaticCRC.csv"
    #demographics_file_path = "data_crc/Demographics.csv"
    #practice_file_path = "data_crc/Practice.csv"
    #mortality_file_path = "data_crc/Enhanced_Mortality_V2.csv"
    #biomarkers_file_path = "data_crc/Enhanced_MetCRCBiomarkers.csv"
    #ecog_file_path = "data_crc/ECOG.csv"
    #vitals_file_path = "data_crc/Vitals.csv"
    #lab_file_path =  "data_crc/Lab.csv"
    #medication_file_path =  "data_crc/MedicationAdministration.csv"
    #diagnosis_file_path = "data_crc/Diagnosis.csv"
    #insurance_file_path = "data_crc/Insurance.csv"

    #df = pd.read_csv(enhanced_file_path)
    
    # Process datasets
    #enhanced_df = processor.process_enhanced(enhanced_file_path)

    #demographics_df = processor.process_demographics(demographics_file_path, 
    #                                                 index_date_df=df, 
    #                                                 index_date_column='MetDiagnosisDate')
    
    #practice_df = processor.process_practice(practice_file_path)

    #mortality_df = processor.process_mortality(mortality_file_path,
    #                                          index_date_df=df,
    #                                           index_date_column='MetDiagnosisDate',
    #                                           visit_path="data_crc/Visit.csv",
    #                                           oral_path="data_crc/Enhanced_MetCRC_Orals.csv",
    #                                           drop_dates = True)
    
    #biomarkers_df = processor.process_biomarkers(biomarkers_file_path,
    #                                             index_date_df=df,
    #                                             index_date_column='MetDiagnosisDate',
    #                                             days_before=None,
    #                                             days_after=14)

    #ecog_df =  processor.process_ecog(ecog_file_path,
    #                                  index_date_df=df,
    #                                  index_date_column='MetDiagnosisDate',
    #                                  days_before=90,
    #                                  days_after=0,
    #                                  days_before_further=360)
    
    #vitals_df = processor.process_vitals(vitals_file_path,
    #                                     index_date_df=df,
    #                                     index_date_column='MetDiagnosisDate',
    #                                     weight_days_before = 90,
    #                                     days_after = 0,
    #                                     vital_summary_lookback = 180,
    #                                     abnormal_reading_threshold = 1)
    
    #labs_df = processor.process_labs(lab_file_path,
    #                                 index_date_df=df,
    #                                 index_date_column='MetDiagnosisDate',
    #                                 days_before = 90,
    #                                 days_after = 0,
    #                                 summary_lookback = 180)
    
    #medications_df = processor.process_medications(medication_file_path,
    #                                              index_date_df = df,
    #                                              index_date_column='MetDiagnosisDate',
    #                                              days_before=180,
    #                                              days_after=0)
    

    #diagnosis_df = processor.process_diagnosis(diagnosis_file_path,
    #                                           index_date_df = df,
    #                                           index_date_column='MetDiagnosisDate',
    #                                           days_before=None,
    #                                           days_after=0)
    
    #insurance_df = processor.process_insurance(insurance_file_path,
    #                                           index_date_df = df,
    #                                           index_date_column='MetDiagnosisDate',
    #                                           days_before=None,
    #                                           days_after=0,
    #                                           missing_date_strategy = 'liberal')
    


    # Merge datasets
    #merged_data = merge_dataframes()
    
    #if merged_data is not None:
    #    print(merged_data.head())
    #    print(merged_data.dtypes)
    #    embed()