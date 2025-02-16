import pandas as pd
from urothelial_processor import DataProcessorUrothelial
from merge_utils import merge_dataframes
from IPython import embed     

if __name__ == "__main__":
    processor = DataProcessorUrothelial()
    
    df = pd.read_csv('data/Enhanced_AdvUrothelial.csv')
    id_list = df.sample(500).PatientID.to_list()  

    enhanced_file_path = "data/Enhanced_AdvUrothelial.csv"
    demographics_file_path = "data/Demographics.csv"
    practice_file_path = "data/Practice.csv"
    mortality_file_path = "data/Enhanced_Mortality_V2.csv"
    biomarkers_file_path = "data/Enhanced_AdvUrothelialBiomarkers.csv"
    
    # Process datasets
    enhanced_df = processor.process_enhanced_adv(enhanced_file_path)

    demographics_df = processor.process_demographics(demographics_file_path, 
                                                     index_date_df=df, 
                                                     index_date_column='AdvancedDiagnosisDate')
    
    practice_df = processor.process_practice(practice_file_path)

    mortality_df = processor.process_mortality(mortality_file_path,
                                               index_date_df=df,
                                               index_date_column='AdvancedDiagnosisDate',
                                               df_merge_type='left',
                                               visit_path="data/Visit.csv",
                                               oral_path="data/Enhanced_AdvUrothelial_Orals.csv",
                                               drop_dates = False)
    
    biomarkers_df = processor.process_biomarkers(biomarkers_file_path,
                                                 index_date_df=df,
                                                 index_date_column='AdvancedDiagnosisDate',
                                                 days_before=90,
                                                 days_after=14)


    # Merge datasets
    merged_data = merge_dataframes(enhanced_df, demographics_df, practice_df, mortality_df, biomarkers_df)
    
    if merged_data is not None:
        print(merged_data.head())
        print(merged_data.dtypes)
        embed()