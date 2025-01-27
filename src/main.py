import pandas as pd
from urothelial_processor import DataProcessorUrothelial
from merge_utils import merge_dataframes    

if __name__ == "__main__":
    processor = DataProcessorUrothelial()
    
    df = pd.read_csv('data/Enhanced_AdvUrothelial.csv')  

    enhanced_file_path = "data/Enhanced_AdvUrothelial.csv"
    demographics_file_path = "data/Demographics.csv"
    
    # Process both datasets
    enhanced_df = processor.process_enhanced_adv(enhanced_file_path)
    demographics_df = processor.process_demographics(demographics_file_path, df, 'AdvancedDiagnosisDate')

    # Merge datasets
    merged_data = merge_dataframes(enhanced_df, demographics_df)
    
    if merged_data is not None:
        print(merged_data.head())
        print(merged_data.dtypes)