from urothelial_processor import DataProcessorUrothelial
from merge_utils import merge_dataframes    

if __name__ == "__main__":
    processor = DataProcessorUrothelial()
    
    enhanced_file_path = "data/Enhanced_AdvUrothelial.csv"
    demographics_file_path = "data/Demographics.csv"
    
    # Process both datasets
    enhanced_df = processor.process_enhanced_adv(enhanced_file_path)
    demographics_df = processor.process_demographics(demographics_file_path)
    
    # Merge datasets
    merged_data = processor.merge_dataframes(enhanced_df, demographics_df)
    
    if merged_data is not None:
        print("\nFirst few rows of merged data:")
        print(merged_data.head())