import pandas as pd
import logging

logging.basicConfig(
    level = logging.INFO,                                 
    format = '%(asctime)s - %(levelname)s - %(message)s'  
)

def merge_dataframes(*dataframes: pd.DataFrame) -> pd.DataFrame:
    """
    Outer merge of multiple datasets based on PatientID
    
    Parameters:
    *dataframes: Variable number of pandas DataFrames to merge
    
    Returns:
    pd.DataFrame: Merged dataset
    """
    # Check if any dataframes are provided
    if not dataframes:
        logging.error("No dataframes provided for merging")
        
    try:
        for i, df in enumerate(dataframes):
            if 'PatientID' not in df.columns:
                raise KeyError(f"Dataset {i+1} missing PatientID column")
            logging.info(f"Dataset {i+1} shape: {df.shape}, unique PatientIDs: {df.PatientID.nunique()}")
        
        merged_df = dataframes[0]
        for i, df in enumerate(dataframes[1:], 2):
            merged_df = pd.merge(merged_df, df, on='PatientID', how='outer')
            logging.info(f"After merge {i-1} shape: {merged_df.shape}, unique PatientIDs {merged_df.PatientID.nunique()}")
        
        return merged_df
    
    except Exception as e:
        logging.error(f"Error merging datasets: {e}")
        return None