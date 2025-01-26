import pandas as pd
import logging

logging.basicConfig(
    level = logging.INFO,                                 
    format = '%(asctime)s - %(levelname)s - %(message)s'  
)

def merge_dataframes(self, *dataframes: pd.DataFrame) -> pd.DataFrame:
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
        return None
        
    try:
        # Log pre-merge information for each dataset
        for i, df in enumerate(dataframes):
            logging.info(f"Pre-merge unique PatientIDs in dataset {i+1}: {df.PatientID.nunique()}")
        
        # Start with first dataframe and merge others iteratively
        merged_df = dataframes[0]
        
        for i, df in enumerate(dataframes[1:], 2):
            merged_df = pd.merge(
                merged_df,
                df,
                on = 'PatientID',
                how = 'outer'
            )
            logging.info(f"After merging dataset {i}, shape: {merged_df.shape}, unique PatientIDs: {merged_df.PatientID.nunique()}")
            
        # Log final merge information
        logging.info(f"Final merged dataset shape: {merged_df.shape} with {merged_df.PatientID.nunique()} unique PatientIDs")
        
        return merged_df
    
    except Exception as e:
        logging.error(f"Error merging datasets: {e}")
        return None