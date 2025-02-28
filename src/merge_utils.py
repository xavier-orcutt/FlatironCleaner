import pandas as pd
import logging

logging.basicConfig(
    level = logging.INFO,                                 
    format = '%(asctime)s - %(levelname)s - %(message)s'  
)

def merge_dataframes(*dataframes: pd.DataFrame,
                     merge_type: str = 'outer') -> pd.DataFrame:
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

    # Check for None values in the provided dataframes
    if any(df is None for df in dataframes):
        logging.error("One or more input dataframes are None. Cannot proceed with merging.")
        return None
        
    try:
        # Calculate anticipated merges and final column count
        num_merges = len(dataframes) - 1
        total_columns = sum(len(df.columns) for df in dataframes) - (num_merges * 1)  # Subtract shared PatientID columns
        logging.info(f"Anticipated number of merges: {num_merges}")
        logging.info(f"Anticipated number of columns in final dataframe presuming all columns are unique except for PatientID: {total_columns}")

        for i, df in enumerate(dataframes):
            if 'PatientID' not in df.columns:
                raise KeyError(f"Dataframe {i+1} missing PatientID column")
            logging.info(f"Dataset {i+1} shape: {df.shape}, unique PatientIDs: {df.PatientID.nunique()}")
        
        merged_df = dataframes[0]
        for i, df in enumerate(dataframes[1:], 2):
            merged_df = pd.merge(merged_df, df, on = 'PatientID', how = merge_type)
            logging.info(f"After merge {i-1} shape: {merged_df.shape}, unique PatientIDs {merged_df.PatientID.nunique()}")
        
        return merged_df
    
    except Exception as e:
        logging.error(f"Error merging datasets: {e}")
        return None