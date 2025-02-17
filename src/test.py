import pandas as pd
import numpy as np
import logging
from IPython import embed
from typing import Optional

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

def process_ecog(file_path: str,
                 index_date_df: pd.DataFrame,
                 index_date_column: str, 
                 days_before: Optional[int] = 30,
                 days_after: int = 0) -> pd.DataFrame:
    """
    Processes ECOG.csv by determining the ECOG for patients closest to time of index. In the case of two ECOG scores 
    on the same day or equidistant but on opposite sides of the index date, the higher ECOG score (worse performance) 
    will be selected.

    Parameters
    ----------
    file_path : str
        Path to ECOG.csv file
    index_date_df : pd.DataFrame
        DataFrame containing PatientID and index dates. Only ECOGs for PatientIDs present in this DataFrame will be processed
    index_date_column : str
        Column name in index_date_df containing the index date
    days_before : int, optional
        Number of days before the index date to include. Must be >= 0 or None. If None, includes all prior results. Default: 30
    days_after : int, optional
        Number of days after the index date to include. Must be >= 0. Default: 0
    
    Returns
    -------
    pd.DataFrame
        Processed DataFrame containing:
        - PatientID : unique patient identifier
        - ecog_index : ECOG score closest to index date
        - ecog_increased_2 : binary metric seeing if ECOG score increased in the preceding 6 months by 2 points 

    Notes
    ------
    Duplicate PatientIDs are logged as warnings if found
    Processed DataFrame is stored in self.ecog_df
    """

    # Input validation
    if not isinstance(index_date_df, pd.DataFrame) or 'PatientID' not in index_date_df.columns:
        raise ValueError("index_date_df must be a DataFrame containing 'PatientID' column")
    if not index_date_column or index_date_column not in index_date_df.columns:
        raise ValueError(f"Column '{index_date_column}' not found in index_date_df")
    
    if days_before is not None:
        if not isinstance(days_before, int) or days_before < 0:
            raise ValueError("days_before must be a non-negative integer or None")
    if not isinstance(days_after, int) or days_after < 0:
        raise ValueError("days_after must be a non-negative integer")

    try:
        df = pd.read_csv(file_path)
        logging.info(f"Successfully read ECOG.csv file with shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")

        df['EcogDate'] = pd.to_datetime(df['EcogDate'])

        index_date_df[index_date_column] = pd.to_datetime(index_date_df[index_date_column])

        # Select PatientIDs that are included in the index_date_df the merge on 'left'
        df = df[df.PatientID.isin(index_date_df.PatientID)]
        df = pd.merge(
             df,
             index_date_df[['PatientID', index_date_column]],
             on = 'PatientID',
             how = 'left'
             )
        logging.info(f"Successfully merged ECOG.csv df with index_date_df resulting in shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")
                    
        # Create new variable 'index_to_ecog' that notes difference in days between ECOG date and index date
        df['index_to_ecog'] = (df['EcogDate'] - df[index_date_column]).dt.days
        
        # Select ECOG that fall within desired before and after index date
        if days_before is None:
            # Only filter for days after
            df_filtered = df[df['index_to_ecog'] <= days_after].copy()
            window_desc = f"negative infinity to +{days_after} days from index date"
        else:
            # Filter for both before and after
            df_filtered = df[
                (df['index_to_ecog'] <= days_after) & 
                (df['index_to_ecog'] >= -days_before)
            ].copy()
            window_desc = f"-{days_before} to +{days_after} days from index date"

        logging.info(f"After applying window period of {window_desc}, "f"remaining records: {df_filtered.shape}, "f"unique PatientIDs: {df_filtered['PatientID'].nunique()}")


        # Check for duplicate PatientIDs
        if len(final_df) > final_df['PatientID'].nunique():
            logging.error(f"Duplicate PatientIDs found")
            return None

        logging.info(f"Successfully processed Enhanced_AdvUrothelialBiomarkers.csv file with final shape: {final_df.shape} and unique PatientIDs: {(final_df['PatientID'].nunique())}")
        return final_df

    except Exception as e:
        logging.error(f"Error processing Enhanced_AdvUrothelialBiomarkers.csv file: {e}")
        return None
        
# TESTING 
index_date_df = pd.read_csv("data/Enhanced_AdvUrothelial.csv")
a = process_biomarkers(file_path="data/ECOG.csv",
                      index_date_df=index_date_df.sample(500),
                      index_date_column='AdvancedDiagnosisDate',
                      days_before=90
                     )

embed()