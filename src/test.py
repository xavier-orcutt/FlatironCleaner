import pandas as pd
import numpy as np
import logging
from IPython import embed
from typing import Optional
import re 

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

def process_insurance(file_path: str,
                      index_date_df: pd.DataFrame,
                      index_date_column: str,
                      days_before: Optional[int] = None,
                      days_after: int = 0) -> pd.DataFrame:
    """
    Processes insurance data to identify insurance coverage relative to a specified index date.
    Insurance types are grouped into four categories: Medicare, Medicaid, Commercial, and Other. 
    
    Parameters
    ----------
    file_path : str
        Path to Insurance.csv file
    index_date_df : pd.DataFrame
        DataFrame containing PatientID and index dates. Only insurances for PatientIDs present in this DataFrame will be processed
    index_date_column : str
        Column name in index_date_df containing the index date
    days_before : int | None, optional
        Number of days before the index date to include for window period. Must be >= 0 or None. If None, includes all prior results. Default: None
    days_after : int, optional
        Number of days after the index date to include for window period. Must be >= 0. Default: 0
    
    Returns
    -------
    pd.DataFrame
        Processed DataFrame containing:
        - PatientID : unique patient identifier
        - medicare : binary indicator (0/1) for Medicare coverage
        - medicaid : binary indicator (0/1) for Medicaid coverage
        - commercial : binary indicator (0/1) for commercial insuarnce coverage
        - other : binaroy indicator (0/1) for other insurance types (eg., other payer, other government program, patient assistance program, 
          self pay, and workers compensation)

    Notes
    -----
    Insurance is considered active if:
    1. StartDate falls before or during the specified time window AND
    2. Either:
        - EndDate is missing (considered still active) OR
        - EndDate falls on or after the start of the time window 
    EndDate is missing for most patients
    Missing StartDate values are conservatively imputed with EndDate values
    Duplicate PatientIDs are logged as warnings but retained in output
    Results are stored in self.insurance_df attribute
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
        logging.info(f"Successfully read Insurance.csv file with shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")

        df['StartDate'] = pd.to_datetime(df['StartDate'])
        df['EndDate'] = pd.to_datetime(df['EndDate'])

        # Impute missing StartDate with EndDate
        df['StartDate'] = np.where(df['StartDate'].isna(), df['EndDate'], df['StartDate'])

        index_date_df[index_date_column] = pd.to_datetime(index_date_df[index_date_column])

        # Select PatientIDs that are included in the index_date_df the merge on 'left'
        df = df[df.PatientID.isin(index_date_df.PatientID)]
        df = pd.merge(
             df,
             index_date_df[['PatientID', index_date_column]],
             on = 'PatientID',
             how = 'left'
             )
        logging.info(f"Successfully merged Insurance.csv df with index_date_df resulting in shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")

        # Calculate days relative to index date for start 
        df['days_to_start'] = (df['StartDate'] - df[index_date_column]).dt.days

        # Define window boundaries
        window_start = -days_before if days_before is not None else float('-inf')
        window_end = days_after

        # Insurance is active if it:
        # 1. Starts before or during the window AND
        # 2. Either has no end date OR ends after window starts
        df_filtered = df[
            (df['days_to_start'] <= window_end) &  # Starts before window ends
            (
                df['EndDate'].isna() |  # Either has no end date (presumed to be still active)
                ((df['EndDate'] - df[index_date_column]).dt.days >= window_start)  # Or ends after window starts
            )
        ].copy()

        df_filtered['PayerCategory'] = df_filtered['PayerCategory'].replace(INSURANCE_MAPPING)

        final_df = (
            df_filtered
            .drop_duplicates(subset = ['PatientID', 'PayerCategory'], keep = 'first')
            .assign(value=1)
            .pivot(index = 'PatientID', columns = 'PayerCategory', values = 'value')
            .fillna(0) 
            .astype(int)  
            .rename_axis(columns = None)
            .reset_index()
        )

        # Check for duplicate PatientIDs
        if len(final_df) > final_df['PatientID'].nunique():
            logging.error(f"Duplicate PatientIDs found")
            return None

        logging.info(f"Successfully processed Insurance.csv file with final shape: {final_df.shape} and unique PatientIDs: {(final_df['PatientID'].nunique())}")
        return final_df

    except Exception as e:
        logging.error(f"Error processing Insurance.csv file: {e}")
        return None
        
# TESTING 
index_date_df = pd.read_csv("data/Enhanced_AdvUrothelial.csv")
a = process_insurance(file_path="data/Insurance.csv",
                      index_date_df=index_date_df,
                      index_date_column='AdvancedDiagnosisDate',
                      days_before = None,
                      days_after = 0)

embed()