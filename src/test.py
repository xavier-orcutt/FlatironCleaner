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
                      days_after: int = 0,
                      missing_date_strategy: str = 'conservative') -> pd.DataFrame:
    """
    Processes insurance data to identify insurance coverage relative to a specified index date.
    Insurance types are grouped into four categories: Medicare, Medicaid, Commercial, and Other Insurance. 
    
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
    missing_date_strategy : str
        Strategy for handling missing StartDate:
        - 'conservative': Excludes records with both StartDate and EndDate missing and imputes EndDate for missing StartDate (may underestimate coverage)
        - 'liberal': Assumes records with missing StartDates are always active and imputes default date of 2000-01-01 (may overestimate coverage)
    
    Returns
    -------
    pd.DataFrame
        - PatientID : object
            unique patient identifier
        - medicare : Int64
            binary indicator (0/1) for Medicare coverage
        - medicaid : Int64
            binary indicator (0/1) for Medicaid coverage
        - commercial : Int64
            binary indicator (0/1) for commercial insuarnce coverage
        - other_insurance : Int64
            binaroy indicator (0/1) for other insurance types (eg., other payer, other government program, patient assistance program, self pay, and workers compensation)

    Notes
    -----
    Insurance is considered active if:
    1. StartDate falls before or during the specified time window AND
    2. Either:
        - EndDate is missing (considered still active) OR
        - EndDate falls on or after the start of the time window 

    Insurance categorization logic:
    1. Medicaid takes priority over Medicare for dual-eligible patients
    2. Records are classified as Medicare if:
       - PayerCategory is 'Medicare' OR
       - PayerCategory is not 'Medicaid' AND IsMedicareAdv is 'Yes' AND IsManagedMedicaid is not 'Yes' AND IsMedicareMedicaid is not 'Yes' OR
       - PayerCategory is not 'Medicaid' AND IsMedicareSupp is 'Yes' AND IsManagedMedicaid is not 'Yes' AND IsMedicareMedicaid is not 'Yes'
    3. Records are classified as Medicaid if:
       - PayerCategory is 'Medicaid' OR
       - IsManagedMedicaid is 'Yes' OR
       - IsMedicareMedicaid is 'Yes'
    4. Records are classified as Commercial if PayerCategory is 'Commercial Health Plan' after above reclassification
    5. All other records are classified as Other Insurance

    EndDate is missing for most patients
    All PatientIDs from index_date_df are included in the output
    Duplicate PatientIDs are logged as warnings but retained in output
    Results are stored in self.insurance_df attribute
    """
    # Input validation
    if not isinstance(index_date_df, pd.DataFrame):
        raise ValueError("index_date_df must be a pandas DataFrame")
    if 'PatientID' not in index_date_df.columns:
        raise ValueError("index_date_df must contain a 'PatientID' column")
    if not index_date_column or index_date_column not in index_date_df.columns:
        raise ValueError(f"Column '{index_date_column}' not found in index_date_df")
    
    if days_before is not None:
        if not isinstance(days_before, int) or days_before < 0:
            raise ValueError("days_before must be a non-negative integer or None")
    if not isinstance(days_after, int) or days_after < 0:
        raise ValueError("days_after must be a non-negative integer")
    
    if not isinstance(missing_date_strategy, str):
        raise ValueError("missing_date_strategy must be a string")    
    valid_strategies = ['conservative', 'liberal']
    if missing_date_strategy not in valid_strategies:
        raise ValueError("missing_date_strategy must be 'conservative' or 'liberal'")

    try:
        df = pd.read_csv(file_path)
        logging.info(f"Successfully read Insurance.csv file with shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")

        df['StartDate'] = pd.to_datetime(df['StartDate'])
        df['EndDate'] = pd.to_datetime(df['EndDate'])

        both_dates_missing = df['StartDate'].isna() & df['EndDate'].isna()
        start_date_missing = df['StartDate'].isna()

        if missing_date_strategy == 'conservative':
            # Exclude records with both dates missing, and impute EndDate for missing StartDate
            df = df[~both_dates_missing]
            df['StartDate'] = np.where(df['StartDate'].isna(), df['EndDate'], df['StartDate'])
        elif missing_date_strategy == 'liberal':
            # Assume always active by setting StartDate to default date of 2000-01-01
            df.loc[start_date_missing, 'StartDate'] = pd.Timestamp('2000-01-01')

        # Filter for StartDate after 1900-01-01
        df = df[df['StartDate'] > pd.Timestamp('1900-01-01')]

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

        # Reclassify Commerical Health Plans that should be Medicare or Medicaid
        # Identify Medicare Advantage plans
        df['PayerCategory'] = np.where((df['PayerCategory'] != 'Medicaid') & (df['IsMedicareAdv'] == 'Yes') & (df['IsManagedMedicaid'] != 'Yes') & (df['IsMedicareMedicaid'] != 'Yes'),
                                       'Medicare',
                                       df['PayerCategory'])

        # Identify Medicare Supplement plans incorrectly labeled as Commercial
        df['PayerCategory'] = np.where((df['PayerCategory'] != 'Medicaid') & (df['IsMedicareSupp'] == 'Yes') & (df['IsManagedMedicaid'] != 'Yes') & (df['IsMedicareMedicaid'] != 'Yes'),
                                       'Medicare',
                                       df['PayerCategory'])
        
        # Identify Managed Medicaid plans incorrectly labeled as Commercial
        df['PayerCategory'] = np.where((df['IsManagedMedicaid'] == 'Yes'),
                                       'Medicaid',
                                       df['PayerCategory'])
        
        # Identify Medicare-Medicaid dual eligible plans incorrectly labeled as Commercial
        df['PayerCategory'] = np.where((df['IsMedicareMedicaid'] == 'Yes'),
                                       'Medicaid',
                                       df['PayerCategory'])

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

        INSURANCE_MAPPING = {
            'Commercial Health Plan': 'commercial',
            'Medicare': 'medicare',
            'Medicaid': 'medicaid',
            'Other Payer - Type Unknown': 'other_insurance',
            'Other Government Program': 'other_insurance',
            'Patient Assistance Program': 'other_insurance',
            'Self Pay': 'other_insurance',
            'Workers Compensation': 'other_insurance'
        }

        df_filtered['PayerCategory'] = df_filtered['PayerCategory'].replace(INSURANCE_MAPPING)

        final_df = (
            df_filtered
            .drop_duplicates(subset = ['PatientID', 'PayerCategory'], keep = 'first')
            .assign(value=1)
            .pivot(index = 'PatientID', columns = 'PayerCategory', values = 'value')
            .fillna(0) 
            .astype('Int64')  
            .rename_axis(columns = None)
            .reset_index()
        )

        # Merger index_date_df to ensure all PatientIDs are included
        final_df = pd.merge(index_date_df[['PatientID']], final_df, on = 'PatientID', how = 'left')
        
        insurance_columns = list(set(INSURANCE_MAPPING.values()))
        for col in insurance_columns:
            final_df[col] = final_df[col].fillna(0).astype('Int64')

        # Check for duplicate PatientIDs
        if len(final_df) > final_df['PatientID'].nunique():
            duplicate_ids = final_df[final_df.duplicated(subset = ['PatientID'], keep = False)]['PatientID'].unique()
            logging.warning(f"Duplicate PatientIDs found: {duplicate_ids}")

        logging.info(f"Successfully processed Insurance.csv file with final shape: {final_df.shape} and unique PatientIDs: {(final_df['PatientID'].nunique())}")
        return final_df

    except Exception as e:
        logging.error(f"Error processing Insurance.csv file: {e}")
        return None
        
# TESTING
df = pd.read_csv('data_nsclc/Enhanced_AdvancedNSCLC.csv')
a = process_insurance(file_path="data_nsclc/Insurance.csv",
                       index_date_df= df,
                       index_date_column= 'AdvancedDiagnosisDate',
                       days_before=None,
                       days_after=14,
                       missing_date_strategy = 'conservative')

embed()