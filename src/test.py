import pandas as pd
import numpy as np
import logging
from IPython import embed
from typing import Optional

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

PDL1_PERCENT_STAINING_MAPPING = { 
    np.nan: 0,
    '0%': 1, 
    '< 1%': 2,
    '1%': 3, 
    '2% - 4%': 4,
    '5% - 9%': 5,
    '10% - 19%': 6,  
    '20% - 29%': 7, 
    '30% - 39%': 8, 
    '40% - 49%': 9, 
    '50% - 59%': 10, 
    '60% - 69%': 11, 
    '70% - 79%': 12, 
    '80% - 89%': 13, 
    '90% - 99%': 14,
    '100%': 15
}

def process_biomarkers(
                      file_path: str,
                      index_date_df: pd.DataFrame,
                      index_date_column: str, 
                      days_before: Optional[int] = None,
                      days_after: int = 0) -> pd.DataFrame:
    """
    Processes Enhanced_AdvUrothelialBiomarkers.csv by determining FGFR and PDL1 status for each patient within a specified time window relative to a reference index date
    For each biomarker:
    - FGFR status is classified as:
        - 'positive' if any test result is positive (ever-positive)
        - 'negative' if any test is negative without positives (only-negative) 
        - 'unknown' if all results are indeterminate
    - PDL1 status follows the same classification logic
    - PDL1 staining percentage is also captured

    Parameters
    ----------
    file_path : str
        Path to Enhanced_AdvUrothelialBiomarkers.csv file
    index_date_df : pd.DataFrame
        DataFrame containing PatientID and reference dates for defining analysis windows
    index_date_column : str
        Column name in index_date_df containing the index date
    days_before : int, optional
        Number of days before the index date to include. Enter number >= 0. If None, includes all prior results (default: None)
    days_after : int, optional
        Number of days after the index date to include. Enter number >= 0 (default: 0)
    
    Returns
    -------
    pd.DataFrame
        Processed DataFrame containing:
        - PatientID : unique patient identifier
        - fgfr_status : positive if ever-positive, negative if only-negative, otherwise unknown
        - pdl1_status : positive if ever-positive, negative if only-negative, otherwise unknown
        - pdl1_staining : returns a patient's maximum percent staining for PDL1  

    Notes
    ------
    Missing ResultDate is imputed with SpecimenReceivedDate.
    Duplicate PatientIDs are logged as warnings if found
    Processed DataFrame is stored in self.biomarkers_df
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
        logging.info(f"Successfully read Enhanced_AdvUrothelialBiomarkers.csv file with shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")

        df['ResultDate'] = pd.to_datetime(df['ResultDate'])
        df['SpecimenReceivedDate'] = pd.to_datetime(df['SpecimenReceivedDate'])

        # Impute missing ResultDate with SpecimenReceivedDate
        df['ResultDate'] = np.where(df['ResultDate'].isna(), df['SpecimenReceivedDate'], df['ResultDate'])

        # Process index dates and merge
        index_date_df[index_date_column] = pd.to_datetime(index_date_df[index_date_column])
        df = pd.merge(
            df,
            index_date_df[['PatientID', index_date_column]],
            on = 'PatientID',
            how = 'left'
        )
        logging.info(f"Successfully merged Enhanced_AdvUrothelialBiomarkers.csv df with index_date_df resulting in shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")
        
        # Create new variable 'index_to_result' that notes difference in days between resulted specimen and index date
        df['index_to_result'] = (df['ResultDate'] - df[index_date_column]).dt.days
        
        # Select biomarkers that fall within desired before and after index date
        if days_before is None:
            # Only filter for days after
            df_filtered = df[df['index_to_result'] <= days_after].copy()
            window_desc = f"negative infinity up to index date"
        else:
            # Filter for both before and after
            df_filtered = df[
                (df['index_to_result'] <= days_after) & 
                (df['index_to_result'] >= -days_before)
            ].copy()
            window_desc = f"-{days_before} to +{days_after} days from index"

        logging.info(f"After applying window period {window_desc}, "f"remaining records: {df_filtered.shape}, "f"unique PatientIDs: {df_filtered['PatientID'].nunique()}")

        # Process FGFR status
        fgfr_df = (
            df_filtered
            .query('BiomarkerName == "FGFR"')
            .groupby('PatientID')['BiomarkerStatus']
            .agg(lambda x: 'positive' if any ('Positive' in val for val in x)
                 else ('negative' if any('Negative' in val for val in x)
                       else 'unknown'))
            .reset_index()
            .rename(columns={'BiomarkerStatus': 'fgfr_status'})
            )
        
        # Process PDL1 status
        pdl1_df = (
            df_filtered
            .query('BiomarkerName == "PDL1"')
            .groupby('PatientID')['BiomarkerStatus']
            .agg(lambda x: 'positive' if any ('PD-L1 positive' in val for val in x)
                 else ('negative' if any('PD-L1 negative/not detected' in val for val in x)
                       else 'unknown'))
            .reset_index()
            .rename(columns={'BiomarkerStatus': 'pdl1_status'})
            )

        # Process PDL1 staining 
        pdl1_staining_df = (
            df_filtered
            .query('BiomarkerName == "PDL1"')
            .query('BiomarkerStatus == "PD-L1 positive"')
            .groupby('PatientID')['PercentStaining']
            .apply(lambda x: x.map(PDL1_PERCENT_STAINING_MAPPING))
            .groupby('PatientID')
            .agg('max')
            .to_frame(name = 'pdl1_ordinal_value')
            .reset_index()
            )
        
        # Create reverse mapping to convert back to percentage strings
        reverse_pdl1_dict = {v: k for k, v in PDL1_PERCENT_STAINING_MAPPING.items()}
        pdl1_staining_df['PercentStaining'] = pdl1_staining_df['pdl1_ordinal_value'].map(reverse_pdl1_dict)
        pdl1_staining_df = pdl1_staining_df.drop(columns = ['pdl1_ordinal_value'])

        # Merge dataframes
        final_df = pd.merge(pdl1_df, pdl1_staining_df, on = 'PatientID', how = 'left')
        final_df = pd.merge(final_df, fgfr_df, on = 'PatientID', how = 'outer')

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
a = process_biomarkers(file_path="data/Enhanced_AdvUrothelialBiomarkers.csv",
                      index_date_df=index_date_df,
                      index_date_column='AdvancedDiagnosisDate',
                      days_before=90,
                      days_after=14
                     )

embed()