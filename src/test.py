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
    Processes ECOG.csv to determine patient ECOG scores relative to a reference index date.
    For each patient, finds:
    1. The ECOG score closest to index date (selecting higher score in case of ties)
    2. Whether ECOG newly increased to ≥2 from 0-1 in the prior 6 months

    Parameters
    ----------
    file_path : str
        Path to ECOG.csv file
    index_date_df : pd.DataFrame
        DataFrame containing PatientID and index dates. Only ECOGs for PatientIDs present in this DataFrame will be processed
    index_date_column : str
        Column name in index_date_df containing the index date
    days_before : int | None, optional
        Number of days before the index date to include. Must be >= 0 or None. If None, includes all prior results. Default: 30
    days_after : int, optional
        Number of days after the index date to include. Must be >= 0. Default: 0
    
    Returns
    -------
    pd.DataFrame
        Processed DataFrame containing:
        - PatientID : unique patient identifier
        - ecog_index : ECOG score (0-5) closest to index date, categorical
        - ecog_newly_gte2 : boolean indicating if ECOG increased from 0-1 to ≥2 in 6 months before index

    Notes
    ------
    When multiple ECOG scores are equidistant to index date, selects higher score
    Uses fixed 6-month lookback for newly_gte2 calculation regardless of days_before
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
        df['EcogValue'] = pd.to_numeric(df['EcogValue'], errors = 'coerce').astype('Int64')

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
                (df['index_to_ecog'] >= -days_before)].copy()
            window_desc = f"-{days_before} to +{days_after} days from index date"

        logging.info(f"After applying window period of {window_desc}, "f"remaining records: {df_filtered.shape}, "f"unique PatientIDs: {df_filtered['PatientID'].nunique()}")

        # Find EcogValue closest to index date within specified window periods
        ecog_index_df = (
            df_filtered
            .assign(abs_days_to_index = lambda x: abs(x['index_to_ecog']))
            .sort_values(
                by=['PatientID', 'abs_days_to_index', 'EcogValue'], 
                ascending=[True, True, False])
            .groupby('PatientID')
            .first()
            .reset_index()
            [['PatientID', 'EcogValue']]
            .rename(columns = {'EcogValue': 'ecog_index'})
            .assign(
                ecog_index = lambda x: x['ecog_index'].astype(pd.CategoricalDtype(categories = [0, 1, 2, 3, 4, 5], ordered = True))
                )
            )
        
        # Find EcogValue newly greater than or equal to 2 by time of index date with 6 month look back using pre-specified days_after
        # First get 6-month window data 
        df_6month = df[
                (df['index_to_ecog'] <= days_after) & 
                (df['index_to_ecog'] >= -180)].copy()
        
        # Create flag for ECOG newly greater than or equal to 2
        ecog_newly_gte2_df = (
            df_6month
            .sort_values(['PatientID', 'EcogDate']) 
            .groupby('PatientID')
            .agg({
                'EcogValue': lambda x: (
                    # 1. Last ECOG is ≥2
                    (x.iloc[-1] >= 2) and 
                    # 2. Any previous ECOG was 0 or 1
                    any(x.iloc[:-1].isin([0, 1]))
                )
            })
            .reset_index()
            .rename(columns={'EcogValue': 'ecog_newly_gte2'})
        )

        # Merge back with df_filtered
        unique_patient_df = df[['PatientID']].drop_duplicates()
        final_df = pd.merge(unique_patient_df, ecog_index_df, on = 'PatientID', how = 'left')
        final_df = pd.merge(final_df, ecog_newly_gte2_df, on = 'PatientID', how = 'left')
        final_df['ecog_newly_gte2'] = final_df['ecog_newly_gte2'].astype('boolean')

        logging.info(f"Successfully processed ECOG.csv file with final shape: {final_df.shape} and unique PatientIDs: {(final_df['PatientID'].nunique())}")
        return final_df

    except Exception as e:
        logging.error(f"Error processing ECOG.csv file: {e}")
        return None
        
# TESTING 
index_date_df = pd.read_csv("data/Enhanced_AdvUrothelial.csv")
a = process_ecog(file_path="data/ECOG.csv",
                      index_date_df=index_date_df.sample(1500),
                      index_date_column='AdvancedDiagnosisDate',
                      days_before=90
                     )

embed()