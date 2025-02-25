import pandas as pd
import numpy as np
import logging
from IPython import embed
from typing import Optional
import re 

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

def process_ecog(file_path: str,
                 index_date_df: pd.DataFrame,
                 index_date_column: str, 
                 days_before: int = 90,
                 days_after: int = 0, 
                 days_before_further: int = 180) -> pd.DataFrame:
    """
    Processes ECOG.csv to determine patient ECOG scores and progression patterns relative 
    to a reference index date. Uses two different time windows for distinct clinical purposes:
    
    1. A smaller window near the index date to find the most clinically relevant ECOG score
        that represents the patient's status at that time point
    2. A larger lookback window to detect clinically significant ECOG progression,
        specifically looking for patients whose condition worsened from ECOG 0-1 to ≥2
    
    This dual-window approach allows for both accurate point-in-time assessment and
    detection of deteriorating performance status over a clinically meaningful period.

    For each patient, finds:
    1. The ECOG score closest to index date (selecting higher score in case of ties)
    2. Whether ECOG newly increased to ≥2 from 0-1 in the lookback period

    Parameters
    ----------
    file_path : str
        Path to ECOG.csv file
    index_date_df : pd.DataFrame
        DataFrame containing PatientID and index dates. Only ECOGs for PatientIDs present in this DataFrame will be processed
    index_date_column : str
        Column name in index_date_df containing the index date
    days_before : int, optional
        Number of days before the index date to include. Must be >= 0. Default: 90
    days_after : int, optional
        Number of days after the index date to include. Must be >= 0. Default: 0
    days_before_further : int, optional
        Number of days before index date to look for ECOG progression (0-1 to ≥2). Must be >= 0. Consdier
        selecting a larger integer than days_before to capture meaningful clinical deterioration over time.
        Default: 180
        
    Returns
    -------
    pd.DataFrame
        - PatientID : object
            unique patient identifier
        - ecog_index : category, ordered 
            ECOG score (0-5) closest to index date
        - ecog_newly_gte2 : Int64
            binary indicator (0/1) for ECOG increased from 0-1 to ≥2 in 6 months before index

    Notes
    ------
    When multiple ECOG scores are equidistant to index date, the higher score is selected
    All PatientIDs from index_date_df are included in the output and values will be NaN for patients without ECOG values
    Duplicate PatientIDs are logged as warnings if found
    Processed DataFrame is stored in self.ecog_df
    """
    # Input validation
    if not isinstance(index_date_df, pd.DataFrame):
        raise ValueError("index_date_df must be a pandas DataFrame")
    if 'PatientID' not in index_date_df.columns:
        raise ValueError("index_date_df must contain a 'PatientID' column")
    if not index_date_column or index_date_column not in index_date_df.columns:
        raise ValueError(f"Column '{index_date_column}' not found in index_date_df")
    
    if not isinstance(days_before, int) or days_before < 0:
        raise ValueError("days_before must be a non-negative integer")
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
        df_closest_window = df[
            (df['index_to_ecog'] <= days_after) & 
            (df['index_to_ecog'] >= -days_before)].copy()

        # Find EcogValue closest to index date within specified window periods
        ecog_index_df = (
            df_closest_window
            .assign(abs_days_to_index = lambda x: abs(x['index_to_ecog']))
            .sort_values(
                by=['PatientID', 'abs_days_to_index', 'EcogValue'], 
                ascending=[True, True, False]) # Last False means highest ECOG is selected in ties 
            .groupby('PatientID')
            .first()
            .reset_index()
            [['PatientID', 'EcogValue']]
            .rename(columns = {'EcogValue': 'ecog_index'})
            .assign(
                ecog_index = lambda x: x['ecog_index'].astype(pd.CategoricalDtype(categories = [0, 1, 2, 3, 4, 5], ordered = True))
                )
        )
        
        # Filter dataframe using farther back window
        df_progression_window = df[
                (df['index_to_ecog'] <= days_after) & 
                (df['index_to_ecog'] >= -days_before_further)].copy()
        
        # Create flag for ECOG newly greater than or equal to 2
        ecog_newly_gte2_df = (
            df_progression_window
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

        # Merge dataframes - start with index_date_df to ensure all PatientIDs are included
        final_df = index_date_df[['PatientID']].copy()
        final_df = pd.merge(final_df, ecog_index_df, on = 'PatientID', how = 'left')
        final_df = pd.merge(final_df, ecog_newly_gte2_df, on = 'PatientID', how = 'left')
        
        # Assign datatypes 
        final_df['ecog_index'] = final_df['ecog_index'].astype(pd.CategoricalDtype(categories=[0, 1, 2, 3, 4, 5], ordered=True))
        final_df['ecog_newly_gte2'] = final_df['ecog_newly_gte2'].astype('Int64')

        # Check for duplicate PatientIDs
        if len(final_df) > final_df['PatientID'].nunique():
            logging.error(f"Duplicate PatientIDs found")
            return None
            
        logging.info(f"Successfully processed ECOG.csv file with final shape: {final_df.shape} and unique PatientIDs: {(final_df['PatientID'].nunique())}")
        return final_df

    except Exception as e:
        logging.error(f"Error processing ECOG.csv file: {e}")
        return None
        
# TESTING
df = pd.read_csv('data_nsclc/Enhanced_AdvancedNSCLC.csv')
a = process_ecog(file_path="data_nsclc/ECOG.csv",
                       index_date_df= df,
                       index_date_column= 'AdvancedDiagnosisDate',
                       days_before=90,
                       days_after=14,
                       days_before_further = 360)

embed()