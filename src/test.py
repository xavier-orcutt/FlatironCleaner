import pandas as pd
import numpy as np
import logging
from IPython import embed
from typing import Optional
import re 

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

def process_practice(file_path: str,
                     patient_ids: list = None) -> pd.DataFrame:
    """
    Processes Practice.csv to consolidate practice types per patient into a single categorical value indicating academic, community, or both settings.

    Parameters
    ----------
    file_path : str
        Path to Practice.csv file
    patient_ids : list, optional
        List of specific PatientIDs to process. If None, processes all patients

    Returns
    -------
    pd.DataFrame
        - PatientID : object
            unique patient identifier  
        - PracticeType_mod : category
            practice setting (ACADEMIC, COMMUNITY, or BOTH)
    
    Notes
    -----
    PracticeID and PrimaryPhysicianID are removed 
    Duplicate PatientIDs are logged as warnings if found
    Processed DataFrame is stored in self.practice_df
    """
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Successfully read Practice.csv file with shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")

        # Filter for specific PatientIDs if provided
        if patient_ids is not None:
            logging.info(f"Filtering for {len(patient_ids)} specific PatientIDs")
            df = df[df['PatientID'].isin(patient_ids)]
            logging.info(f"Successfully filtered Practice.csv file with shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")

        df = df[['PatientID', 'PracticeType']]

        # Group by PatientID and get set of unique PracticeTypes
        grouped = df.groupby('PatientID')['PracticeType'].unique()
        grouped_df = pd.DataFrame(grouped).reset_index()

        # Function to determine the practice type
        def get_practice_type(practice_types):
            if len(practice_types) > 1:
                return 'BOTH'
            return practice_types[0]
        
        # Apply the function to the column containing sets
        grouped_df['PracticeType_mod'] = grouped_df['PracticeType'].apply(get_practice_type).astype('category')

        final_df = grouped_df[['PatientID', 'PracticeType_mod']]

        # Check for duplicate PatientIDs
        if len(final_df) > final_df['PatientID'].nunique():
            logging.error(f"Duplicate PatientIDs found")
            return None
        
        logging.info(f"Successfully processed Practice.csv file with final shape: {final_df.shape} and unique PatientIDs: {(final_df['PatientID'].nunique())}")
        return final_df

    except Exception as e:
        logging.error(f"Error processing Practice.csv file: {e}")
        return None
    
        
# TESTING
df = pd.read_csv('data_nsclc/Enhanced_AdvancedNSCLC.csv')
ids = df.sample(n=1000).PatientID.to_list() 
a = process_practice(file_path="data_nsclc/Practice.csv",
                     patient_ids = ids)

embed()