import pandas as pd
import numpy as np
import logging
import re 
from typing import Optional

logging.basicConfig(
    level = logging.INFO,                                 
    format = '%(asctime)s - %(levelname)s - %(message)s'  
)

class DataProcessorColorectal:
    GROUP_STAGE_MAPPING = {
        # Stage 0/Occult
        '0': '0',
        
        # Stage I
        'I': 'I',
        
        # Stage II
        'II': 'II',
        'IIA': 'II',
        'IIB': 'II',
        'IIC': 'II',
        
        # Stage III
        'III': 'III',
        'IIIA': 'III',
        'IIIB': 'III',
        'IIIC': 'III',
        
        # Stage IV
        'IV': 'IV',
        'IVA': 'IV',
        'IVB': 'IV',
        'IVC': 'IV',
        
        # Unknown
        'Unknown': 'unknown'
    }

    def __init__(self):
        self.enhanced_df = None

    def process_enhanced(self,
                         file_path: str,
                         patient_ids: list = None,
                         drop_stage: bool = True, 
                         drop_dates: bool = True) -> pd.DataFrame: 
        """
        Processes Enhanced_MetastaticCRC.csv to standardize categories, consolidate staging information, and calculate time-based metrics between key clinical events.

        Parameters
        ----------
        file_path : str
            Path to Enhanced_AdvancedNSCLC.csv file
        patient_ids : list, optional
            List of specific PatientIDs to process. If None, processes all patients
        drop_stage : bool, default=True
            If True, drops original GroupStage after consolidating into major groups
        drop_dates : bool, default=True
            If True, drops date columns (DiagnosisDate and MetDiagnosisDate) after calculating durations

        Returns
        -------
        pd.DataFrame
            - PatientID : object
                unique patient identifier
            - GroupStage_mod : categorical
                consolidated overall staging (0-IV, Unknown)
            - days_diagnosis_to_met : float
                days from diagnosis to metastatic disease 
            - adv_diagnosis_year : categorical
                year of metastatic diagnosis 
            
            Original staging and date columns retained if respective drop_* = False

        Notes
        -----
        Duplicate PatientIDs are logged as warnings if found but retained in output
        Processed DataFrame is stored in self.enhanced_df
        """
        try:
            df = pd.read_csv(file_path)
            logging.info(f"Successfully read Enhanced_MetastaticCRC.csv file with shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")

            # Filter for specific PatientIDs if provided
            if patient_ids is not None:
                logging.info(f"Filtering for {len(patient_ids)} specific PatientIDs")
                df = df[df['PatientID'].isin(patient_ids)]
                logging.info(f"Successfully filtered Enhanced_MetastaticCRC.csv file with shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")
        
            # Convert categorical columns
            categorical_cols = ['GroupStage', 'CrcSite']
            df[categorical_cols] = df[categorical_cols].astype('category')

            # Recode stage variable using class-level mapping and create new column
            df['GroupStage_mod'] = df['GroupStage'].map(self.GROUP_STAGE_MAPPING).astype('category')

            # Drop original stage variable if specified
            if drop_stage:
                df = df.drop(columns=['GroupStage'])

            # Convert date columns
            date_cols = ['DiagnosisDate', 'MetDiagnosisDate']
            for col in date_cols:
                df[col] = pd.to_datetime(df[col])

            # Generate new variables 
            df['days_diagnosis_to_met'] = (df['MetDiagnosisDate'] - df['DiagnosisDate']).dt.days
            df['met_diagnosis_year'] = pd.Categorical(df['MetDiagnosisDate'].dt.year)

            if drop_dates:
                df = df.drop(columns = ['MetDiagnosisDate', 'DiagnosisDate'])

            # Check for duplicate PatientIDs
            if len(df) > df['PatientID'].nunique():
                duplicate_ids = df[df.duplicated(subset = ['PatientID'], keep = False)]['PatientID'].unique()
                logging.warning(f"Duplicate PatientIDs found: {duplicate_ids}")

            logging.info(f"Successfully processed Enhanced_MetastaticCRC.csv file with final shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")
            return df

        except Exception as e:
            logging.error(f"Error processing Enhanced_MetastaticCRC.csv file: {e}")
            return None