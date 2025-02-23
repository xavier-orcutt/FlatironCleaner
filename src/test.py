import pandas as pd
import numpy as np
import logging
from IPython import embed
from typing import Optional
import re 

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

def process_enhanced_adv(file_path: str,
                         patient_ids: list = None,
                         drop_stage: bool = True, 
                         drop_dates: bool = True) -> pd.DataFrame: 
        """
        Processes Enhanced_AdvancedNSCLC.csv to standardize categories, consolidate staging information, and calculate time-based metrics between key clinical events.

        Parameters
        ----------
        file_path : str
            Path to Enhanced_AdvancedNSCLC.csv file
        patient_ids : list, optional
            List of specific PatientIDs to process. If None, processes all patients
        drop_stage : bool, default=True
            If True, drops original GroupStage after consolidating into major groups
        drop_dates : bool, default=True
            If True, drops date columns (DiagnosisDate and AdvancedDiagnosisDate) after calculating durations

        Returns
        -------
        pd.DataFrame
            Processed DataFrame containing:
            - PatientID : object
                unique patient identifier
            - Histology : categorical
                histology type 
            - SmokingStatus : categorical
                smoking history
            - GroupStage_mod : categorical
                consolidated overall staging (0-IV, Unknown)
            - days_diagnosis_to_adv : float
                days from diagnosis to advanced disease 
            - adv_diagnosis_year : categorical
                year of advanced diagnosis 
            
            Original staging and date columns retained if respective drop_* = False

        Notes
        -----
        - Duplicate PatientIDs are logged as warnings if found
        - Processed DataFrame is stored in self.enhanced_df
        """
        try:
            df = pd.read_csv(file_path)
            logging.info(f"Successfully read Enhanced_AdvancedNSCLC.csv file with shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")

            # Filter for specific PatientIDs if provided
            if patient_ids is not None:
                logging.info(f"Filtering for {len(patient_ids)} specific PatientIDs")
                df = df[df['PatientID'].isin(patient_ids)]
                logging.info(f"Successfully filtered Enhanced_AdvancedNSCLC.csv file with shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")
        
            # Convert categorical columns
            categorical_cols = ['Histology', 
                                'SmokingStatus',
                                'GroupStage']
        
            df[categorical_cols] = df[categorical_cols].astype('category')

            GROUP_STAGE_MAPPING = {
                # Stage 0/Occult
                'Stage 0': '0',
                'Occult': '0',
                
                # Stage I
                'Stage I': 'I',
                'Stage IA': 'I',
                'Stage IA1': 'I',
                'Stage IA2': 'I',
                'Stage IA3': 'I',
                'Stage IB': 'I',
                
                # Stage II
                'Stage II': 'II',
                'Stage IIA': 'II',
                'Stage IIB': 'II',
                
                # Stage III
                'Stage III': 'III',
                'Stage IIIA': 'III',
                'Stage IIIB': 'III',
                'Stage IIIC': 'III',
                
                # Stage IV
                'Stage IV': 'IV',
                'Stage IVA': 'IV',
                'Stage IVB': 'IV',
                
                # Unknown/Not reported
                'Group stage is not reported': 'Unknown'
            }

            # Recode stage variables using class-level mapping and create new column
            df['GroupStage_mod'] = df['GroupStage'].map(GROUP_STAGE_MAPPING).astype('category')

            # Drop original stage variables if specified
            if drop_stage:
                df = df.drop(columns=['GroupStage'])

            # Convert date columns
            date_cols = ['DiagnosisDate', 'AdvancedDiagnosisDate']
            for col in date_cols:
                df[col] = pd.to_datetime(df[col])

            # Generate new variables 
            df['days_diagnosis_to_adv'] = (df['AdvancedDiagnosisDate'] - df['DiagnosisDate']).dt.days
            df['adv_diagnosis_year'] = pd.Categorical(df['AdvancedDiagnosisDate'].dt.year)
    
            if drop_dates:
                df = df.drop(columns = ['AdvancedDiagnosisDate', 'DiagnosisDate'])

            # Check for duplicate PatientIDs
            if len(df) > df['PatientID'].nunique():
                logging.error(f"Duplicate PatientIDs found")
                return None

            logging.info(f"Successfully processed Enhanced_AdvancedNSCLC.csv file with final shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")
            return df

        except Exception as e:
            logging.error(f"Error processing Enhanced_AdvancedNSCLC.csv file: {e}")
            return None
    
        
# TESTING 
a = process_enhanced_adv(file_path="data_nsclc/Enhanced_AdvancedNSCLC.csv",
                         drop_stage = False,
                         drop_dates = False)

embed()