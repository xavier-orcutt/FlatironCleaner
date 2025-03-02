import pandas as pd
import numpy as np
import logging
import math 
import re 
from typing import Optional

logging.basicConfig(
    level = logging.INFO,                                 
    format = '%(asctime)s - %(levelname)s - %(message)s'  
)

class DataProcessorProstate:

    GROUP_STAGE_MAPPING = {        
        # Stage IV
        'IV': 'IV',
        'IVA': 'IV',
        'IVB': 'IV',

        # Stage III
        'III': 'III',
        'IIIA': 'III',
        'IIIB': 'III',
        'IIIC': 'III',

        # Stage II
        'II': 'II',
        'IIA': 'II',
        'IIB': 'II',
        'IIC': 'II',

        # Stage I
        'I': 'I',
        
        # Unknown
        'Unknown / Not documented': 'unknown'
    }

    T_STAGE_MAPPING = {
        'T4': 'T4',
        'T3': 'T3',
        'T3a': 'T3',
        'T3b': 'T3',
        'T2': 'T2',
        'T2a': 'T2',
        'T2b': 'T2',
        'T2c': 'T2',
        'T1': 'T1',
        'T1a': 'T1',
        'T1b': 'T1',
        'T1c': 'T1',
        'T0': 'T1',
        'TX': 'unknown',
        'Unknown / Not documented': 'unknown'
    }

    N_STAGE_MAPPING = {
        'N1': 'N1',
        'N0': 'N0',
        'NX': 'unknown',
        'Unknown / Not documented': 'unknown'
    }

    M_STAGE_MAPPING = {
        'M1': 'M1',
        'M1a': 'M1',
        'M1b': 'M1',
        'M1c': 'M1',
        'M0': 'M0',
        'Unknown / Not documented': 'unknown'
    }

    GLEASON_MAPPING = {
        '10': 5,
        '9': 5,
        '8': 4,  
        '4 + 3 = 7': 3,
        '7 (when breakdown not available)': 3,
        '3 + 4 = 7': 2, 
        'Less than or equal to 6': 1,  
        'Unknown / Not documented': 'unknown'
    }

    def __init__(self):
        self.enhanced_df = None

    def process_enhanced(self,
                         file_path: str,
                         index_date_column: str = 'MetDiagnosisDate',
                         patient_ids: list = None,
                         index_date_df: pd.DataFrame = None,
                         drop_stages: bool = True,
                         drop_dates: bool = True) -> pd.DataFrame: 
        """
        Processes Enhanced_MetProstate.csv to standardize categories, consolidate staging information, and calculate time-based metrics between key clinical events.
        
        The index date is used to determine castrate-resistance status by that time and to calculate time from diagnosis to castrate resistance. 
        The default index date is 'MetDiagnosisDate.' For an alternative index date, provide an index_date_df and specify the index_date_column accordingly.
        
        To process only specific patients, either:
        1. Provide patient_ids when using the default index date ('MetDiagnosisDate')
        2. Include only the desired PatientIDs in index_date_df when using a custom index date
        
        Parameters
        ----------
        file_path : str
            Path to Enhanced_MetProstate.csv file
        index_date_column : str
            name of column for index date of interest 
        patient_ids : list, optional
            List of specific PatientIDs to process. If None, processes all patients
        index_date_df : pd.DataFrame
            DataFrame containing unique PatientIDs and their corresponding index dates. Only data for PatientIDs present in this DataFrame will be processed
        drop_stages : bool, default=True
            If True, drops original staging columns (GroupStage, TStage, and MStage) after creating modified versions
        drop_dates : bool, default=True
            If True, drops date columns (DiagnosisDate, MetDiagnosisDate, and CRPCDate) after calculating durations

        Returns
        -------
        pd.DataFrame
            - PatientID : object
                unique patient identifier
            - GroupStage_mod : category
                consolidated overall staging (I-IV and unknown) at time of first diagnosis
            - TStage_mod : category
                consolidated tumor staging (T1-T4 and unknown) at time of first diagnosis
            - NStage_mod : category
                consolidated lymph node staging (N0, N1, and unknown) at time of first diagnosis
            - MStage_mod : category
                consolidated metastasis staging (M0, M1, and unknown) at time of first diagnosis
            - GleasonScore_mod : category
                consolidated Gleason scores into Grade Groups (1-5 and unknown) at time of first diagnosis 
            - Histology : category
                histology (adenocarcinoma and NOS) at time of initial diagnosis 
            - days_diagnosis_to_met : float
                days from first diagnosis to metastatic disease 
            - met_diagnosis_year : category
                year of metastatic diagnosis 
            - IsCRPC : Int64
                binary (0/1) indicator for CRPC, determined by whether CRPC date is earlier than the index date 
            - days_diagnosis_to_crpc : float
                days from diagnosis to CRPC, calculated only when CRPC date is prior to index date (i.e., IsCRPC == 1)
            - PSADiagnosis : float, ng/mL
                PSA at time of first diagnosis
            - PSAMetDiagnosis : float
                PSA at time of metastatic diagnosis 
            - psa_doubling : float, months
                PSA doubling time for those with both a PSA at time of first and metastatic diagnosis  
            - psa_velocity : float, ng/mL/month
                PSA velocity for those with both a PSA at time of first and metastatic diagnosis

            Original date columns (DiagnosisDate, MetDiagnosisDate, and CRPCDate) retained if drop_dates = False

        Notes
        -----
        Notable T-Stage consolidation decisions:
            - T0 is included in T1 
            - TX and Unknown/not documented are categorized as 'unknown' 

        Notable Gleanson score consolidation decisions: 
            - 7 (when breakdown not available) was placed into Grade Group 3

        PSA doubilng time formula: 
            - ln(2)/PSA slope
            - PSA slope = (ln(PSAMetDiagnosis) - ln(PSADiagnosis))/(MetDiagnosisDate - DiagnosisDate)

        PSA velocity formula: 
            - (PSAMetDiagnosis - PSADiagnosis)/(MetDiagnosisDate - DiagnosisDate)

        Output handling: 
        - Duplicate PatientIDs are logged as warnings if found but retained in output
        - Processed DataFrame is stored in self.enhanced_df
        """
        # Input validation
        if patient_ids is not None:
            if not isinstance(patient_ids, list):
                raise TypeError("patient_ids must be a list or None")
        
        if index_date_df is not None:
            if not isinstance(index_date_df, pd.DataFrame):
                raise ValueError("index_date_df must be a pandas DataFrame")
            if 'PatientID' not in index_date_df.columns:
                raise ValueError("index_date_df must contain a 'PatientID' column")
            if not index_date_column or index_date_column not in index_date_df.columns:
                raise ValueError('index_date_column not found in index_date_df')
            if index_date_df['PatientID'].duplicated().any():
                raise ValueError("index_date_df contains duplicate PatientID values, which is not allowed")

        try:
            df = pd.read_csv(file_path)
            logging.info(f"Successfully read Enhanced_MetProstate.csv file with shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")

            # Case 1: Using default MetDiagnosisDate with specific patients
            if index_date_column == 'MetDiagnosisDate' and patient_ids is not None:
                logging.info(f"Filtering for {len(patient_ids)} specific PatientIDs")
                df = df[df['PatientID'].isin(patient_ids)]
                logging.info(f"Successfully filtered Enhanced_MetProstate.csv file with shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")

            # Case 2: Using custom index date with index_date_df
            elif index_date_column != 'MetDiagnosisDate' and index_date_df is not None:
                index_date_df[index_date_column] = pd.to_datetime(index_date_df[index_date_column])
                df = df[df.PatientID.isin(index_date_df.PatientID)]
                df = pd.merge(
                    df,
                    index_date_df[['PatientID', index_date_column]],
                    on = 'PatientID',
                    how = 'left'
                )
                logging.info(f"Successfully merged Enhanced_MetProstate.csv df with index_date_df resulting in shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")

            # Case 3: Using default MetDiagnosisDate with all patients (no filtering)
            elif index_date_column == 'MetDiagnosisDate' and patient_ids is None:
                logging.info(f"No filtering applied. Using all {df['PatientID'].nunique()} patients in the dataset")

            # Case 4: Error case - custom index date without index_date_df
            else:
                logging.error("If index_date_column is not 'MetDiagnosisDate', an index_date_df must be provided")
                return None
        
            # Convert categorical columns
            categorical_cols = ['GroupStage',
                                'TStage', 
                                'NStage',
                                'MStage', 
                                'GleasonScore', 
                                'Histology']
            
            df[categorical_cols] = df[categorical_cols].astype('category')

            # Recode stage variables using class-level mapping and create new column
            df['GroupStage_mod'] = df['GroupStage'].map(self.GROUP_STAGE_MAPPING).astype('category')
            df['TStage_mod'] = df['TStage'].map(self.T_STAGE_MAPPING).astype('category')
            df['NStage_mod'] = df['NStage'].map(self.N_STAGE_MAPPING).astype('category')
            df['MStage_mod'] = df['MStage'].map(self.M_STAGE_MAPPING).astype('category')
            df['GleasonScore_mod'] = df['GleasonScore'].map(self.GLEASON_MAPPING).astype('category')

            # Drop original stage variables if specified
            if drop_stages:
                df = df.drop(columns=['GroupStage', 'TStage', 'NStage', 'MStage', 'GleasonScore'])

            # Convert date columns to datetime
            date_cols = ['DiagnosisDate', 'MetDiagnosisDate', 'CRPCDate']
            for col in date_cols:
                df[col] = pd.to_datetime(df[col])

            # Generate new time-based variables 
            df['days_diagnosis_to_met'] = (df['MetDiagnosisDate'] - df['DiagnosisDate']).dt.days
            df['met_diagnosis_year'] = pd.Categorical(df['MetDiagnosisDate'].dt.year)

            # Recoding IsCRPC to be 1 if CRPCDate is less than or equal to index date 
            # Calculate time from diagnosis to CRPC (presuming before metdiagnosis or index)
            if index_date_column == "MetDiagnosisDate":
                df['IsCRPC'] = np.where(df['CRPCDate'] <= df['MetDiagnosisDate'], 1, 0)
                df['days_diagnosis_to_crpc'] = np.where(df['IsCRPC'] == 1,
                                                        (df['CRPCDate'] - df['DiagnosisDate']).dt.days,
                                                        np.nan)
            else:
                df['IsCRPC'] = np.where(df['CRPCDate'] <= df[index_date_column], 1, 0)
                df['days_diagnosis_to_crpc'] = np.where(df['IsCRPC'] == 1,
                                                        (df['CRPCDate'] - df['DiagnosisDate']).dt.days,
                                                        np.nan)

            num_cols = ['PSADiagnosis', 'PSAMetDiagnosis']
            for col in num_cols:
                df[col] = pd.to_numeric(df[col], errors = 'coerce').astype('float')

            # Calculating PSA doubling time in months 
            df_doubling = (
                df
                .query('DiagnosisDate.notna()')
                .query('days_diagnosis_to_met > 30') # At least 30 days from first diagnosis to metastatic diagnosis
                .query('PSADiagnosis.notna()')
                .query('PSAMetDiagnosis.notna()')
                .query('PSAMetDiagnosis > PSADiagnosis') # Doubling time formula only makes sense for rising numbers 
                .assign(psa_doubling = lambda x: 
                        ((x['days_diagnosis_to_met']/30) * math.log(2))/
                        (np.log(x['PSAMetDiagnosis']) - 
                        np.log(x['PSADiagnosis']))
                        )
                [['PatientID', 'psa_doubling']]
            )

            # Calculating PSA velocity with time in months 
            df_velocity = (
                df
                .query('DiagnosisDate.notna()')
                .query('days_diagnosis_to_met > 30') # At least 30 days from first diagnosis to metastatic diagnosis
                .query('PSADiagnosis.notna()')
                .query('PSAMetDiagnosis.notna()')
                .assign(psa_velocity = lambda x: (x['PSAMetDiagnosis'] - x['PSADiagnosis']) / (x['days_diagnosis_to_met']/30))
                [['PatientID', 'psa_velocity']]
            )

            final_df = pd.merge(df, df_doubling, on = 'PatientID', how = 'left')
            final_df = pd.merge(final_df, df_velocity, on = 'PatientID', how = 'left')

            if drop_dates:
                final_df = final_df.drop(columns = ['MetDiagnosisDate', 'DiagnosisDate', 'CRPCDate'])

            # Check for duplicate PatientIDs
            if len(final_df) > final_df['PatientID'].nunique():
                duplicate_ids = final_df[final_df.duplicated(subset = ['PatientID'], keep = False)]['PatientID'].unique()
                logging.warning(f"Duplicate PatientIDs found: {duplicate_ids}")

            logging.info(f"Successfully processed Enhanced_MetProstate.csv file with final shape: {final_df.shape} and unique PatientIDs: {(final_df['PatientID'].nunique())}")
            self.enhanced_df = final_df
            return final_df

        except Exception as e:
            logging.error(f"Error processing Enhanced_MetProstate.csv file: {e}")
            return None 