import pandas as pd
import numpy as np
import logging
import re 
from typing import Optional

logging.basicConfig(
    level = logging.INFO,                                 
    format = '%(asctime)s - %(levelname)s - %(message)s'  
)

class DataProcessorNSCLC:

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
        'Group stage is not reported': 'unknown'
    }

    STATE_REGIONS_MAPPING = {
        'ME': 'northeast', 
        'NH': 'northeast',
        'VT': 'northeast', 
        'MA': 'northeast',
        'CT': 'northeast',
        'RI': 'northeast',  
        'NY': 'northeast', 
        'NJ': 'northeast', 
        'PA': 'northeast', 
        'IL': 'midwest', 
        'IN': 'midwest', 
        'MI': 'midwest', 
        'OH': 'midwest', 
        'WI': 'midwest',
        'IA': 'midwest',
        'KS': 'midwest',
        'MN': 'midwest',
        'MO': 'midwest', 
        'NE': 'midwest',
        'ND': 'midwest',
        'SD': 'midwest',
        'DE': 'south',
        'FL': 'south',
        'GA': 'south',
        'MD': 'south',
        'NC': 'south', 
        'SC': 'south',
        'VA': 'south',
        'DC': 'south',
        'WV': 'south',
        'AL': 'south',
        'KY': 'south',
        'MS': 'south',
        'TN': 'south',
        'AR': 'south',
        'LA': 'south',
        'OK': 'south',
        'TX': 'south',
        'AZ': 'west',
        'CO': 'west',
        'ID': 'west',
        'MT': 'west',
        'NV': 'west',
        'NM': 'west',
        'UT': 'west',
        'WY': 'west',
        'AK': 'west',
        'CA': 'west',
        'HI': 'west',
        'OR': 'west',
        'WA': 'west',
        'PR': 'unknown'
    }

    def __init__(self):
        self.enhanced_df = None
        self.demographics_df = None

    def process_enhanced_adv(self, 
                             file_path: str,
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
        Duplicate PatientIDs are logged as warnings if found
        Processed DataFrame is stored in self.enhanced_df
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

            # Recode stage variable using class-level mapping and create new column
            df['GroupStage_mod'] = df['GroupStage'].map(self.GROUP_STAGE_MAPPING).astype('category')

            # Drop original stage variable if specified
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
            self.enhanced_df = df
            return df

        except Exception as e:
            logging.error(f"Error processing Enhanced_AdvancedNSCLC.csv file: {e}")
            return None
        
    def process_demographics(self, 
                             file_path: str,
                             index_date_df: pd.DataFrame,
                             index_date_column: str,
                             drop_state: bool = True) -> pd.DataFrame:
        """
        Processes Demographics.csv by standardizing categorical variables, mapping states to census regions, and calculating age at index date.

        Parameters
        ----------
        file_path : str
            Path to Demographics.csv file
        index_dates_df : pd.DataFrame, optional
            DataFrame containing PatientID and index dates. Only demographics for PatientIDs present in this DataFrame will be processed
        index_date_column : str, optional
            Column name in index_date_df containing index date
        drop_state : bool, default = True
            If True, drops State column after mapping to regions

        Returns
        -------
        pd.DataFrame
            - PatientID : object
                unique patient identifier
            - Gender : category
                gender
            - Race : category
                race (White, Black or African America, Asian, Other Race)
            - Ethnicity : category
                ethnicity (Hispanic or Latino, Not Hispanic or Latino)
            - age : Int64
                age at index date 
            - region : category
                US Census Bureau region
            - State : category
                US state (if drop_state=False)
            
        Notes
        -----
        Imputation for Race and Ethnicity:
            - If Race='Hispanic or Latino', Race value is replaced with NaN
            - If Race='Hispanic or Latino', Ethnicity is set to 'Hispanic or Latino'
            - Otherwise, missing Race and Ethnicity values remain unchanged
        Ages calculated as <18 or >120 are logged as warning if found, but not removed
        Missing States are imputed as unknown during the mapping to regions
        Duplicate PatientIDs are logged as warnings if found
        Processed DataFrame is stored in self.demographics_df
        """
        # Input validation
        if not isinstance(index_date_df, pd.DataFrame):
            raise ValueError("index_date_df must be a pandas DataFrame")
        if 'PatientID' not in index_date_df.columns:
            raise ValueError("index_date_df must contain a 'PatientID' column")
        if not index_date_column or index_date_column not in index_date_df.columns:
            raise ValueError(f"Column '{index_date_column}' not found in index_date_df")
        
        try:
            df = pd.read_csv(file_path)
            logging.info(f"Successfully read Demographics.csv file with shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")

            # Initial data type conversions
            df['BirthYear'] = df['BirthYear'].astype('Int64')
            df['Gender'] = df['Gender'].astype('category')
            df['State'] = df['State'].astype('category')

            index_date_df[index_date_column] = pd.to_datetime(index_date_df[index_date_column])

            # Select PatientIDs that are included in the index_date_df the merge on 'left'
            df = df[df.PatientID.isin(index_date_df.PatientID)]
            df = pd.merge(
                df,
                index_date_df[['PatientID', index_date_column]], 
                on = 'PatientID',
                how = 'left'
            )

            df['age'] = df[index_date_column].dt.year - df['BirthYear']

            # Age validation
            mask_invalid_age = (df['age'] < 18) | (df['age'] > 120)
            if mask_invalid_age.any():
                logging.warning(f"Found {mask_invalid_age.sum()} ages outside valid range (18-120)")

            # Drop the index date column and BirthYear after age calculation
            df = df.drop(columns = [index_date_column, 'BirthYear'])

            # Race and Ethnicity processing
            # If Race == 'Hispanic or Latino', fill 'Hispanic or Latino' for Ethnicity
            df['Ethnicity'] = np.where(df['Race'] == 'Hispanic or Latino', 'Hispanic or Latino', df['Ethnicity'])

            # If Race == 'Hispanic or Latino' replace with Nan
            df['Race'] = np.where(df['Race'] == 'Hispanic or Latino', np.nan, df['Race'])
            df[['Race', 'Ethnicity']] = df[['Race', 'Ethnicity']].astype('category')
            
            # Region processing
            # Group states into Census-Bureau regions  
            df['region'] = (df['State']
                            .map(self.STATE_REGIONS_MAPPING)
                            .fillna('unknown')
                            .astype('category'))

            # Drop State varibale if specified
            if drop_state:               
                df = df.drop(columns = ['State'])

            # Check for duplicate PatientIDs
            if len(df) > df['PatientID'].nunique():
                logging.error(f"Duplicate PatientIDs found")
                return None
            
            logging.info(f"Successfully processed Demographics.csv file with final shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")
            self.enhanced_df = df
            return df

        except Exception as e:
            logging.error(f"Error processing Demographics.csv file: {e}")
            return None