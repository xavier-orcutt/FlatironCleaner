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

    LOINC_MAPPINGS = {
        'hemoglobin': ['718-7', '20509-6'],
        'wbc': ['26464-8', '6690-2'],
        'platelet': ['26515-7', '777-3', '778-1'],
        'creatinine': ['2160-0', '38483-4'],
        'bun': ['3094-0'],
        'sodium': ['2947-0', '2951-2'],
        'bicarbonate': ['1963-8', '1959-6', '14627-4', '1960-4', '2028-9'],
        'chloride': ['2075-0'],
        'potassium': ['6298-4', '2823-3'],
        'albumin': ['1751-7', '35706-1', '13980-8'],
        'calcium': ['17861-6', '49765-1'],
        'total_bilirubin': ['42719-5', '1975-2'],
        'ast': ['1920-8', '30239-8'],
        'alt': ['1742-6', '1743-4', '1744-2'],
        'alp': ['6768-6'],
        'psa': ['2857-1', '35741-8']
    }

    def __init__(self):
        self.enhanced_df = None
        self.demographics_df = None
        self.practice_df = None 
        self.biomarkers_df = None
        self.ecog_df = None 
        self.vitals_df = None
        self.insurance_df = None
        self.labs_df = None 

    def process_enhanced(self,
                         file_path: str,
                         index_date_column: str = 'MetDiagnosisDate',
                         patient_ids: list = None,
                         index_date_df: pd.DataFrame = None,
                         drop_stages: bool = True,
                         drop_dates: bool = True) -> Optional[pd.DataFrame]: 
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
        index_date_column : str, default = 'MetDiagnosisDate'
            name of column for index date of interest 
        patient_ids : list, optional
            List of specific PatientIDs to process. If None, processes all patients
        index_date_df : pd.DataFrame, optional 
            DataFrame containing unique PatientIDs and their corresponding index dates. Only data for PatientIDs present in this DataFrame will be processed
        drop_stages : bool, default=True
            If True, drops original staging columns (GroupStage, TStage, and MStage) after creating modified versions
        drop_dates : bool, default=True
            If True, drops date columns (DiagnosisDate, MetDiagnosisDate, and CRPCDate) after calculating durations

        Returns
        -------
        pd.DataFrame or None 
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
            - IsCRPC_index : Int64
                binary (0/1) indicator for CRPC, determined by whether CRPC date is earlier than the index date 
            - days_diagnosis_to_crpc : float
                days from diagnosis to CRPC, calculated only when CRPC date is prior to index date (i.e., IsCRPC == 1)
            - PSADiagnosis : float, ng/mL
                PSA at time of first diagnosis
            - PSAMetDiagnosis : float
                PSA at time of metastatic diagnosis 
            - psa_doubling_diag_met : float, months
                PSA doubling time from first diagnosis to metastatic disease for those with both a PSA at time of first and metastatic diagnosis, calculated only when PSA was higher at metastatic diagnosis than at initial diagnosis 
            - psa_velocity_diag_met : float, ng/mL/month
                PSA velocity from first diagnosis to metastatic disease for those with both a PSA at time of first and metastatic diagnosis

            Original date columns (DiagnosisDate, MetDiagnosisDate, and CRPCDate) retained if drop_dates = False

        Notes
        -----
        Notable T-Stage consolidation decisions:
            - T0 is included in T1 
            - TX and Unknown/not documented are categorized as 'unknown' 

        Notable Gleanson score consolidation decisions: 
            - 7 (when breakdown not available) was placed into Grade Group 3

        PSA doubling time calculation:
            - Only calculated for patients with PSA values at both initial diagnosis and metastatic diagnosis
            - Only valid when PSA at metastatic diagnosis is higher than at initial diagnosis
            - Formula: (ln(2)/PSA slope), measured in months
            - Where PSA slope = [ln(PSAMetDiagnosis) - ln(PSADiagnosis)]/[(MetDiagnosisDate - DiagnosisDate) in months]

        PSA velocity calculation:
            - Only calculated for patients with PSA values at both initial diagnosis and metastatic diagnosis
            - Formula: (PSAMetDiagnosis - PSADiagnosis)/[(MetDiagnosisDate - DiagnosisDate) in months], measured in ng/mL/month

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

            # Generate IsCRPC_index where 1 if CRPCDate is less than or equal to index date 
            # Calculate time from diagnosis to CRPC (presuming before metastatic diagnosis or index)
            if index_date_column == "MetDiagnosisDate":
                df['IsCRPC_index'] = np.where(df['CRPCDate'] <= df['MetDiagnosisDate'], 1, 0)
                df['days_diagnosis_to_crpc'] = np.where(df['IsCRPC_index'] == 1,
                                                        (df['CRPCDate'] - df['DiagnosisDate']).dt.days,
                                                        np.nan)
            else:
                df['IsCRPC_index'] = np.where(df['CRPCDate'] <= df[index_date_column], 1, 0)
                df['days_diagnosis_to_crpc'] = np.where(df['IsCRPC_index'] == 1,
                                                        (df['CRPCDate'] - df['DiagnosisDate']).dt.days,
                                                        np.nan)
            
            df = df.drop(columns = ['IsCRPC'])

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
                .query('PSADiagnosis > 0') 
                .query('PSAMetDiagnosis > 0')  
                .assign(psa_doubling_diag_met = lambda x: 
                        ((x['days_diagnosis_to_met']/30) * math.log(2))/
                        (np.log(x['PSAMetDiagnosis']) - 
                        np.log(x['PSADiagnosis']))
                        )
                [['PatientID', 'psa_doubling_diag_met']]
            )

            # Calculating PSA velocity with time in months 
            df_velocity = (
                df
                .query('DiagnosisDate.notna()')
                .query('days_diagnosis_to_met > 30') # At least 30 days from first diagnosis to metastatic diagnosis
                .query('PSADiagnosis.notna()')
                .query('PSAMetDiagnosis.notna()')
                .query('PSADiagnosis > 0') 
                .query('PSAMetDiagnosis > 0') 
                .assign(psa_velocity_diag_met = lambda x: (x['PSAMetDiagnosis'] - x['PSADiagnosis']) / (x['days_diagnosis_to_met']/30))
                [['PatientID', 'psa_velocity_diag_met']]
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
        
    def process_demographics(self,
                             file_path: str,
                             index_date_df: pd.DataFrame,
                             index_date_column: str,
                             drop_state: bool = True) -> Optional[pd.DataFrame]:
        """
        Processes Demographics.csv by standardizing categorical variables, mapping states to census regions, and calculating age at index date.

        Parameters
        ----------
        file_path : str
            Path to Demographics.csv file
        index_date_df : pd.DataFrame, optional
            DataFrame containing unique PatientIDs and their corresponding index dates. Only demographic data for PatientIDs present in this DataFrame will be processed
        index_date_column : str, optional
            Column name in index_date_df containing index date
        drop_state : bool, default = True
            If True, drops State column after mapping to regions

        Returns
        -------
        pd.DataFrame or None
            - PatientID : object
                unique patient identifier
            - Race_mod : category
                race (White, Black or African America, Asian, Other Race)
            - Ethnicity_mod : category
                ethnicity (Hispanic or Latino, Not Hispanic or Latino)
            - age : Int64
                age at index date (index year - birth year)
            - region : category
                Maps all 50 states, plus DC and Puerto Rico (PR), to a US Census Bureau region
            - State : category
                US state (if drop_state=False)
            
        Notes
        -----
        Data cleaning and processing: 
        - Imputation for Race and Ethnicity:
            - If Race='Hispanic or Latino', Race value is replaced with NaN
            - If Race='Hispanic or Latino' and Ethnicity is missing, Ethnicity is set to 'Hispanic or Latino'
            - Otherwise, missing Race and Ethnicity values remain unchanged
        - Ages calculated as <18 or >120 are logged as warning if found, but not removed
        - Missing States and Puerto Rico are imputed as unknown during the mapping to regions
        - Gender dropped since all males. 

        Output handling: 
        - Duplicate PatientIDs are logged as warnings if found but retained in output
        - Processed DataFrame is stored in self.demographics_df
        """
        # Input validation
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
            logging.info(f"Successfully read Demographics.csv file with shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")

            # Initial data type conversions
            df['BirthYear'] = df['BirthYear'].astype('Int64')
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
            # If Race == 'Hispanic or Latino' and Ethnicity is empty, fill 'Hispanic or Latino' for Ethnicity
            df['Ethnicity_mod'] = np.where((df['Race'] == 'Hispanic or Latino') & (df['Ethnicity'].isna()), 
                                            'Hispanic or Latino', 
                                            df['Ethnicity'])

            # If Race == 'Hispanic or Latino' replace with Nan
            df['Race_mod'] = np.where(df['Race'] == 'Hispanic or Latino', 
                                      np.nan, 
                                      df['Race'])

            df[['Race_mod', 'Ethnicity_mod']] = df[['Race_mod', 'Ethnicity_mod']].astype('category')
            df = df.drop(columns = ['Race', 'Ethnicity'])
            
            # Region processing
            # Group states into Census-Bureau regions  
            df['region'] = (df['State']
                            .map(self.STATE_REGIONS_MAPPING)
                            .fillna('unknown')
                            .astype('category'))

            # Drop State varibale if specified
            if drop_state:               
                df = df.drop(columns = ['State'])

            df = df.drop(columns = ['Gender'])

            # Check for duplicate PatientIDs
            if len(df) > df['PatientID'].nunique():
                duplicate_ids = df[df.duplicated(subset = ['PatientID'], keep = False)]['PatientID'].unique()
                logging.warning(f"Duplicate PatientIDs found: {duplicate_ids}")
            
            logging.info(f"Successfully processed Demographics.csv file with final shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")
            self.demographics_df = df
            return df

        except Exception as e:
            logging.error(f"Error processing Demographics.csv file: {e}")
            return None
        
    def process_practice(self,
                         file_path: str,
                         patient_ids: list = None) -> Optional[pd.DataFrame]:
        """
        Processes Practice.csv to consolidate practice types per patient into a single categorical value indicating academic, community, or both settings.

        Parameters
        ----------
        file_path : str
            Path to Practice.csv file
        patient_ids : list, optional
            List of PatientIDs to process. If None, processes all patients

        Returns
        -------
        pd.DataFrame or None 
            - PatientID : object
                unique patient identifier  
            - PracticeType_mod : category
                practice setting (ACADEMIC, COMMUNITY, or BOTH)

        Notes
        -----
        Output handling: 
        - PracticeID and PrimaryPhysicianID are removed 
        - Duplicate PatientIDs are logged as warnings if found but retained in output
        - Processed DataFrame is stored in self.practice_df
        """
        # Input validation
        if patient_ids is not None:
            if not isinstance(patient_ids, list):
                raise TypeError("patient_ids must be a list or None")
                
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
                if len(practice_types) == 0:
                    return 'UNKNOWN'
                if len(practice_types) > 1:
                    return 'BOTH'
                return practice_types[0]
            
            # Apply the function to the column containing sets
            grouped_df['PracticeType_mod'] = grouped_df['PracticeType'].apply(get_practice_type).astype('category')

            final_df = grouped_df[['PatientID', 'PracticeType_mod']]

            # Check for duplicate PatientIDs
            if len(final_df) > final_df['PatientID'].nunique():
                duplicate_ids = final_df[final_df.duplicated(subset = ['PatientID'], keep = False)]['PatientID'].unique()
                logging.warning(f"Duplicate PatientIDs found: {duplicate_ids}")
            
            logging.info(f"Successfully processed Practice.csv file with final shape: {final_df.shape} and unique PatientIDs: {(final_df['PatientID'].nunique())}")
            self.practice_df = final_df
            return final_df

        except Exception as e:
            logging.error(f"Error processing Practice.csv file: {e}")
            return None
        
    def process_biomarkers(self,
                           file_path: str,
                           index_date_df: pd.DataFrame,
                           index_date_column: str, 
                           days_before: Optional[int] = None,
                           days_after: int = 0) -> Optional[pd.DataFrame]:
        """
        Processes Enhanced_MetPC_Biomarkers.csv by determining biomarker status for each patient within a specified time window relative to an index date. 

        Parameters
        ----------
        file_path : str
            Path to Enhanced_MetPC_Biomarkers.csv file
        index_date_df : pd.DataFrame
            DataFrame containing unique PatientIDs and their corresponding index dates. Only biomarker data for PatientIDs present in this DataFrame will be processed
        index_date_column : str
            Column name in index_date_df containing the index date
        days_before : int | None, optional
            Number of days before the index date to include. Must be >= 0 or None. If None, includes all prior results. Default: None
        days_after : int, optional
            Number of days after the index date to include. Must be >= 0. Default: 0
        
        Returns
        -------
        pd.DataFrame or None
            - PatientID : object
                unique patient identifier
            - BRCA_status : category
                positive if ever-positive, negative if only-negative, otherwise unknown

        Notes
        ------
        Biomarker cleaning processing: 
        - BRCA status is classifed according to these as:
            - 'positive' if any test result is positive (ever-positive)
            - 'negative' if any test is negative without positives (only-negative) 
            - 'unknown' if all results are indeterminate
        
        - Missing biomarker data handling:
            - All PatientIDs from index_date_df are included in the output
            - Patients without any biomarker tests will have NaN values for all biomarker columns
            - Missing ResultDate is imputed with SpecimenReceivedDate

        Output handling: 
        - Duplicate PatientIDs are logged as warnings if found but retained in output
        - Processed DataFrame is stored in self.biomarkers_df
        """
        # Input validation
        if not isinstance(index_date_df, pd.DataFrame):
            raise ValueError("index_date_df must be a pandas DataFrame")
        if 'PatientID' not in index_date_df.columns:
            raise ValueError("index_date_df must contain a 'PatientID' column")
        if not index_date_column or index_date_column not in index_date_df.columns:
            raise ValueError('index_date_column not found in index_date_df')
        if index_date_df['PatientID'].duplicated().any():
            raise ValueError("index_date_df contains duplicate PatientID values, which is not allowed")
        
        if days_before is not None:
            if not isinstance(days_before, int) or days_before < 0:
                raise ValueError("days_before must be a non-negative integer or None")
        if not isinstance(days_after, int) or days_after < 0:
            raise ValueError("days_after must be a non-negative integer")

        try:
            df = pd.read_csv(file_path)
            logging.info(f"Successfully read Enhanced_MetPC_Biomarkers.csv file with shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")

            df['ResultDate'] = pd.to_datetime(df['ResultDate'])
            df['SpecimenReceivedDate'] = pd.to_datetime(df['SpecimenReceivedDate'])

            # Impute missing ResultDate with SpecimenReceivedDate
            df['ResultDate'] = np.where(df['ResultDate'].isna(), df['SpecimenReceivedDate'], df['ResultDate'])

            index_date_df[index_date_column] = pd.to_datetime(index_date_df[index_date_column])

            # Select PatientIDs that are included in the index_date_df the merge on 'left'
            df = df[df.PatientID.isin(index_date_df.PatientID)]
            df = pd.merge(
                    df,
                    index_date_df[['PatientID', index_date_column]],
                    on = 'PatientID',
                    how = 'left'
            )
            logging.info(f"Successfully merged Enhanced_MetPC_Biomarkers.csv df with index_date_df resulting in shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")
            
            # Create new variable 'index_to_result' that notes difference in days between resulted specimen and index date
            df['index_to_result'] = (df['ResultDate'] - df[index_date_column]).dt.days
            
            # Select biomarkers that fall within desired before and after index date
            if days_before is None:
                # Only filter for days after
                df_filtered = df[df['index_to_result'] <= days_after].copy()
            else:
                # Filter for both before and after
                df_filtered = df[
                    (df['index_to_result'] <= days_after) & 
                    (df['index_to_result'] >= -days_before)
                ].copy()

            # Process BRCA
            positive_values = {
                'BRCA1 mutation identified',
                'BRCA2 mutation identified',
                'Both BRCA1 and BRCA2 mutations identified',
                'BRCA mutation NOS' 
            }

            negative_values = {
                'No BRCA mutation',
                'Genetic Variant Favor Polymorphism',
            }

            brca_df = (
                df_filtered
                .query('BiomarkerName == "BRCA"')
                .groupby('PatientID')['BiomarkerStatus']
                .agg(lambda x: 'positive' if any(val in positive_values for val in x)
                    else ('negative' if any(val in negative_values for val in x)
                        else 'unknown'))
                .reset_index()
                .rename(columns={'BiomarkerStatus': 'BRCA_status'}) 
            )

            # Merge dataframes -- start with index_date_df to ensure all PatientIDs are included
            final_df = index_date_df[['PatientID']].copy()
            final_df = pd.merge(final_df, brca_df, on = 'PatientID', how = 'left')
            final_df['BRCA_status'] = final_df['BRCA_status'].astype('category')
            
            # Check for duplicate PatientIDs
            if len(final_df) > final_df['PatientID'].nunique():
                duplicate_ids = final_df[final_df.duplicated(subset = ['PatientID'], keep = False)]['PatientID'].unique()
                logging.warning(f"Duplicate PatientIDs found: {duplicate_ids}")

            logging.info(f"Successfully processed Enhanced_MetPC_Biomarkers.csv file with final shape: {final_df.shape} and unique PatientIDs: {(final_df['PatientID'].nunique())}")
            self.biomarkers_df = final_df
            return final_df

        except Exception as e:
            logging.error(f"Error processing Enhanced_MetPC_Biomarkers.csv file: {e}")
            return None
        
    def process_ecog(self, 
                     file_path: str,
                     index_date_df: pd.DataFrame,
                     index_date_column: str, 
                     days_before: int = 90,
                     days_after: int = 0, 
                     days_before_further: int = 180) -> Optional[pd.DataFrame]:
        """
        Processes ECOG.csv to determine patient ECOG scores and progression patterns relative 
        to a reference index date. Uses two different time windows for distinct clinical purposes:
        
        1. A smaller window near the index date to find the most clinically relevant ECOG score
            that represents the patient's status at that time point
        2. A larger lookback window to detect clinically significant ECOG progression,
            specifically looking for patients whose condition worsened from ECOG 0-1 to ≥2

        Parameters
        ----------
        file_path : str
            Path to ECOG.csv file
        index_date_df : pd.DataFrame
            DataFrame containing unique PatientIDs and their corresponding index dates. Only ECOGs for PatientIDs present in this DataFrame will be processed
        index_date_column : str
            Column name in index_date_df containing the index date
        days_before : int, optional
            Number of days before the index date to include. Must be >= 0. Default: 90
        days_after : int, optional
            Number of days after the index date to include. Must be >= 0. Default: 0
        days_before_further : int, optional
            Number of days before index date to look for ECOG progression (0-1 to ≥2). Must be >= 0. Consider
            selecting a larger integer than days_before to capture meaningful clinical deterioration over time.
            Default: 180
            
        Returns
        -------
        pd.DataFrame or None
            - PatientID : object
                unique patient identifier
            - ecog_index : category, ordered 
                ECOG score (0-4) closest to index date
            - ecog_newly_gte2 : Int64
                binary indicator (0/1) for ECOG increased from 0-1 to ≥2 in larger lookback window 

        Notes
        ------
        Data cleaning and processing: 
        - The function selects the most clinically relevant ECOG score using the following priority rules:
            1. ECOG closest to index date is selected by minimum absolute day difference
            2. For equidistant measurements, higher ECOG score is selected
        
        Output handling: 
        - All PatientIDs from index_date_df are included in the output and values will be NaN for patients without ECOG values
        - Duplicate PatientIDs are logged as warnings if found but retained in output
        - Processed DataFrame is stored in self.ecog_df
        """
        # Input validation
        if not isinstance(index_date_df, pd.DataFrame):
            raise ValueError("index_date_df must be a pandas DataFrame")
        if 'PatientID' not in index_date_df.columns:
            raise ValueError("index_date_df must contain a 'PatientID' column")
        if not index_date_column or index_date_column not in index_date_df.columns:
            raise ValueError('index_date_column not found in index_date_df')
        if index_date_df['PatientID'].duplicated().any():
            raise ValueError("index_date_df contains duplicate PatientID values, which is not allowed")
        
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
                    ecog_index = lambda x: x['ecog_index'].astype(pd.CategoricalDtype(categories = [0, 1, 2, 3, 4], ordered = True))
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
            final_df['ecog_index'] = final_df['ecog_index'].astype(pd.CategoricalDtype(categories=[0, 1, 2, 3, 4], ordered=True))
            final_df['ecog_newly_gte2'] = final_df['ecog_newly_gte2'].astype('Int64')

            # Check for duplicate PatientIDs
            if len(final_df) > final_df['PatientID'].nunique():
                duplicate_ids = final_df[final_df.duplicated(subset = ['PatientID'], keep = False)]['PatientID'].unique()
                logging.warning(f"Duplicate PatientIDs found: {duplicate_ids}")
                
            logging.info(f"Successfully processed ECOG.csv file with final shape: {final_df.shape} and unique PatientIDs: {(final_df['PatientID'].nunique())}")
            self.ecog_df = final_df
            return final_df

        except Exception as e:
            logging.error(f"Error processing ECOG.csv file: {e}")
            return None
        
    def process_vitals(self,
                       file_path: str,
                       index_date_df: pd.DataFrame,
                       index_date_column: str, 
                       weight_days_before: int = 90,
                       days_after: int = 0,
                       vital_summary_lookback: int = 180, 
                       abnormal_reading_threshold: int = 2) -> Optional[pd.DataFrame]:
        """
        Processes Vitals.csv to determine patient BMI, weight, change in weight, and vital sign abnormalities
        within a specified time window relative to an index date. Two different time windows are used:
        
        1. A smaller window near the index date to find weight and BMI at that time point
        2. A larger lookback window to detect clinically significant vital sign abnormalities 
        suggesting possible deterioration

        Parameters
        ----------
        file_path : str
            Path to Vitals.csv file
        index_date_df : pd.DataFrame
            DataFrame containing unique PatientIDs and their corresponding index dates. Only vitals for PatientIDs present in this DataFrame will be processed
        index_date_column : str
            Column name in index_date_df containing the index date
        weight_days_before : int, optional
            Number of days before the index date to include for weight and BMI calculations. Must be >= 0. Default: 90
        days_after : int, optional
            Number of days after the index date to include for weight and BMI calculations. Also used as the end point for 
            vital sign abnormalities and weight change calculations. Must be >= 0. Default: 0
        vital_summary_lookback : int, optional
            Number of days before index date to assess for weight change, hypotension, tachycardia, and fever. Must be >= 0. Default: 180
        abnormal_reading_threshold: int, optional 
            Number of abnormal readings required to flag a patient with a vital sign abnormality (hypotension, tachycardia, 
            fevers, hypoxemia). Must be >= 1. Default: 2

        Returns
        -------
        pd.DataFrame or None 
            - PatientID : object 
                unique patient identifier
            - weight_index : float
                weight in kg closest to index date within specified window (index_date - weight_days_before) to (index_date + weight_days_after)
            - bmi_index : float
                BMI closest to index date within specified window (index_date - weight_days_before) to (index_date + days_after)
            - percent_change_weight : float
                percentage change in weight over period from (index_date - vital_summary_lookback) to (index_date + days_after)
            - hypotension : Int64
                binary indicator (0/1) for systolic blood pressure <90 mmHg on ≥{abnormal_reading_threshold} separate readings 
                between (index_date - vital_summary_lookback) and (index_date + days_after)
            - tachycardia : Int64
                binary indicator (0/1) for heart rate >100 bpm on ≥{abnormal_reading_threshold} separate readings 
                between (index_date - vital_summary_lookback) and (index_date + days_after)
            - fevers : Int64
                binary indicator (0/1) for temperature >=38°C on ≥{abnormal_reading_threshold} separate readings 
                between (index_date - vital_summary_lookback) and (index_date + days_after)
            - hypoxemia : Int64
                binary indicator (0/1) for SpO2 <90% on ≥{abnormal_reading_threshold} separate readings 
                between (index_date - vital_summary_lookback) and (index_date + days_after)

        Notes
        -----
        Data cleaning and processing: 
        - Missing TestResultCleaned values are imputed using TestResult. For those where units are ambiguous, unit conversion is based on thresholds:
            - For weight: 
                Values >140 are presumed to be in pounds and converted to kg (divided by 2.2046)
                Values <70 are presumed to be already in kg and kept as is
                Values between 70-140 are considered ambiguous and not imputed
            - For height: 
                Values between 55-80 are presumed to be in inches and converted to cm (multiplied by 2.54)
                Values between 140-220 are presumed to be already in cm and kept as is
                Values outside these ranges are considered ambiguous and not imputed
            - For temperature: 
                Values >45 are presumed to be in Fahrenheit and converted to Celsius using (F-32)*5/9
                Values ≤45 are presumed to be already in Celsius
        - Weight closest to index date is selected by minimum absolute day difference
        - BMI is calculated using closest weight to index within specified window and mean height over patient's entire data range (weight(kg)/height(m)²)
        - BMI calucalted as <13 are considered implausible and removed
        - Percent change in weight is calculated as ((end_weight - start_weight) / start_weight) * 100
        - TestDate rather than ResultDate is used since TestDate is always populated and, for vital signs, the measurement date (TestDate) and result date (ResultDate) should be identical since vitals are recorded in real-time
        
        Output handling: 
        - All PatientIDs from index_date_df are included in the output and values will be NaN for patients without weight, BMI, or percent_change_weight, but set to 0 for hypotension, tachycardia, fevers, and hypoxemia 
        - Duplicate PatientIDs are logged as warnings but retained in output
        - Results are stored in self.vitals_df attribute
        """
        # Input validation
        if not isinstance(index_date_df, pd.DataFrame):
            raise ValueError("index_date_df must be a pandas DataFrame")
        if 'PatientID' not in index_date_df.columns:
            raise ValueError("index_date_df must contain a 'PatientID' column")
        if not index_date_column or index_date_column not in index_date_df.columns:
            raise ValueError('index_date_column not found in index_date_df')
        if index_date_df['PatientID'].duplicated().any():
            raise ValueError("index_date_df contains duplicate PatientID values, which is not allowed")
        
        if not isinstance(weight_days_before, int) or weight_days_before < 0:
            raise ValueError("weight_days_before must be a non-negative integer")
        if not isinstance(days_after, int) or days_after < 0:
            raise ValueError("days_after must be a non-negative integer")
        if not isinstance(vital_summary_lookback, int) or vital_summary_lookback < 0:
            raise ValueError("vital_summary_lookback must be a non-negative integer")
        if not isinstance(abnormal_reading_threshold, int) or abnormal_reading_threshold < 1:
            raise ValueError("abnormal_reading_threshold must be an integer ≥1")

        try:
            df = pd.read_csv(file_path, low_memory = False)
            logging.info(f"Successfully read Vitals.csv file with shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")

            df['TestDate'] = pd.to_datetime(df['TestDate'])
            df['TestResult'] = pd.to_numeric(df['TestResult'], errors = 'coerce').astype('float')

            index_date_df[index_date_column] = pd.to_datetime(index_date_df[index_date_column])

            # Select PatientIDs that are included in the index_date_df the merge on 'left'
            df = df[df.PatientID.isin(index_date_df.PatientID)]
            df = pd.merge(
                df,
                index_date_df[['PatientID', index_date_column]],
                on = 'PatientID',
                how = 'left'
            )
            logging.info(f"Successfully merged Vitals.csv df with index_date_df resulting in shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")
                        
            # Create new variable 'index_to_vital' that notes difference in days between vital date and index date
            df['index_to_vital'] = (df['TestDate'] - df[index_date_column]).dt.days
            
            # Select weight vitals, impute missing TestResultCleaned, and filter for weights in selected window  
            weight_df = df.query('Test == "body weight"').copy()
            mask_needs_imputation = weight_df['TestResultCleaned'].isna() & weight_df['TestResult'].notna()
            
            imputed_weights = weight_df.loc[mask_needs_imputation, 'TestResult'].apply(
                lambda x: x/2.2046 if x > 140  # Convert to kg since likely lbs 
                else x if x < 70  # Keep as is if likely kg 
                else None  # Leave as null if ambiguous
            )
            
            weight_df.loc[mask_needs_imputation, 'TestResultCleaned'] = imputed_weights
            weight_df = weight_df.query('TestResultCleaned > 0')
            
            df_weight_filtered = weight_df[
                (weight_df['index_to_vital'] <= days_after) & 
                (weight_df['index_to_vital'] >= -weight_days_before)].copy()

            # Select weight closest to index date 
            weight_index_df = (
                df_weight_filtered
                .assign(abs_days_to_index = lambda x: abs(x['index_to_vital']))
                .sort_values(
                    by=['PatientID', 'abs_days_to_index', 'TestResultCleaned'], 
                    ascending=[True, True, True]) # Last True selects smallest weight for ties 
                .groupby('PatientID')
                .first()
                .reset_index()
                [['PatientID', 'TestResultCleaned']]
                .rename(columns = {'TestResultCleaned': 'weight_index'})
            )
            
            # Impute missing TestResultCleaned heights using TestResult 
            height_df = df.query('Test == "body height"')
            mask_needs_imputation = height_df['TestResultCleaned'].isna() & height_df['TestResult'].notna()
                
            imputed_heights = height_df.loc[mask_needs_imputation, 'TestResult'].apply(
                lambda x: x * 2.54 if 55 <= x <= 80  # Convert to cm if likely inches (about 4'7" to 6'7")
                else x if 140 <= x <= 220  # Keep as is if likely cm (about 4'7" to 7'2")
                else None  # Leave as null if implausible or ambiguous
            )

            height_df.loc[mask_needs_imputation, 'TestResultCleaned'] = imputed_heights

            # Select mean height for patients across all time points
            height_df = (
                height_df
                .groupby('PatientID')['TestResultCleaned'].mean()
                .reset_index()
                .assign(TestResultCleaned = lambda x: x['TestResultCleaned']/100)
                .rename(columns = {'TestResultCleaned': 'height'})
            )
            
            # Merge height_df with weight_df and calculate BMI
            weight_index_df = pd.merge(weight_index_df, height_df, on = 'PatientID', how = 'left')
            
            # Check if both weight and height are present
            has_both_measures = weight_index_df['weight_index'].notna() & weight_index_df['height'].notna()
            
            # Only calculate BMI where both measurements exist
            weight_index_df.loc[has_both_measures, 'bmi_index'] = (
                weight_index_df.loc[has_both_measures, 'weight_index'] / 
                weight_index_df.loc[has_both_measures, 'height']**2
            )

            # Replace implausible BMI values with NaN
            implausible_bmi = weight_index_df['bmi_index'] < 13
            weight_index_df.loc[implausible_bmi, 'bmi_index'] = np.nan
                    
            weight_index_df = weight_index_df.drop(columns=['height'])

            # Calculate change in weight 
            df_change_weight_filtered = weight_df[
                (weight_df['index_to_vital'] <= days_after) & 
                (weight_df['index_to_vital'] >= -vital_summary_lookback)].copy()
            
            change_weight_df = (
                df_change_weight_filtered
                .sort_values(['PatientID', 'TestDate'])
                .groupby('PatientID')
                .filter(lambda x: len(x) >= 2) # Only calculate change in weight for patients >= 2 weight readings
                .groupby('PatientID')
                .agg({'TestResultCleaned': lambda x:
                    ((x.iloc[-1]-x.iloc[0])/x.iloc[0])*100 if x.iloc[0] != 0 and pd.notna(x.iloc[0]) and pd.notna(x.iloc[-1]) # (end-start)/start
                    else None
                    })
                .reset_index()
                .rename(columns = {'TestResultCleaned': 'percent_change_weight'})
            )

            # Create new window period for vital sign abnormalities 
            df_summary_filtered = df[
                (df['index_to_vital'] <= days_after) & 
                (df['index_to_vital'] >= -vital_summary_lookback)].copy()
            
            # Calculate hypotension indicator 
            bp_df = df_summary_filtered.query("Test == 'systolic blood pressure'").copy()

            bp_df['TestResultCleaned'] = np.where(bp_df['TestResultCleaned'].isna(),
                                                  bp_df['TestResult'],
                                                  bp_df['TestResultCleaned'])

            hypotension_df = (
                bp_df
                .sort_values(['PatientID', 'TestDate'])
                .groupby('PatientID')
                .agg({
                    'TestResultCleaned': lambda x: (
                        sum(x < 90) >= abnormal_reading_threshold) 
                })
                .reset_index()
                .rename(columns = {'TestResultCleaned': 'hypotension'})
            )

            # Calculate tachycardia indicator
            hr_df = df_summary_filtered.query("Test == 'heart rate'").copy()

            hr_df['TestResultCleaned'] = np.where(hr_df['TestResultCleaned'].isna(),
                                                  hr_df['TestResult'],
                                                  hr_df['TestResultCleaned'])

            tachycardia_df = (
                hr_df 
                .sort_values(['PatientID', 'TestDate'])
                .groupby('PatientID')
                .agg({
                    'TestResultCleaned': lambda x: (
                        sum(x > 100) >= abnormal_reading_threshold) 
                })
                .reset_index()
                .rename(columns = {'TestResultCleaned': 'tachycardia'})
            )

            # Calculate fevers indicator
            temp_df = df_summary_filtered.query("Test == 'body temperature'").copy()
            
            mask_needs_imputation = temp_df['TestResultCleaned'].isna() & temp_df['TestResult'].notna()
            
            imputed_temps = temp_df.loc[mask_needs_imputation, 'TestResult'].apply(
                lambda x: (x - 32) * 5/9 if x > 45  # Convert to C since likely F
                else x # Leave as C
            )

            temp_df.loc[mask_needs_imputation, 'TestResultCleaned'] = imputed_temps

            fevers_df = (
                temp_df
                .sort_values(['PatientID', 'TestDate'])
                .groupby('PatientID')
                .agg({
                    'TestResultCleaned': lambda x: sum(x >= 38) >= abnormal_reading_threshold 
                })
                .reset_index()
                .rename(columns={'TestResultCleaned': 'fevers'})
            )

            # Calculate hypoxemia indicator 
            oxygen_df = df_summary_filtered.query("Test == 'oxygen saturation in arterial blood by pulse oximetry'").copy()

            oxygen_df['TestResultCleaned'] = np.where(oxygen_df['TestResultCleaned'].isna(),
                                                      oxygen_df['TestResult'],
                                                      oxygen_df['TestResultCleaned'])
            
            hypoxemia_df = (
                oxygen_df
                .sort_values(['PatientID', 'TestDate'])
                .groupby('PatientID')
                .agg({
                    'TestResultCleaned': lambda x: sum(x < 90) >= abnormal_reading_threshold 
                })
                .reset_index()
                .rename(columns={'TestResultCleaned': 'hypoxemia'})
            )

            # Merge dataframes - start with index_date_df to ensure all PatientIDs are included
            final_df = index_date_df[['PatientID']].copy()
            final_df = pd.merge(final_df, weight_index_df, on = 'PatientID', how = 'left')
            final_df = pd.merge(final_df, change_weight_df, on = 'PatientID', how = 'left')
            final_df = pd.merge(final_df, hypotension_df, on = 'PatientID', how = 'left')
            final_df = pd.merge(final_df, tachycardia_df, on = 'PatientID', how = 'left')
            final_df = pd.merge(final_df, fevers_df, on = 'PatientID', how = 'left')
            final_df = pd.merge(final_df, hypoxemia_df, on = 'PatientID', how = 'left')

            boolean_columns = ['hypotension', 'tachycardia', 'fevers', 'hypoxemia']
            for col in boolean_columns:
                final_df[col] = final_df[col].fillna(0).astype('Int64')
            
            if len(final_df) > final_df['PatientID'].nunique():
                duplicate_ids = final_df[final_df.duplicated(subset=['PatientID'], keep=False)]['PatientID'].unique()
                logging.warning(f"Duplicate PatientIDs found: {duplicate_ids}")

            logging.info(f"Successfully processed Vitals.csv file with final shape: {final_df.shape} and unique PatientIDs: {(final_df['PatientID'].nunique())}")
            self.vitals_df = final_df
            return final_df

        except Exception as e:
            logging.error(f"Error processing Vitals.csv file: {e}")
            return None
    
    def process_insurance(self,
                          file_path: str,
                          index_date_df: pd.DataFrame,
                          index_date_column: str,
                          days_before: Optional[int] = None,
                          days_after: int = 0,
                          missing_date_strategy: str = 'conservative') -> Optional[pd.DataFrame]:
        """
        Processes insurance data to identify insurance coverage relative to a specified index date.
        Insurance types are grouped into four categories: Medicare, Medicaid, Commercial, and Other Insurance. 
        
        Parameters
        ----------
        file_path : str
            Path to Insurance.csv file
        index_date_df : pd.DataFrame
            DataFrame containing unique PatientIDs and their corresponding index dates. Only insurances for PatientIDs present in this DataFrame will be processed
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
        pd.DataFrame or None
            - PatientID : object
                unique patient identifier
            - medicare : Int64
                binary indicator (0/1) for Medicare coverage
            - medicaid : Int64
                binary indicator (0/1) for Medicaid coverage
            - commercial : Int64
                binary indicator (0/1) for commercial insurance coverage
            - other_insurance : Int64
                binary indicator (0/1) for other insurance types (eg., other payer, other government program, patient assistance program, self pay, and workers compensation)

        Notes
        -----
        Insurance is considered active if:
        1. StartDate falls before or during the specified time window AND
        2. Either:
            - EndDate is missing (considered still active) OR
            - EndDate falls on or after the start of the time window 

        Date filtering:
        - Records with StartDate or EndDate before 1900-01-01 are excluded to prevent integer overflow issues
        when calculating date differences. This is a data quality measure as extremely old dates are likely
        erroneous and can cause numerical problems in pandas datetime calculations.
        - About 5% of the full dataset has misisng StartDate and EndDate.

        Insurance categorization logic:
        1. Original payer categories are preserved but enhanced with hybrid categories:
        - Commercial_Medicare: Commercial plans with Medicare Advantage or Supplement
        - Commercial_Medicaid: Commercial plans with Managed Medicaid
        - Commercial_Medicare_Medicaid: Commercial plans with both Medicare and Medicaid indicators
        - Other_Medicare: Other government program or other payer plans with Medicare Advantage or Supplement
        - Other_Medicaid: Other government program or other payer plans with Managed Medicaid
        - Other_Medicare_Medicaid: Other government program or other payer plans with both Medicare and Medicaid indicators
            
        2. Final insurance indicators are set as follows:
        - medicare: Set to 1 for PayerCategory = Medicare, Commercial_Medicare, Commercial_Medicare_Medicaid, Other_Medicare, or Other_Medicare_Medicaid
        - medicaid: Set to 1 for PayerCategory = Medicaid, Commercial_Medicaid, Commercial_Medicare_Medicaid, Other_Medicaid, or Other_Medicare_Medicaid
        - commercial: Set to 1 for PayerCategory = Commercial Health Plan, Commercial_Medicare, Commercial_Medicaid, or Commercial_Medicare_Medicaid
        - other_insurance: Set to 1 for PayerCategory = Other Payer - Type Unknown, Other Government Program, Patient Assistance Program, Self Pay, 
            Workers Compensation, Other_Medicare, Other_Medicaid, Other_Medicare_Medicaid

        Output handling:
        - All PatientIDs from index_date_df are included in the output and value is set to 0 for those without insurance type 
        - Duplicate PatientIDs are logged as warnings but retained in output
        - Results are stored in self.insurance_df attribute
        """
        # Input validation
        if not isinstance(index_date_df, pd.DataFrame):
            raise ValueError("index_date_df must be a pandas DataFrame")
        if 'PatientID' not in index_date_df.columns:
            raise ValueError("index_date_df must contain a 'PatientID' column")
        if not index_date_column or index_date_column not in index_date_df.columns:
            raise ValueError('index_date_column not found in index_date_df')
        if index_date_df['PatientID'].duplicated().any():
            raise ValueError("index_date_df contains duplicate PatientID values, which is not allowed")
        
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
            # Filter for Enddate missing or after 1900-01-01
            df = df[(df['EndDate'].isna()) | (df['EndDate'] > pd.Timestamp('1900-01-01'))]

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

            # Reclassify Commerical Health Plans that have elements of Medicare, Medicaid, or Both
            # Identify Commerical plus Medicare Advantage or Supplement plans
            df['PayerCategory'] = np.where((df['PayerCategory'] == 'Commercial Health Plan') & ((df['IsMedicareAdv'] == 'Yes') | (df['IsMedicareSupp'] == 'Yes')) & (df['IsMedicareMedicaid'] != 'Yes') & (df['IsManagedMedicaid'] != 'Yes'),
                                            'Commercial_Medicare',
                                            df['PayerCategory'])

            # Identify Commerical plus Managed Medicaid plans
            df['PayerCategory'] = np.where((df['PayerCategory'] == 'Commercial Health Plan') & (df['IsManagedMedicaid'] == 'Yes') & (df['IsMedicareMedicaid'] != 'Yes') & (df['IsMedicareAdv'] != 'Yes') & (df['IsMedicareSupp'] != 'Yes'),
                                            'Commercial_Medicaid',
                                            df['PayerCategory'])
            
            # Identify Commercial plus MedicareMedicaid plan
            df['PayerCategory'] = np.where((df['PayerCategory'] == 'Commercial Health Plan') & (df['IsMedicareMedicaid'] == 'Yes'),
                                            'Commercial_Medicare_Medicaid',
                                            df['PayerCategory'])

            # Identify Commercial plus Managed Medicaid and Medicare Advantage or Supplement plans
            df['PayerCategory'] = np.where((df['PayerCategory'] == 'Commercial Health Plan') & (df['IsManagedMedicaid'] == 'Yes') & ((df['IsMedicareAdv'] == 'Yes') | (df['IsMedicareSupp'] == 'Yes')),
                                            'Commercial_Medicare_Medicaid',
                                            df['PayerCategory'])
            

            # Reclassify Other Health Plans that have elements of Medicare, Medicaid, or Both
            # Identify Other plus Medicare Advantage or Supplement plans
            df['PayerCategory'] = np.where(((df['PayerCategory'] == 'Other Payer - Type Unknown') | (df['PayerCategory'] == 'Other Government Program')) & ((df['IsMedicareAdv'] == 'Yes') | (df['IsMedicareSupp'] == 'Yes')) & (df['IsMedicareMedicaid'] != 'Yes') & (df['IsManagedMedicaid'] != 'Yes'),
                                            'Other_Medicare',
                                            df['PayerCategory'])

            # Identify Other plus Managed Medicaid plans
            df['PayerCategory'] = np.where(((df['PayerCategory'] == 'Other Payer - Type Unknown') | (df['PayerCategory'] == 'Other Government Program')) & (df['IsManagedMedicaid'] == 'Yes') & (df['IsMedicareMedicaid'] != 'Yes') & (df['IsMedicareAdv'] != 'Yes') & (df['IsMedicareSupp'] != 'Yes'),
                                            'Other_Medicaid',
                                            df['PayerCategory'])
            
            # Identify Other plus MedicareMedicaid plan
            df['PayerCategory'] = np.where(((df['PayerCategory'] == 'Other Payer - Type Unknown') | (df['PayerCategory'] == 'Other Government Program')) & (df['IsMedicareMedicaid'] == 'Yes'),
                                            'Other_Medicare_Medicaid',
                                            df['PayerCategory'])

            # Identify Other plus Managed Medicaid and Medicare Advantage or Supplement plans
            df['PayerCategory'] = np.where(((df['PayerCategory'] == 'Other Payer - Type Unknown') | (df['PayerCategory'] == 'Other Government Program')) & (df['IsManagedMedicaid'] == 'Yes') & ((df['IsMedicareAdv'] == 'Yes') | (df['IsMedicareSupp'] == 'Yes')),
                                            'Other_Medicare_Medicaid',
                                            df['PayerCategory'])
            
            # Add hybrid insurance schems to mapping
            self.INSURANCE_MAPPING['Commercial_Medicare'] = 'commercial_medicare'
            self.INSURANCE_MAPPING['Commercial_Medicaid'] = 'commercial_medicaid'
            self.INSURANCE_MAPPING['Commercial_Medicare_Medicaid'] = 'commercial_medicare_medicaid'
            self.INSURANCE_MAPPING['Other_Medicare'] = 'other_medicare'
            self.INSURANCE_MAPPING['Other_Medicaid'] = 'other_medicaid'
            self.INSURANCE_MAPPING['Other_Medicare_Medicaid'] = 'other_medicare_medicaid'

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

            df_filtered['PayerCategory'] = df_filtered['PayerCategory'].replace(self.INSURANCE_MAPPING)

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

            # Adjust column indicators for commercial and other with medicare and medicaid plans
            if 'commercial_medicare' in final_df.columns:
                final_df.loc[final_df['commercial_medicare'] == 1, 'commercial'] = 1
                final_df.loc[final_df['commercial_medicare'] == 1, 'medicare'] = 1
            
            if 'commercial_medicaid' in final_df.columns:
                final_df.loc[final_df['commercial_medicaid'] == 1, 'commercial'] = 1
                final_df.loc[final_df['commercial_medicaid'] == 1, 'medicaid'] = 1

            if 'commercial_medicare_medicaid' in final_df.columns:
                final_df.loc[final_df['commercial_medicare_medicaid'] == 1, 'commercial'] = 1
                final_df.loc[final_df['commercial_medicare_medicaid'] == 1, 'medicare'] = 1
                final_df.loc[final_df['commercial_medicare_medicaid'] == 1, 'medicaid'] = 1
            
            if 'other_medicare' in final_df.columns:
                final_df.loc[final_df['other_medicare'] == 1, 'other_insurance'] = 1
                final_df.loc[final_df['other_medicare'] == 1, 'medicare'] = 1
            
            if 'other_medicaid' in final_df.columns:
                final_df.loc[final_df['other_medicaid'] == 1, 'other_insurance'] = 1
                final_df.loc[final_df['other_medicaid'] == 1, 'medicaid'] = 1

            if 'other_medicare_medicaid' in final_df.columns:
                final_df.loc[final_df['other_medicare_medicaid'] == 1, 'other_insurance'] = 1
                final_df.loc[final_df['other_medicare_medicaid'] == 1, 'medicare'] = 1
                final_df.loc[final_df['other_medicare_medicaid'] == 1, 'medicaid'] = 1

            # Merger index_date_df to ensure all PatientIDs are included
            final_df = pd.merge(index_date_df[['PatientID']], final_df, on = 'PatientID', how = 'left')
            
            # Ensure all core insurance columns exist 
            core_insurance_columns = ['medicare', 'medicaid', 'commercial', 'other_insurance']
            for col in core_insurance_columns:
                if col not in final_df.columns:
                    final_df[col] = 0
                final_df[col] = final_df[col].fillna(0).astype('Int64')

            # Safely drop hybrid columns if they exist
            hybrid_columns = ['commercial_medicare', 
                              'commercial_medicaid', 
                              'commercial_medicare_medicaid',
                              'other_medicare', 
                              'other_medicaid', 
                              'other_medicare_medicaid']
            
            # Drop hybrid columns; errors = 'ignore' prevents error in the setting when column doesn't exist 
            final_df = final_df.drop(columns=hybrid_columns, errors='ignore')

            # Check for duplicate PatientIDs
            if len(final_df) > final_df['PatientID'].nunique():
                duplicate_ids = final_df[final_df.duplicated(subset = ['PatientID'], keep = False)]['PatientID'].unique()
                logging.warning(f"Duplicate PatientIDs found: {duplicate_ids}")

            logging.info(f"Successfully processed Insurance.csv file with final shape: {final_df.shape} and unique PatientIDs: {(final_df['PatientID'].nunique())}")
            self.insurance_df = final_df
            return final_df

        except Exception as e:
            logging.error(f"Error processing Insurance.csv file: {e}")
            return None
    
    def process_labs(self,
                     file_path: str,
                     index_date_df: pd.DataFrame,
                     index_date_column: str, 
                     additional_loinc_mappings: dict = None,
                     days_before: int = 90,
                     days_after: int = 0,
                     summary_lookback: int = 180) -> Optional[pd.DataFrame]:
        """
        Processes Lab.csv to determine patient lab values within a specified time window relative to an index date. Returns CBC, CMP, and PSA total values 
        nearest to index date, along with summary statistics (max, min, standard deviation, and slope) calculated over the summary period. PSA doubling time
        is also calculated over summary period. Additional lab tests can be included by providing corresponding LOINC code mappings.

        Parameters
        ----------
        file_path : str
            Path to Labs.csv file
        index_date_df : pd.DataFrame
            DataFrame containing unique PatientIDs and their corresponding index dates. Only labs for PatientIDs present in this DataFrame will be processed
        index_date_column : str
            Column name in index_date_df containing the index date
        additional_loinc_mappings : dict, optional
            Dictionary of additional lab names and their LOINC codes to add to the default mappings.
            Example: {'CEA': ['2039-6'], 'CA_15-3': ['6875-9'], 'CA_27-29': ['17842-6'], 'another_lab': ['12345', '67889-0']}
        days_before : int, optional
            Number of days before the index date to include for baseline lab values. Must be >= 0. Default: 90
        days_after : int, optional
            Number of days after the index date to include for baseline lab values. Also used as the end point for 
            summary statistics calculations. Must be >= 0. Default: 0
        summary_lookback : int, optional
            Number of days before index date to begin analyzing summary statistics. Analysis period extends 
            from (index_date - summary_lookback) to (index_date + days_after). Must be >= 0. Default: 180

        Returns
        -------
        pd.DataFrame or None
            - PatientID : object
                unique patient identifier

            Baseline values (closest to index date within days_before/days_after window):
            - hemoglobin : float, g/dL
            - wbc : float, K/uL
            - platelet : float, 10^9/L
            - creatinine : float, mg/dL
            - bun : float, mg/dL
            - sodium : float, mmol/L
            - chloride : float, mmol/L
            - bicarbonate : float, mmol/L
            - potassium : float, mmol/L
            - calcium : float, mg/dL
            - alp : float, U/L
            - ast : float, U/L
            - alt : float, U/L
            - total_bilirubin : float, mg/dL
            - albumin : float, g/L
            - psa : float, ug/L

            Summary statistics (calculated over period from index_date - summary_lookback to index_date + days_after):
            For each lab above, includes:
            - {lab}_max : float, maximum value
            - {lab}_min : float, minimum value
            - {lab}_std : float, standard deviation
            - {lab}_slope : float, rate of change over time (days)
            - psa_doubling_time : float, months

        Notes
        -----
        Data cleaning and processing: 
        - Imputation strategy for lab dates: missing ResultDate is imputed with TestDate
        - Imputation strategy for lab values:
            - For each lab, missing TestResultCleaned values are imputed from TestResult after removing flags (L, H, <, >)
            - Values outside physiological ranges for each lab are filtered out
        -Unit conversion corrections:
            - Hemoglobin: Values in g/uL are divided by 100,000 to convert to g/dL
            - WBC/Platelet: Values in 10*3/L are multiplied by 1,000,000; values in /mm3 or 10*3/mL are multiplied by 1,000
            - Creatinine/BUN/Calcium: Values in mg/L are multiplied by 10 to convert to mg/dL
            - Albumin: Values in mg/dL are multiplied by 1,000 to convert to g/L; values 1-6 are assumed to be g/dL and multiplied by 10
        - Lab value selection: 
            - Baseline lab value closest to index date is selected by minimum absolute day difference within window period of 
            (index_date - days_before) to (index_date + days_after)
            - Summary lab values are calculated within window period of (index_date - summary_lookback) to (index_date + days_after)
        For slope and PSA doubling time calculation:
            - Patient needs at least 2 valid measurements, at 2 valid time points, and time points must not be identical
            - PSA doubling formula: (ln(2)/PSA slope), measured in months
            - Where PSA slope = [ln(PSAMetDiagnosis) - ln(PSADiagnosis)]/[(MetDiagnosisDate - DiagnosisDate) in months]
        
        Output handling: 
        - All PatientIDs from index_date_df are included in the output and values are NaN for patients without lab values 
        - Duplicate PatientIDs are logged as warnings but retained in output 
        - Results are stored in self.labs_df attribute
        """
        # Input validation
        if not isinstance(index_date_df, pd.DataFrame):
            raise ValueError("index_date_df must be a pandas DataFrame")
        if 'PatientID' not in index_date_df.columns:
            raise ValueError("index_date_df must contain a 'PatientID' column")
        if not index_date_column or index_date_column not in index_date_df.columns:
            raise ValueError('index_date_column not found in index_date_df')
        if index_date_df['PatientID'].duplicated().any():
            raise ValueError("index_date_df contains duplicate PatientID values, which is not allowed")
        
        if not isinstance(days_before, int) or days_before < 0:
            raise ValueError("days_before must be a non-negative integer")
        if not isinstance(days_after, int) or days_after < 0:
            raise ValueError("days_after must be a non-negative integer")
        if not isinstance(summary_lookback, int) or summary_lookback < 0:
            raise ValueError("summary_lookback must be a non-negative integer")
        
        # Add user-provided mappings if they exist
        if additional_loinc_mappings is not None:
            if not isinstance(additional_loinc_mappings, dict):
                raise ValueError("Additional LOINC mappings must be provided as a dictionary")
            if not all(isinstance(v, list) for v in additional_loinc_mappings.values()):
                raise ValueError("LOINC codes must be provided as lists of strings")
                
            # Update the default mappings with additional ones
            self.LOINC_MAPPINGS.update(additional_loinc_mappings)

        try:
            df = pd.read_csv(file_path)
            logging.info(f"Successfully read Lab.csv file with shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")

            df['ResultDate'] = pd.to_datetime(df['ResultDate'])
            df['TestDate'] = pd.to_datetime(df['TestDate'])

            # Impute TestDate for missing ResultDate. 
            df['ResultDate'] = np.where(df['ResultDate'].isna(), df['TestDate'], df['ResultDate'])

            index_date_df[index_date_column] = pd.to_datetime(index_date_df[index_date_column])

            # Select PatientIDs that are included in the index_date_df the merge on 'left'
            df = df[df.PatientID.isin(index_date_df.PatientID)]
            df = pd.merge(
                df,
                index_date_df[['PatientID', index_date_column]],
                on = 'PatientID',
                how = 'left'
            )
            logging.info(f"Successfully merged Lab.csv df with index_date_df resulting in shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")
            
            # Flatten LOINC codes 
            all_loinc_codes = sum(self.LOINC_MAPPINGS.values(), [])

            # Filter for LOINC codes 
            df = df[df['LOINC'].isin(all_loinc_codes)]

            # Map LOINC codes to lab names
            for lab_name, loinc_codes in self.LOINC_MAPPINGS.items():
                mask = df['LOINC'].isin(loinc_codes)
                df.loc[mask, 'lab_name'] = lab_name

            ## CBC PROCESSING ##
            
            # Hemoglobin conversion correction
            # TestResultCleaned incorrectly stored g/uL values 
            # Example: 12 g/uL was stored as 1,200,000 g/dL instead of 12 g/dL
            # Need to divide by 100,000 to restore correct value
            mask = (
                (df['lab_name'] == 'hemoglobin') & 
                (df['TestUnits'] == 'g/uL')
            )
            df.loc[mask, 'TestResultCleaned'] = df.loc[mask, 'TestResultCleaned'] / 100000 

            # WBC and Platelet conversion correction
            # TestResultCleaned incorrectly stored 10*3/L values 
            # Example: 9 10*3/L was stored as 0.000009 10*9/L instead of 9 10*9/L
            # Need to multipley 1,000,000 to restore correct value
            mask = (
                ((df['lab_name'] == 'wbc') | (df['lab_name'] == 'platelet')) & 
                (df['TestUnits'] == '10*3/L')
            )
            df.loc[mask, 'TestResultCleaned'] = df.loc[mask, 'TestResultCleaned'] * 1000000

           # WBC and Platelet conversion correction
            # TestResultCleaned incorrectly stored /mm3 and 10*3/mL values
            # Example: 9 /mm3 and 9 10*3/mL was stored as 0.009 10*9/L instead of 9 10*9/L
            # Need to multipley 1,000 to restore correct value
            mask = (
                ((df['lab_name'] == 'wbc') | (df['lab_name'] == 'platelet')) & 
                ((df['TestUnits'] == '/mm3') | (df['TestUnits'] == '10*3/mL'))
            )
            df.loc[mask, 'TestResultCleaned'] = df.loc[mask, 'TestResultCleaned'] * 1000

            # Hemoglobin: Convert to TestResult to numeric after removing L, H, <, and >; filter for ranges from 3-20; and impute to TestResultCleaned
            mask = df.query('lab_name == "hemoglobin" and TestResultCleaned.isna() and TestResult.notna()').index
            df.loc[mask, 'TestResultCleaned'] = (
                pd.to_numeric(df.loc[mask, 'TestResult'].str.replace(r'[LH<>]', '', regex=True).str.strip(), errors='coerce')
                .where(lambda x: (x >= 3) & (x <= 20))
            )

            # WBC: Convert to TestResult to numeric after removing L, H, <, and >; filter for ranges from 0-40; and impute to TestResultCleaned
            mask = df.query('lab_name == "wbc" and TestResultCleaned.isna() and TestResult.notna()').index
            df.loc[mask, 'TestResultCleaned'] = (
                pd.to_numeric(df.loc[mask, 'TestResult'].str.replace(r'[LH<>]', '', regex=True).str.strip(), errors='coerce')
                .where(lambda x: (x >= 0) & (x <= 40))
            )
            
            # Platelet: Convert to TestResult to numeric after removing L, H, <, and >; filter for ranges from 0-1000; and impute to TestResultCleaned
            mask = df.query('lab_name == "platelet" and TestResultCleaned.isna() and TestResult.notna()').index
            df.loc[mask, 'TestResultCleaned'] = (
                pd.to_numeric(df.loc[mask, 'TestResult'].str.replace(r'[LH<>]', '', regex=True).str.strip(), errors='coerce')
                .where(lambda x: (x >= 0) & (x <= 1000))
            )

            ## CMP PROCESSING ##
            # Creatinine, BUN, and calcium conversion correction
            # TestResultCleaned incorrectly stored mg/L values 
            # Example: 1.6 mg/L was stored as 0.16 mg/dL instead of 1.6 mg/dL
            # Need to divide by 10 to restore correct value
            mask = (
                ((df['lab_name'] == 'creatinine') | (df['lab_name'] == 'bun') | (df['lab_name'] == 'calcium')) & 
                (df['TestUnits'] == 'mg/L')
            )
            df.loc[mask, 'TestResultCleaned'] = df.loc[mask, 'TestResultCleaned'] * 10 

            # Albumin conversion correction
            # TestResultCleaned incorrectly stored mg/dL values 
            # Example: 3.7 mg/dL was stored as 0.037 g/L instead of 37 g/L
            # Need to multiply 1000 to restore correct value
            mask = (
                (df['lab_name'] == 'albumin') & 
                (df['TestUnits'] == 'mg/dL')
            )
            df.loc[mask, 'TestResultCleaned'] = df.loc[mask, 'TestResultCleaned'] * 1000         

            # Creatinine: Convert to TestResult to numeric after removing L, H, <, and >; filter for ranges from 0-5; and impute to TestResultCleaned 
            mask = df.query('lab_name == "creatinine" and TestResultCleaned.isna() and TestResult.notna()').index
            df.loc[mask, 'TestResultCleaned'] = (
                pd.to_numeric(df.loc[mask, 'TestResult'].str.replace(r'[LH<>]', '', regex=True).str.strip(), errors='coerce')
                .where(lambda x: (x >= 0) & (x <= 5))
            )
            
            # BUN: Convert to TestResult to numeric after removing L, H, <, and >; filter for ranges from 0-100; and impute to TestResultCleaned 
            mask = df.query('lab_name == "bun" and TestResultCleaned.isna() and TestResult.notna()').index
            df.loc[mask, 'TestResultCleaned'] = (
                pd.to_numeric(df.loc[mask, 'TestResult'].str.replace(r'[LH<>]', '', regex=True).str.strip(), errors='coerce')
                .where(lambda x: (x >= 0) & (x <= 100))
            )

            # Sodium: Convert to TestResult to numeric after removing L, H, <, and >; filter for ranges from 110-160; and impute to TestResultCleaned 
            mask = df.query('lab_name == "sodium" and TestResultCleaned.isna() and TestResult.notna()').index
            df.loc[mask, 'TestResultCleaned'] = (
                pd.to_numeric(df.loc[mask, 'TestResult'].str.replace(r'[LH<>]', '', regex=True).str.strip(), errors='coerce')
                .where(lambda x: (x >= 110) & (x <= 160))
            )
            
            # Chloride: Convert to TestResult to numeric after removing L, H, <, and >; filter for ranges from 70-140; and impute to TestResultCleaned 
            mask = df.query('lab_name == "chloride" and TestResultCleaned.isna() and TestResult.notna()').index
            df.loc[mask, 'TestResultCleaned'] = (
                pd.to_numeric(df.loc[mask, 'TestResult'].str.replace(r'[LH<>]', '', regex=True).str.strip(), errors='coerce')
                .where(lambda x: (x >= 70) & (x <= 140))
            )
            
            # Bicarbonate: Convert to TestResult to numeric after removing L, H, <, and >; filter for ranges from 5-50; and impute to TestResultCleaned 
            mask = df.query('lab_name == "bicarbonate" and TestResultCleaned.isna() and TestResult.notna()').index
            df.loc[mask, 'TestResultCleaned'] = (
                pd.to_numeric(df.loc[mask, 'TestResult'].str.replace(r'[LH<>]', '', regex=True).str.strip(), errors='coerce')
                .where(lambda x: (x >= 5) & (x <= 50))
            )

            # Potassium: Convert to TestResult to numeric after removing L, H, <, and >; filter for ranges from 2-8; and impute to TestResultCleaned  
            mask = df.query('lab_name == "potassium" and TestResultCleaned.isna() and TestResult.notna()').index
            df.loc[mask, 'TestResultCleaned'] = (
                pd.to_numeric(df.loc[mask, 'TestResult'].str.replace(r'[LH<>]', '', regex=True).str.strip(), errors='coerce')
                .where(lambda x: (x >= 2) & (x <= 8))
            )
            
            # Calcium: Convert to TestResult to numeric after removing L, H, <, and >; filter for ranges from 5-15; and impute to TestResultCleaned 
            mask = df.query('lab_name == "calcium" and TestResultCleaned.isna() and TestResult.notna()').index
            df.loc[mask, 'TestResultCleaned'] = (
                pd.to_numeric(df.loc[mask, 'TestResult'].str.replace(r'[LH<>]', '', regex=True).str.strip(), errors='coerce')
                .where(lambda x: (x >= 5) & (x <= 15))
            )
            
            # ALP: Convert to TestResult to numeric after removing L, H, <, and >; filter for ranges from 20-3000; and impute to TestResultCleaned
            mask = df.query('lab_name == "alp" and TestResultCleaned.isna() and TestResult.notna()').index
            df.loc[mask, 'TestResultCleaned'] = (
                pd.to_numeric(df.loc[mask, 'TestResult'].str.replace(r'[LH<>]', '', regex=True).str.strip(), errors='coerce')
                .where(lambda x: (x >= 20) & (x <= 3000))
            )
            
            # AST: Convert to TestResult to numeric after removing L, H, <, and >; filter for ranges from 5-2000; and impute to TestResultCleaned
            mask = df.query('lab_name == "ast" and TestResultCleaned.isna() and TestResult.notna()').index
            df.loc[mask, 'TestResultCleaned'] = (
                pd.to_numeric(df.loc[mask, 'TestResult'].str.replace(r'[LH<>]', '', regex=True).str.strip(), errors='coerce')
                .where(lambda x: (x >= 5) & (x <= 2000))
            )
            
            # ALT: Convert to TestResult to numeric after removing L, H, <, and >; filter for ranges from 5-2000; and impute to TestResultCleaned
            mask = df.query('lab_name == "alt" and TestResultCleaned.isna() and TestResult.notna()').index
            df.loc[mask, 'TestResultCleaned'] = (
                pd.to_numeric(df.loc[mask, 'TestResult'].str.replace(r'[LH<>]', '', regex=True).str.strip(), errors='coerce')
                .where(lambda x: (x >= 5) & (x <= 2000))
            )
            
            # Total bilirubin: Convert to TestResult to numeric after removing L, H, <, and >; filter for ranges from 0-40; and impute to TestResultCleaned
            mask = df.query('lab_name == "total_bilirubin" and TestResultCleaned.isna() and TestResult.notna()').index
            df.loc[mask, 'TestResultCleaned'] = (
                pd.to_numeric(df.loc[mask, 'TestResult'].str.replace(r'[LH<>]', '', regex=True).str.strip(), errors='coerce')
                .where(lambda x: (x >= 0) & (x <= 40))
            )
            
            # Albumin
            mask = df.query('lab_name == "albumin" and TestResultCleaned.isna() and TestResult.notna()').index
            
            # First get the cleaned numeric values
            cleaned_alb_values = pd.to_numeric(df.loc[mask, 'TestResult'].str.replace(r'[LH<>]', '', regex=True).str.strip(), errors='coerce')

            # Identify which values are likely in which unit system
            # Values 1-6 are likely g/dL and need to be converted to g/L
            gdl_mask = (cleaned_alb_values >= 1) & (cleaned_alb_values <= 6)
            # Values 10-60 are likely already in g/L
            gl_mask = (cleaned_alb_values >= 10) & (cleaned_alb_values <= 60)

            # Convert g/dL values to g/L (multiply by 10)
            df.loc[mask[gdl_mask], 'TestResultCleaned'] = cleaned_alb_values[gdl_mask] * 10

            # Keep g/L values as they are
            df.loc[mask[gl_mask], 'TestResultCleaned'] = cleaned_alb_values[gl_mask]

            ## PSA PROCESSING ##
            # PSA conversion correction
            # TestResultCleaned incorrectly stored mg/dL values 
            # Example: 6.7 mg/dL was stored as 66,700 ug/L instead 6.7 ug/L
            # Need to divide by 1000 to restore correct value
            mask = (
                (df['lab_name'] == 'psa') & 
                (df['TestUnits'] == 'mg/dL')
            )
            df.loc[mask, 'TestResultCleaned'] = df.loc[mask, 'TestResultCleaned'] / 1000

            # PSA: Convert to TestResult to numeric after removing L, H, <, and >; filter for ranges from 0-1000; and impute to TestResultCleaned
            mask = df.query('lab_name == "psa" and TestResultCleaned.isna() and TestResult.notna()').index
            df.loc[mask, 'TestResultCleaned'] = (
                pd.to_numeric(df.loc[mask, 'TestResult'].str.replace(r'[LH<>]', '', regex=True).str.strip(), errors='coerce')
                .where(lambda x: (x >= 0) & (x <= 1000))
            ) 

            # Filter for desired window period for baseline labs after removing missing values after above imputation
            df = df.query('TestResultCleaned.notna()')
            df['index_to_lab'] = (df['ResultDate'] - df[index_date_column]).dt.days
            
            df_lab_index_filtered = df[
                (df['index_to_lab'] <= days_after) & 
                (df['index_to_lab'] >= -days_before)].copy()
            
            lab_df = (
                df_lab_index_filtered
                .assign(abs_index_to_lab = lambda x: abs(x['index_to_lab']))
                .sort_values('abs_index_to_lab')  
                .groupby(['PatientID', 'lab_name'])
                .first()  
                .reset_index()
                .pivot(index = 'PatientID', columns = 'lab_name', values = 'TestResultCleaned')
                .rename_axis(columns = None)
                .reset_index()
            )

            # Filter for desired window period for summary labs 
            df_lab_summary_filtered = df[
                (df['index_to_lab'] <= days_after) & 
                (df['index_to_lab'] >= -summary_lookback)].copy()
            
            max_df = (
                df_lab_summary_filtered
                .groupby(['PatientID', 'lab_name'])['TestResultCleaned'].max()
                .reset_index()
                .pivot(index = 'PatientID', columns = 'lab_name', values = 'TestResultCleaned')
                .rename_axis(columns = None)
                .rename(columns = lambda x: f'{x}_max')
                .reset_index()
            )
            
            min_df = (
                df_lab_summary_filtered
                .groupby(['PatientID', 'lab_name'])['TestResultCleaned'].min()
                .reset_index()
                .pivot(index = 'PatientID', columns = 'lab_name', values = 'TestResultCleaned')
                .rename_axis(columns = None)
                .rename(columns = lambda x: f'{x}_min')
                .reset_index()
            )
            
            std_df = (
                df_lab_summary_filtered
                .groupby(['PatientID', 'lab_name'])['TestResultCleaned'].std()
                .reset_index()
                .pivot(index = 'PatientID', columns = 'lab_name', values = 'TestResultCleaned')
                .rename_axis(columns = None)
                .rename(columns = lambda x: f'{x}_std')
                .reset_index()
            )
            
            slope_df = (
                df_lab_summary_filtered
                .groupby(['PatientID', 'lab_name'])[['index_to_lab', 'TestResultCleaned']]
                .apply(lambda x: np.polyfit(x['index_to_lab'],
                                            x['TestResultCleaned'],
                                            1)[0]                       # Extract slope coefficient with [0]
                    if (x['TestResultCleaned'].notna().sum() > 1 and    # Need at least 2 valid measurements
                        x['index_to_lab'].notna().sum() > 1 and         # Need at least 2 valid time points
                        len(x['index_to_lab'].unique()) > 1)            # Time points must not be identical
                    else np.nan)                                        # Return NaN if conditions for valid slope calculation aren't met
                .reset_index()
                .pivot(index = 'PatientID', columns = 'lab_name', values = 0)
                .rename_axis(columns = None)
                .rename(columns = lambda x: f'{x}_slope')
                .reset_index()
            )

            psa_doubling_df = (
                df_lab_summary_filtered.query('lab_name == "psa"')
                .groupby('PatientID')[['index_to_lab', 'TestResultCleaned']]
                .apply(lambda x: 
                    # Inner lambda: calculates slope, once passes data check
                    (lambda slope = np.polyfit(
                        x['index_to_lab']/30,                                       # /30 to convert days to months 
                        np.log(x['TestResultCleaned']),                             # log transform PSA values    
                        1)[0]:                                                      # Extract slope coefficient with [0]
                        
                        # Ceck if slope is positive before calc doubling time
                        math.log(2)/slope if slope > 0 else np.nan)()               # Return NaN if slope is negative or zero

                    # Outer lambda: data quality check     
                    if (x['TestResultCleaned'].notna().sum() > 1 and                # Need at least 2 valid measurements
                        x['index_to_lab'].notna().sum() > 1 and                     # Need at least 2 valid time points
                        len(x['index_to_lab'].unique()) > 1)                        # Time points must not be identical
                    else np.nan)
                .reset_index()
                .rename(columns={0: 'psa_doubling'})
            )
            
            # Merge dataframes - start with index_date_df to ensure all PatientIDs are included
            final_df = index_date_df[['PatientID']].copy()
            final_df = pd.merge(final_df, lab_df, on = 'PatientID', how = 'left')
            final_df = pd.merge(final_df, max_df, on = 'PatientID', how = 'left')
            final_df = pd.merge(final_df, min_df, on = 'PatientID', how = 'left')
            final_df = pd.merge(final_df, std_df, on = 'PatientID', how = 'left')
            final_df = pd.merge(final_df, slope_df, on = 'PatientID', how = 'left')
            final_df = pd.merge(final_df, psa_doubling_df, on = 'PatientID', how = 'left')

            # Check for duplicate PatientIDs
            if len(final_df) > final_df['PatientID'].nunique():
                duplicate_ids = final_df[final_df.duplicated(subset = ['PatientID'], keep = False)]['PatientID'].unique()
                logging.warning(f"Duplicate PatientIDs found: {duplicate_ids}")

            logging.info(f"Successfully processed Lab.csv file with final shape: {final_df.shape} and unique PatientIDs: {(final_df['PatientID'].nunique())}")
            self.labs_df = final_df
            return final_df

        except Exception as e:
            logging.error(f"Error processing Lab.csv file: {e}")
            return None