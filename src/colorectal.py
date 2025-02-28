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
        self.practice_df = None
        self.biomarkers_df = None 
        self.ecog_df = None

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
        # Input validation
        if patient_ids is not None:
            if not isinstance(patient_ids, list):
                raise TypeError("patient_ids must be a list or None")
        
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
            self.enhanced_df = df
            return df

        except Exception as e:
            logging.error(f"Error processing Enhanced_MetastaticCRC.csv file: {e}")
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
                age at index date (index year - birth year)
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
        Duplicate PatientIDs are logged as warnings if found but retained in output
        Processed DataFrame is stored in self.demographics_df
        """
        # Input validation
        if not isinstance(index_date_df, pd.DataFrame):
            raise ValueError("index_date_df must be a pandas DataFrame")
        if 'PatientID' not in index_date_df.columns:
            raise ValueError("index_date_df must contain a 'PatientID' column")
        if not index_date_column or index_date_column not in index_date_df.columns:
            raise ValueError('index_date_column not found in index_date_df')

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
        Duplicate PatientIDs are logged as warnings if found but retained in output
        Processed DataFrame is stored in self.practice_df
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
                           days_after: int = 0) -> pd.DataFrame:
        """
        Processes Enhanced_MetCRCCBiomarkers.csv by determining biomarker status for each patient within a specified time window relative to an index date. 

        Parameters
        ----------
        file_path : str
            Path to Enhanced_AdvNSCLCBiomarkers.csv file
        index_date_df : pd.DataFrame
            DataFrame containing PatientID and index dates. Only biomarkers for PatientIDs present in this DataFrame will be processed
        index_date_column : str
            Column name in index_date_df containing the index date
        days_before : int | None, optional
            Number of days before the index date to include. Must be >= 0 or None. If None, includes all prior results. Default: None
        days_after : int, optional
            Number of days after the index date to include. Must be >= 0. Default: 0
        
        Returns
        -------
        pd.DataFrame
            - PatientID : object
                unique patient identifier
            - BRAF_status : category
                positive if ever-positive, negative if only-negative, otherwise unknown
            - KRAS_status : cateogory
                positive if ever-positive, negative if only-negative, otherwise unknown
            - NRAS_status : cateogory
                positive if ever-positive, negative if only-negative, otherwise unknown
            - MMR/MSI_status : cateogory
                positive if ever-positive, negative if only-negative, otherwise unknown

        Notes
        ------
        For each biomarker, status is classified as:
            - 'positive' if any test result is positive (ever-positive)
            - 'negative' if any test is negative without positives (only-negative) 
            - 'unknown' if all results are indeterminate
        MMR/MSI classification:
            - 'positive' includes: 'Loss of MMR protein expression (MMR protein deficiency found)' and 'MSI-H'
            - 'negative' includes: 'Normal MMR protein expression', 'MSS', and 'MSI-L'
            - MSI-L is classified as negative because these patients typically do not receive checkpoint 
              inhibitors like MSI-H patients and are clinically managed more similarly to MSS patients
        
        Missing ResultDate is imputed with SpecimenReceivedDate.
        All PatientIDs from index_date_df are included in the output and values will be NaN for patients without any biomarker tests
        Duplicate PatientIDs are logged as warnings if found but retained in output
        Processed DataFrame is stored in self.biomarkers_df
        """
        # Input validation
        if not isinstance(index_date_df, pd.DataFrame):
            raise ValueError("index_date_df must be a pandas DataFrame")
        if 'PatientID' not in index_date_df.columns:
            raise ValueError("index_date_df must contain a 'PatientID' column")
        if not index_date_column or index_date_column not in index_date_df.columns:
            raise ValueError('index_date_column not found in index_date_df')
        
        if days_before is not None:
            if not isinstance(days_before, int) or days_before < 0:
                raise ValueError("days_before must be a non-negative integer or None")
        if not isinstance(days_after, int) or days_after < 0:
            raise ValueError("days_after must be a non-negative integer")

        try:
            df = pd.read_csv(file_path)
            logging.info(f"Successfully read Enhanced_MetCRCCBiomarkers.csv file with shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")

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
            logging.info(f"Successfully merged Enhanced_MetCRCCBiomarkers.csv df with index_date_df resulting in shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")
            
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

            # Create an empty dictionary to store the dataframes
            biomarker_dfs = {}

            # Process BRAF, KRAS, and NRAS
            for biomarker in ['BRAF', 'KRAS', 'NRAS']:
                biomarker_dfs[biomarker] = (
                    df_filtered
                    .query(f'BiomarkerName == "{biomarker}"')
                    .groupby('PatientID')['BiomarkerStatus']
                    .agg(lambda x: 'positive' if any('Mutation positive' in val for val in x)
                        else ('negative' if any('Mutation negative' in val for val in x)
                            else 'unknown'))
                    .reset_index()
                    .rename(columns={'BiomarkerStatus': f'{biomarker}_status'})  # Rename for clarity
            )
                
            # Process MMR/MSI
            positive_values = {
                'Loss of MMR protein expression (MMR protein deficiency found)',
                'MSI-H' 
            }

            negative_values = {
                'Normal MMR protein expression (No loss of nuclear expression of MMR protein)',
                'MSS',
                'MSI-L'
            }

            for biomarker in ['MMR/MSI']:
                biomarker_dfs[biomarker] = (
                    df_filtered
                    .query(f'BiomarkerName == "{biomarker}"')
                    .groupby('PatientID')['BiomarkerStatus']
                    .agg(lambda x: 'positive' if any(val in positive_values for val in x)
                        else ('negative' if any(val in negative_values for val in x)
                            else 'unknown'))
                    .reset_index()
                    .rename(columns={'BiomarkerStatus': f'{biomarker}_status'})  # Rename for clarity
            )

            # Merge dataframes -- start with index_date_df to ensure all PatientIDs are included
            final_df = index_date_df[['PatientID']].copy()

            for biomarker in ['BRAF', 'KRAS', 'NRAS', 'MMR/MSI']:
                final_df = pd.merge(final_df, biomarker_dfs[biomarker], on = 'PatientID', how = 'left')

            # Convert to category type
            for biomarker_status in ['BRAF_status', 'KRAS_status', 'NRAS_status', 'MMR/MSI_status']:
                final_df[biomarker_status] = final_df[biomarker_status].astype('category')

            # Check for duplicate PatientIDs
            if len(final_df) > final_df['PatientID'].nunique():
                duplicate_ids = final_df[final_df.duplicated(subset = ['PatientID'], keep = False)]['PatientID'].unique()
                logging.warning(f"Duplicate PatientIDs found: {duplicate_ids}")

            logging.info(f"Successfully processed Enhanced_MetCRCCBiomarkers.csv file with final shape: {final_df.shape} and unique PatientIDs: {(final_df['PatientID'].nunique())}")
            self.biomarkers_df = final_df
            return final_df

        except Exception as e:
            logging.error(f"Error processing Enhanced_MetCRCCBiomarkers.csv file: {e}")
            return None
    def process_ecog(self, 
                     file_path: str,
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
            Number of days before index date to look for ECOG progression (0-1 to ≥2). Must be >= 0. Consider
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
        Duplicate PatientIDs are logged as warnings if found but retained in output
        Processed DataFrame is stored in self.ecog_df
        """
        # Input validation
        if not isinstance(index_date_df, pd.DataFrame):
            raise ValueError("index_date_df must be a pandas DataFrame")
        if 'PatientID' not in index_date_df.columns:
            raise ValueError("index_date_df must contain a 'PatientID' column")
        if not index_date_column or index_date_column not in index_date_df.columns:
            raise ValueError('index_date_column not found in index_date_df')
        
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
                duplicate_ids = final_df[final_df.duplicated(subset = ['PatientID'], keep = False)]['PatientID'].unique()
                logging.warning(f"Duplicate PatientIDs found: {duplicate_ids}")
                
            logging.info(f"Successfully processed ECOG.csv file with final shape: {final_df.shape} and unique PatientIDs: {(final_df['PatientID'].nunique())}")
            self.ecog_df = final_df
            return final_df

        except Exception as e:
            logging.error(f"Error processing ECOG.csv file: {e}")
            return None