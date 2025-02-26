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
        'platelet': ['26515-7', '777-3', '778-1', '49497-1'],
        'creatinine': ['2160-0', '38483-4'],
        'bun': ['3094-0'],
        'sodium': ['2947-0', '2951-2'],
        'bicarb': ['1963-8', '1959-6', '14627-4', '1960-4', '2028-9'],
        'chloride': ['2075-0'],
        'potassium': ['6298-4', '2823-3'],
        'albumin': ['1751-7'],
        'calcium': ['17861-6', '49765-1'],
        'total_bilirubin': ['42719-5', '1975-2'],
        'ast': ['1920-8'],
        'alt': ['1742-6', '1743-4', '1744-2'],
        'alp': ['6768-6']
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
        Duplicate PatientIDs are logged as warnings if found but retained in output
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
                duplicate_ids = df[df.duplicated(subset = ['PatientID'], keep = False)]['PatientID'].unique()
                logging.warning(f"Duplicate PatientIDs found: {duplicate_ids}")

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
        Duplicate PatientIDs are logged as warnings if found but retained in output
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
        Processes Enhanced_AdvNSCLCBiomarkers.csv by determining biomarker status for each patient within a specified time window relative to an index date. 

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
            - EGFR_status : category
                positive if ever-positive, negative if only-negative, otherwise unknown
            - KRAS_status : cateogory
                positive if ever-positive, negative if only-negative, otherwise unknown
            - BRAF_status : cateogory
                positive if ever-positive, negative if only-negative, otherwise unknown
            - ALK_status : cateogory
                positive if ever-positive, negative if only-negative, otherwise unknown
            - ROS1_status : cateogory
                positive if ever-positive, negative if only-negative, otherwise unknown
            - MET_status : category
                positive if ever-positive, negative if only-negative, otherwise unknown
            - RET_status : category
                positive if ever-positive, negative if only-negative, otherwise unknown
            - NTRK_status : category
                positive if ever-positive, negative if only-negative, otherwise unknown
            - PDL1_status : cateogry 
                positive if ever-positive, negative if only-negative, otherwise unknown
            - PDL1_percent_staining : category, ordered 
                returns a patient's maximum percent staining for PDL1

        Notes
        ------
        Missing ResultDate is imputed with SpecimenReceivedDate.
        All PatientIDs from index_date_df are included in the output and values will be NaN for patients without any biomarker tests
        NTRK genes (NTRK1, NTRK2, NTRK3, NTRK - other, NTRK - unknown) are grouped given that gene type does not impact treatment decisions
        For each biomarker, status is classified as:
            - 'positive' if any test result is positive (ever-positive)
            - 'negative' if any test is negative without positives (only-negative) 
            - 'unknown' if all results are indeterminate
        Duplicate PatientIDs are logged as warnings if found but retained in output
        Processed DataFrame is stored in self.biomarkers_df
        """
        # Input validation
        if not isinstance(index_date_df, pd.DataFrame):
            raise ValueError("index_date_df must be a pandas DataFrame")
        if 'PatientID' not in index_date_df.columns:
            raise ValueError("index_date_df must contain a 'PatientID' column")
        if not index_date_column or index_date_column not in index_date_df.columns:
            raise ValueError(f"Column '{index_date_column}' not found in index_date_df")
        
        if days_before is not None:
            if not isinstance(days_before, int) or days_before < 0:
                raise ValueError("days_before must be a non-negative integer or None")
        if not isinstance(days_after, int) or days_after < 0:
            raise ValueError("days_after must be a non-negative integer")

        try:
            df = pd.read_csv(file_path)
            logging.info(f"Successfully read Enhanced_AdvNSCLCBiomarkers.csv file with shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")

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
            logging.info(f"Successfully merged Enhanced_AdvNSCLCBiomarkers.csv df with index_date_df resulting in shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")
            
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

            # Group NTRK genes
            df_filtered['BiomarkerName'] = (
                np.where(df_filtered['BiomarkerName'].isin(["NTRK1", "NTRK2", "NTRK3", "NTRK - unknown gene type","NTRK - other"]),
                        "NTRK",
                        df_filtered['BiomarkerName'])
            )

            # Create an empty dictionary to store the dataframes
            biomarker_dfs = {}

            # Process EGFR, KRAS, and BRAF 
            for biomarker in ['EGFR', 'KRAS', 'BRAF']:
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
                
            # Process ALK and ROS1
            for biomarker in ['ALK', 'ROS1']:
                biomarker_dfs[biomarker] = (
                    df_filtered
                    .query(f'BiomarkerName == "{biomarker}"')
                    .groupby('PatientID')['BiomarkerStatus']
                    .agg(lambda x: 'positive' if any('Rearrangement present' in val for val in x)
                        else ('negative' if any('Rearrangement not present' in val for val in x)
                            else 'unknown'))
                    .reset_index()
                    .rename(columns={'BiomarkerStatus': f'{biomarker}_status'})  # Rename for clarity
            )
                
            # Process MET, RET, and NTRK
            positive_values = {
                "Protein expression positive",
                "Mutation positive",
                "Amplification positive",
                "Rearrangement positive",
                "Other result type positive",
                "Unknown result type positive"
            }

            for biomarker in ['MET', 'RET', 'NTRK']:
                biomarker_dfs[biomarker] = (
                    df_filtered
                    .query(f'BiomarkerName == "{biomarker}"')
                    .groupby('PatientID')['BiomarkerStatus']
                    .agg(lambda x: 'positive' if any(val in positive_values for val in x)
                        else ('negative' if any('Negative' in val for val in x)
                            else 'unknown'))
                    .reset_index()
                    .rename(columns={'BiomarkerStatus': f'{biomarker}_status'})  # Rename for clarity
            )
            
            # Process PDL1 and add to biomarker_dfs
            biomarker_dfs['PDL1'] = (
                df_filtered
                .query('BiomarkerName == "PDL1"')
                .groupby('PatientID')['BiomarkerStatus']
                .agg(lambda x: 'positive' if any ('PD-L1 positive' in val for val in x)
                    else ('negative' if any('PD-L1 negative/not detected' in val for val in x)
                        else 'unknown'))
                .reset_index()
                .rename(columns={'BiomarkerStatus': 'PDL1_status'})
            )

            # Process PDL1 percent staining 
            PDL1_staining_df = (
                df_filtered
                .query('BiomarkerName == "PDL1"')
                .query('BiomarkerStatus == "PD-L1 positive"')
                .groupby('PatientID')['PercentStaining']
                .apply(lambda x: x.map(self.PDL1_PERCENT_STAINING_MAPPING))
                .groupby('PatientID')
                .agg('max')
                .to_frame(name = 'PDL1_ordinal_value')
                .reset_index()
            )
            
            # Create reverse mapping to convert back to percentage strings
            reverse_pdl1_dict = {v: k for k, v in self.PDL1_PERCENT_STAINING_MAPPING.items()}
            PDL1_staining_df['PDL1_percent_staining'] = PDL1_staining_df['PDL1_ordinal_value'].map(reverse_pdl1_dict)
            PDL1_staining_df = PDL1_staining_df.drop(columns = ['PDL1_ordinal_value'])

            # Merge dataframes -- start with index_date_df to ensure all PatientIDs are included
            final_df = index_date_df[['PatientID']].copy()

            for biomarker in ['EGFR', 'KRAS', 'BRAF', 'ALK', 'ROS1', 'MET', 'RET', 'NTRK', 'PDL1']:
                final_df = pd.merge(final_df, biomarker_dfs[biomarker], on = 'PatientID', how = 'left')

            final_df = pd.merge(final_df, PDL1_staining_df, on = 'PatientID', how = 'left')

            # Convert to category type
            for biomarker_status in ['EGFR_status', 'KRAS_status', 'BRAF_status', 'ALK_status', 'ROS1_status', 'MET_status', 'RET_status', 'NTRK_status', 'PDL1_status']:
                final_df[biomarker_status] = final_df[biomarker_status].astype('category')

            staining_dtype = pd.CategoricalDtype(
                categories = ['0%', '< 1%', '1%', '2% - 4%', '5% - 9%', '10% - 19%',
                                '20% - 29%', '30% - 39%', '40% - 49%', '50% - 59%',
                                '60% - 69%', '70% - 79%', '80% - 89%', '90% - 99%', '100%'],
                                ordered = True
            )
            
            final_df['PDL1_percent_staining'] = final_df['PDL1_percent_staining'].astype(staining_dtype)

           # Check for duplicate PatientIDs
            if len(final_df) > final_df['PatientID'].nunique():
                duplicate_ids = final_df[final_df.duplicated(subset = ['PatientID'], keep = False)]['PatientID'].unique()
                logging.warning(f"Duplicate PatientIDs found: {duplicate_ids}")

            logging.info(f"Successfully processed Enhanced_AdvNSCLCBiomarkers.csv file with final shape: {final_df.shape} and unique PatientIDs: {(final_df['PatientID'].nunique())}")
            self.biomarkers_df = final_df
            return final_df

        except Exception as e:
            logging.error(f"Error processing Enhanced_AdvNSCLCBiomarkers.csv file: {e}")
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
        Duplicate PatientIDs are logged as warnings if found but retained in output
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
                       abnormal_reading_threshold: int = 2) -> pd.DataFrame:
        """
        Processes Vitals.csv to determine patient BMI, weight, change in weight, and vital sign abnormalities
        within a specified time window relative to an index date. Two different time windows are used for distinct 
        clinical purposes:
        
        1. A smaller window near the index date to find weight and BMI at that time point
        2. A larger lookback window to detect clinically significant vital sign abnormalities 
        suggesting possible deterioration

        Parameters
        ----------
        file_path : str
            Path to Vitals.csv file
        index_date_df : pd.DataFrame
            DataFrame containing PatientID and index dates. Only vitals for PatientIDs present in this DataFrame will be processed
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
        pd.DataFrame
            - PatientID : object 
                unique patient identifier
            - weight : float
                weight in kg closest to index date within specified window (index_date - weight_days_before) to (index_date + weight_days_after)
            - bmi : float
                BMI closest to index date within specified window (index_date - weight_days_before) to (index_date + days_after)
            - percent_change_weight : float
                percentage change in weight over period from (index_date - vital_summary_lookback) to (index_date + days_after)
            - hypotension : Int64
                binary indicator (0/1) for systolic blood pressure <90 mmHg on ≥{abnormal_reading_threshold} separate readings between (index_date - vital_summary_lookback) and (index_date + days_after)
            - tachycardia : Int64
                binary indicator (0/1) for heart rate >100 bpm on ≥{abnormal_reading_threshold} separate readings between (index_date - vital_summary_lookback) and (index_date + days_after)
            - fevers : Int64
                binary indicator (0/1) for temperature >=38°C on ≥{abnormal_reading_threshold} separate readings between (index_date - vital_summary_lookback) and (index_date + days_after)
            - hypoxemia : Int64
                binary indicator (0/1) for SpO2 <=88% on ≥{abnormal_reading_threshold} separate readings between (index_date - vital_summary_lookback) and (index_date + days_after)

        Notes
        -----
        Missing TestResultCleaned values are imputed using TestResult. For those where units are ambiguous, unit conversion is based on thresholds:
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
        
        BMI is calculated using weight closest to index date within specified window while height outside the specified window may be used. The equation used: weight (kg)/height (m)^2
        BMI calucalted as <13 are considered implausible and removed
        Percent change in weight is calculated as ((end_weight - start_weight) / start_weight) * 100
        TestDate rather than ResultDate is used since TestDate is always populated and, for vital signs, the measurement date (TestDate) and result date (ResultDate) should be identical since vitals are recorded in real-time
        All PatientIDs from index_date_df are included in the output and values will be NaN for patients without weight, BMI, or percent_change_weight, but set to 0 for hypotension, tachycardia, and fevers
        Duplicate PatientIDs are logged as warnings but retained in output
        Results are stored in self.vitals_df attribute
        """
        # Input validation
        if not isinstance(index_date_df, pd.DataFrame):
            raise ValueError("index_date_df must be a pandas DataFrame")
        if 'PatientID' not in index_date_df.columns:
            raise ValueError("index_date_df must contain a 'PatientID' column")
        if not index_date_column or index_date_column not in index_date_df.columns:
            raise ValueError(f"Column '{index_date_column}' not found in index_date_df")
        
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
            
            # Remove all rows with missing TestResult
            df = df.query('TestResult.notna()')

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
                .rename(columns = {'TestResultCleaned': 'weight'})
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
            has_both_measures = weight_index_df['weight'].notna() & weight_index_df['height'].notna()
            
            # Only calculate BMI where both measurements exist
            weight_index_df.loc[has_both_measures, 'bmi'] = (
                weight_index_df.loc[has_both_measures, 'weight'] / 
                weight_index_df.loc[has_both_measures, 'height']**2
            )

            # Replace implausible BMI values with NaN
            implausible_bmi = weight_index_df['bmi'] < 13
            weight_index_df.loc[implausible_bmi, 'bmi'] = np.nan
                    
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
                    'TestResultCleaned': lambda x: sum(x <= 88) >= abnormal_reading_threshold 
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
                          missing_date_strategy: str = 'conservative') -> pd.DataFrame:
        """
        Processes insurance data to identify insurance coverage relative to a specified index date.
        Insurance types are grouped into four categories: Medicare, Medicaid, Commercial, and Other Insurance. 
        
        Parameters
        ----------
        file_path : str
            Path to Insurance.csv file
        index_date_df : pd.DataFrame
            DataFrame containing PatientID and index dates. Only insurances for PatientIDs present in this DataFrame will be processed
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
        pd.DataFrame
            - PatientID : object
                unique patient identifier
            - medicare : Int64
                binary indicator (0/1) for Medicare coverage
            - medicaid : Int64
                binary indicator (0/1) for Medicaid coverage
            - commercial : Int64
                binary indicator (0/1) for commercial insuarnce coverage
            - other_insurance : Int64
                binaroy indicator (0/1) for other insurance types (eg., other payer, other government program, patient assistance program, self pay, and workers compensation)

        Notes
        -----
        Insurance is considered active if:
        1. StartDate falls before or during the specified time window AND
        2. Either:
            - EndDate is missing (considered still active) OR
            - EndDate falls on or after the start of the time window 

        Insurance categorization logic:
        1. Medicaid takes priority over Medicare for dual-eligible patients
        2. Records are classified as Medicare if:
        - PayerCategory is 'Medicare' OR
        - PayerCategory is not 'Medicaid' AND IsMedicareAdv is 'Yes' AND IsManagedMedicaid is not 'Yes' AND IsMedicareMedicaid is not 'Yes' OR
        - PayerCategory is not 'Medicaid' AND IsMedicareSupp is 'Yes' AND IsManagedMedicaid is not 'Yes' AND IsMedicareMedicaid is not 'Yes'
        3. Records are classified as Medicaid if:
        - PayerCategory is 'Medicaid' OR
        - IsManagedMedicaid is 'Yes' OR
        - IsMedicareMedicaid is 'Yes'
        4. Records are classified as Commercial if PayerCategory is 'Commercial Health Plan' after above reclassification
        5. All other records are classified as Other Insurance

        EndDate is missing for most patients
        All PatientIDs from index_date_df are included in the output
        Duplicate PatientIDs are logged as warnings but retained in output
        Results are stored in self.insurance_df attribute
        """
        # Input validation
        if not isinstance(index_date_df, pd.DataFrame):
            raise ValueError("index_date_df must be a pandas DataFrame")
        if 'PatientID' not in index_date_df.columns:
            raise ValueError("index_date_df must contain a 'PatientID' column")
        if not index_date_column or index_date_column not in index_date_df.columns:
            raise ValueError(f"Column '{index_date_column}' not found in index_date_df")
        
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

            # Reclassify Commerical Health Plans that should be Medicare or Medicaid
            # Identify Medicare Advantage plans
            df['PayerCategory'] = np.where((df['PayerCategory'] != 'Medicaid') & (df['IsMedicareAdv'] == 'Yes') & (df['IsManagedMedicaid'] != 'Yes') & (df['IsMedicareMedicaid'] != 'Yes'),
                                           'Medicare',
                                           df['PayerCategory'])

            # Identify Medicare Supplement plans incorrectly labeled as Commercial
            df['PayerCategory'] = np.where((df['PayerCategory'] != 'Medicaid') & (df['IsMedicareSupp'] == 'Yes') & (df['IsManagedMedicaid'] != 'Yes') & (df['IsMedicareMedicaid'] != 'Yes'),
                                           'Medicare',
                                           df['PayerCategory'])
            
            # Identify Managed Medicaid plans incorrectly labeled as Commercial
            df['PayerCategory'] = np.where((df['IsManagedMedicaid'] == 'Yes'),
                                           'Medicaid',
                                           df['PayerCategory'])
            
            # Identify Medicare-Medicaid dual eligible plans incorrectly labeled as Commercial
            df['PayerCategory'] = np.where((df['IsMedicareMedicaid'] == 'Yes'),
                                           'Medicaid',
                                           df['PayerCategory'])

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

            # Merger index_date_df to ensure all PatientIDs are included
            final_df = pd.merge(index_date_df[['PatientID']], final_df, on = 'PatientID', how = 'left')
            
            insurance_columns = list(set(self.INSURANCE_MAPPING.values()))
            for col in insurance_columns:
                final_df[col] = final_df[col].fillna(0).astype('Int64')

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
                     summary_lookback: int = 180) -> pd.DataFrame:
        """
        Processes Lab.csv to determine patient lab values within a specified time window relative to an index date. Returns CBC and CMP values 
        nearest to index date, along with summary statistics (max, min, standard deviation, and slope) calculated over the summary period. 
        Additional lab tests can be included by providing corresponding LOINC code mappings.

        Parameters
        ----------
        file_path : str
            Path to Labs.csv file
        index_date_df : pd.DataFrame
            DataFrame containing PatientID and index dates. Only labs for PatientIDs present in this DataFrame will be processed
        index_date_column : str
            Column name in index_date_df containing the index date
        additional_loinc_mappings : dict, optional
            Dictionary of additional lab names and their LOINC codes to add to the default mappings.
            Example: {'new_lab': ['1234-5'], 'another_lab': ['6789-0', '9876-5']}
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
        pd.DataFrame
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

            Summary statistics (calculated over period from index_date - summary_lookback to index_date + days_after):
            For each lab above, includes:
            - {lab}_max : float, maximum value
            - {lab}_min : float, minimum value
            - {lab}_std : float, standard deviation
            - {lab}_slope : float, rate of change over time

        Notes
        -----
        Missing ResultDate is imputed with TestDate
        
        Imputation strategy for lab values:
        - For each lab, missing TestResultCleaned values are imputed from TestResult after removing flags (L, H, <, >)
        - Values outside physiological ranges for each lab are filtered out

        Unit conversion corrections:
        - Hemoglobin: Values in g/uL are divided by 100,000 to convert to g/dL
        - WBC/Platelet: Values in 10*3/L are multiplied by 1,000,000; values in /mm3 or 10*3/mL are multiplied by 1,000
        - Creatinine/BUN/Calcium: Values in mg/L are multiplied by 10 to convert to mg/dL
        - Albumin: Values in mg/dL are multiplied by 100 to convert to g/L; values 1-6 are assumed to be g/dL and multiplied by 10
        
        All PatientIDs from index_date_df are included in the output and values will be NaN for patients without lab values 
        Duplicate PatientIDs are logged as warnings but retained in output 
        Results are stored in self.labs_df attribute
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
            
            # Platelet: Convert to TestResult to numeric after removing L, H, <, and >; filter for ranges from 0-750; and impute to TestResultCleaned
            mask = df.query('lab_name == "platelet" and TestResultCleaned.isna() and TestResult.notna()').index
            df.loc[mask, 'TestResultCleaned'] = (
                pd.to_numeric(df.loc[mask, 'TestResult'].str.replace(r'[LH<>]', '', regex=True).str.strip(), errors='coerce')
                .where(lambda x: (x >= 0) & (x <= 750))
            )
            
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

            ## CMP PROCESSING ##

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
            # Need to multiply 100 to restore correct value
            mask = (
                (df['lab_name'] == 'albumin') & 
                (df['TestUnits'] == 'mg/dL')
            )
            df.loc[mask, 'TestResultCleaned'] = df.loc[mask, 'TestResultCleaned'] * 100         

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
                df_lab_index_filtered
                .groupby(['PatientID', 'lab_name'])[['index_to_lab', 'TestResultCleaned']]
                .apply(lambda x: np.polyfit(x['index_to_lab'],
                                        x['TestResultCleaned'],
                                        1)[0] # Extract slope coefficient from linear fit
                    if (x['TestResultCleaned'].notna().sum() > 1 and # Need at least 2 valid measurements
                        x['index_to_lab'].notna().sum() > 1 and      # Need at least 2 valid time points
                        len(x['index_to_lab'].unique()) > 1)         # Time points must not be identical
                    else np.nan) # Return NaN if conditions for valid slope calculation aren't met
                .reset_index()
                .pivot(index = 'PatientID', columns = 'lab_name', values = 0)
                .rename_axis(columns = None)
                .rename(columns = lambda x: f'{x}_slope')
                .reset_index()
            )
            
            # Merge dataframes - start with index_date_df to ensure all PatientIDs are included
            final_df = index_date_df[['PatientID']].copy()
            final_df = pd.merge(final_df, lab_df, on = 'PatientID', how = 'left')
            final_df = pd.merge(final_df, max_df, on = 'PatientID', how = 'left')
            final_df = pd.merge(final_df, min_df, on = 'PatientID', how = 'left')
            final_df = pd.merge(final_df, std_df, on = 'PatientID', how = 'left')
            final_df = pd.merge(final_df, slope_df, on = 'PatientID', how = 'left')

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