import pandas as pd
import numpy as np
import logging
import re 
from typing import Optional

logging.basicConfig(
    level = logging.INFO,                                 
    format = '%(asctime)s - %(levelname)s - %(message)s'  
)

class DataProcessorBreast: 

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

    def __init__(self):
        self.enhanced_df = None
        self.demographics_df = None
        self.practice_df = None
        self.biomarkers_df = None
        self.ecog_df = None
        self.vitals_df = None
        self.insurance_df = None 
    
    def process_enhanced(self,
                         file_path: str,
                         patient_ids: list = None,
                         drop_dates: bool = True) -> pd.DataFrame: 
        """
        Processes Enhanced_MetastaticBreast.csv to standardize categories, consolidate staging information, and calculate time-based metrics between key clinical events.

        Parameters
        ----------
        file_path : str
            Path to Enhanced_MetastaticBreast.csv file
        patient_ids : list, optional
            List of specific PatientIDs to process. If None, processes all patients
        drop_dates : bool, default=True
            If True, drops date columns (DiagnosisDate and MetDiagnosisDate) after calculating durations

        Returns
        -------
        pd.DataFrame
            - PatientID : object
                unique patient identifier
            - GroupStage : category
                stage at time of first diagnosis
            - days_diagnosis_to_met : float
                days from first diagnosis to metastatic disease 
            - adv_diagnosis_year : categorical
                year of metastatic diagnosis 
            
            Original date columns (DiagnosisDate and MetDiagnosisDate) retained if drop_dates = False

        Notes
        -----
        Output handling: 
        - GroupStage is not consolidated since already organized (0-IV, Not documented)
        - Duplicate PatientIDs are logged as warnings if found but retained in output
        - Processed DataFrame is stored in self.enhanced_df
        """
        # Input validation
        if patient_ids is not None:
            if not isinstance(patient_ids, list):
                raise TypeError("patient_ids must be a list or None")
        
        try:
            df = pd.read_csv(file_path)
            logging.info(f"Successfully read Enhanced_MetastaticBreast.csv file with shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")

            # Filter for specific PatientIDs if provided
            if patient_ids is not None:
                logging.info(f"Filtering for {len(patient_ids)} specific PatientIDs")
                df = df[df['PatientID'].isin(patient_ids)]
                logging.info(f"Successfully filtered Enhanced_MetastaticBreast.csv file with shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")
        
            df['GroupStage'] = df['GroupStage'].astype('category')

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

            logging.info(f"Successfully processed Enhanced_MetastaticBreast.csv file with final shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")
            self.enhanced_df = df
            return df

        except Exception as e:
            logging.error(f"Error processing Enhanced_MetastaticBreast.csv file: {e}")
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
                Maps all 50 states, plus DC and Puerto Rico (PR), to a US Census Bureau region
            - State : category
                US state (if drop_state=False)
            
        Notes
        -----
        Data cleaning and processing: 
        - Imputation for Race and Ethnicity:
            - If Race='Hispanic or Latino', Race value is replaced with NaN
            - If Race='Hispanic or Latino', Ethnicity is set to 'Hispanic or Latino'
            - Otherwise, missing Race and Ethnicity values remain unchanged
        - Ages calculated as <18 or >120 are logged as warning if found, but not removed
        - Missing States and Puerto Rico are imputed as unknown during the mapping to regions

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
                           days_after: int = 0) -> pd.DataFrame:
        """
        Processes Enhanced_MetBreastBiomarkers.csv by determining biomarker status for each patient within a specified time window relative to an index date. 

        Parameters
        ----------
        file_path : str
            Path to Enhanced_MetBreastBiomarkers.csv file
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
            - ER_status : category
                returns positive or negative status closest to index date 
            - PR_status : category
                returns positive or negative status closest to index date
            - HER2_status : category
                returns positive or negative status closest to index date
            - BRCA_status : category
                positive if ever-positive, negative if only-negative, otherwise unknown
            - PIK3CA_status : category
                positive if ever-positive, negative if only-negative, otherwise unknown
            - PDL1_status : category 
                positive if ever-positive, negative if only-negative, otherwise unknown
            - PDL1_percent_staining : category, ordered 
                returns a patient's maximum percent staining for PDL1 (NaN if no positive PDL1 results)

        Notes
        ------
        Biomarker cleaning processing: 
        - For ER, PR, and HER2, status is classifed according to these rules: 
            - Mos recent definitive result (positive or negative) within the specified window is selected
            - Unknowns are filtered out before selecting the most recent result to avoid
                mistakenly classify someone as "unknown" just because their most recent 
                test was indeterminate when they have previous definitive results
            - In the event of discordant definitive results on the same day, positive is selected 
                over negative. 
        
        - For the other biomarkers, status is classified as:
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
        
        if days_before is not None:
            if not isinstance(days_before, int) or days_before < 0:
                raise ValueError("days_before must be a non-negative integer or None")
        if not isinstance(days_after, int) or days_after < 0:
            raise ValueError("days_after must be a non-negative integer")

        try:
            df = pd.read_csv(file_path)
            logging.info(f"Successfully read Enhanced_MetBreastBiomarkers.csv file with shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")

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
            logging.info(f"Successfully merged Enhanced_MetBreastBiomarkers.csv df with index_date_df resulting in shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")
            
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

            for biomarker in ['ER', 'PR']:
                biomarker_dfs[biomarker] = (
                    df_filtered
                    .query(f'BiomarkerName == "{biomarker}"')
                    .assign(numeric_status = lambda x: x['BiomarkerStatus'].map({'Positive': 2, 'Negative': 1}).fillna(0))
                    .query('numeric_status != 0') # Filter out unknowns before selecting most recent result 
                    .sort_values(
                        by=['PatientID', 'ResultDate', 'numeric_status'], 
                        ascending=[True, False, False]) # Second False makes most recent ResultDate first and last False means highest numeric_status is first in ties
                    .groupby('PatientID')
                    .first()
                    .reset_index()
                    [['PatientID', 'BiomarkerStatus']]
                    .assign(BiomarkerStatus = lambda x: x['BiomarkerStatus'].map({'Positive': 'positive', 'Negative': 'negative'}).fillna('unknown'))
                    .rename(columns={'BiomarkerStatus': f'{biomarker}_status'})
                )

            # Process HER2
            positive_values = {
                'IHC positive (3+)',
                'FISH positive/amplified',
                'Positive NOS',
                'NGS positive (ERBB2 amplified)' 
            }

            negative_values = {
                'FISH negative/not amplified',
                'IHC 0',
                'IHC 1+',
                'Negative NOS',
                'NGS negative (ERBB2 not amplified)',
                'IHC negative, NOS'
            }

            biomarker_dfs['HER2'] = (
                df_filtered
                .query('BiomarkerName == "HER2"')
                .assign(numeric_status = lambda x: x['BiomarkerStatus'].apply(
                    lambda val: 2 if val in positive_values else (1 if val in negative_values else 0)
                ))
                .query('numeric_status != 0') # Filter out unknowns before selecting most recent result 
                .sort_values(
                    by=['PatientID', 'ResultDate', 'numeric_status'], 
                    ascending=[True, False, False]) # Second False makes most recent ResultDate first and last False means highest numeric_status is first in ties
                .groupby('PatientID')
                .first()
                .reset_index()
                [['PatientID', 'BiomarkerStatus']]
                .assign(BiomarkerStatus = lambda x: x['BiomarkerStatus'].apply(
                    lambda val: 'positive' if val in positive_values else ('negative' if val in negative_values else 'unknown')
                ))
                .rename(columns={'BiomarkerStatus': 'HER2_status'})
            )

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

            biomarker_dfs['BRCA'] = (
                df_filtered
                .query('BiomarkerName == "BRCA"')
                .groupby('PatientID')['BiomarkerStatus']
                .agg(lambda x: 'positive' if any(val in positive_values for val in x)
                    else ('negative' if any(val in negative_values for val in x)
                        else 'unknown'))
                .reset_index()
                .rename(columns={'BiomarkerStatus': 'BRCA_status'}) 
            )

            # Process PIK3CA and add to biomarker_dfs
            biomarker_dfs['PIK3CA'] = (
                df_filtered
                .query('BiomarkerName == "PIK3CA"')
                .groupby('PatientID')['BiomarkerStatus']
                .agg(lambda x: 'positive' if any ('Positive' in val for val in x)
                    else ('negative' if any('Negative' in val for val in x)
                        else 'unknown'))
                .reset_index()
                .rename(columns={'BiomarkerStatus': 'PIK3CA_status'})
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

            for biomarker in ['ER', 'PR', 'HER2', 'BRCA', 'PIK3CA', 'PDL1']:
                final_df = pd.merge(final_df, biomarker_dfs[biomarker], on = 'PatientID', how = 'left')

            # Convert to category type
            for biomarker_status in ['ER_status', 'PR_status', 'HER2_status', 'BRCA_status', 'PIK3CA_status', 'PDL1_status']:
                final_df[biomarker_status] = final_df[biomarker_status].astype('category')
            
            final_df = pd.merge(final_df, PDL1_staining_df, on = 'PatientID', how = 'left')

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

            logging.info(f"Successfully processed Enhanced_MetBreastBiomarkers.csv file with final shape: {final_df.shape} and unique PatientIDs: {(final_df['PatientID'].nunique())}")
            self.biomarkers_df = final_df
            return final_df

        except Exception as e:
            logging.error(f"Error processing Enhanced_MetBreastBiomarkers.csv file: {e}")
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
                       abnormal_reading_threshold: int = 2) -> pd.DataFrame:
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
        - All PatientIDs from index_date_df are included in the output and values will be NaN for patients without weight, BMI, or percent_change_weight, but set to 0 for hypotension, tachycardia, and fevers
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