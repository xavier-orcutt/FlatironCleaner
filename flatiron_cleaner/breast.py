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

    def __init__(self):
        self.enhanced_df = None
        self.demographics_df = None
        self.practice_df = None
        self.biomarkers_df = None
    
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
        GroupStage is not consolidated since already organized (0-IV, Not documented)
        Duplicate PatientIDs are logged as warnings if found but retained in output
        Processed DataFrame is stored in self.enhanced_df
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
        Imputation for Race and Ethnicity:
            - If Race='Hispanic or Latino', Race value is replaced with NaN
            - If Race='Hispanic or Latino', Ethnicity is set to 'Hispanic or Latino'
            - Otherwise, missing Race and Ethnicity values remain unchanged
        Ages calculated as <18 or >120 are logged as warning if found, but not removed
        Missing States and Puerto Rico are imputed as unknown during the mapping to regions
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
        For ER, PR, and HER2, status is classifed according to these rules: 
            - Mos recent definitive result (positive or negative) within the specified window is selected
            - Unknowns are filtered out before selecting the most recent result to avoid
                mistakenly classify someone as "unknown" just because their most recent 
                test was indeterminate when they have previous definitive results
            - In the event of discordant definitive results on the same day, positive is selected 
                over negative. 
        
        For the other biomarkers, status is classified as:
            - 'positive' if any test result is positive (ever-positive)
            - 'negative' if any test is negative without positives (only-negative) 
            - 'unknown' if all results are indeterminate
        
        Missing biomarker data handling:
        - All PatientIDs from index_date_df are included in the output
        - Patients without any biomarker tests will have NaN values for all biomarker columns
        - Missing ResultDate is imputed with SpecimenReceivedDate

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
            return final_df

        except Exception as e:
            logging.error(f"Error processing Enhanced_MetBreastBiomarkers.csv file: {e}")
            return None