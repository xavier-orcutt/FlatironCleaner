import pandas as pd
import numpy as np
import logging

logging.basicConfig(
    level = logging.INFO,                                 
    format = '%(asctime)s - %(levelname)s - %(message)s'  
)

class DataProcessorUrothelial:
    
    GROUP_STAGE_MAPPING = {
        'Stage IV': 'Stage IV',
        'Stage IVA': 'Stage IV',
        'Stage IVB': 'Stage IV',
        'Stage III': 'Stage III',
        'Stage IIIA': 'Stage III',
        'Stage IIIB': 'Stage III',
        'Stage II': 'Stage II',
        'Stage I': 'Stage I',
        'Stage 0is': 'Stage 0',
        'Stage 0a': 'Stage 0',
        'Unknown/not documented': 'Unknown/not documented'
    }

    T_STAGE_MAPPING = {
        'T4': 'T4',
        'T4a': 'T4',
        'T4b': 'T4',
        'T3': 'T3',
        'T3a': 'T3',
        'T3b': 'T3',
        'T2': 'T2',
        'T2a': 'T2',
        'T2b': 'T2',
        'T1': 'T1',
        'T0': 'T0',
        'Ta': 'Ta',
        'Tis': 'Tis',
        'TX': 'TX',
        'Unknown/not documented': 'Unknown/not documented'
    }

    M_STAGE_MAPPING = {
        'M1': 'M1',
        'M1a': 'M1',
        'M1b': 'M1',
        'M0': 'M0',
        'MX': 'MX',
        'Unknown/not documented': 'Unknown/not documented'
    }

    STATE_REGIONS = {
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
                             drop_stages: bool = True, 
                             drop_dates: bool = True) -> pd.DataFrame: 
        """
        Process Enhanced_AdvUrothelial.csv file by cleaning data types and calculating time-based variables.

        Parameters:
            file_path (str): Path to Enhanced_AdvUrothelial.csv file
            patient_ids (list, optional): List of specific PatientIDs to process. If None, processes all patients. Defaults to None.
            drop_stages (bool, optional): If True, drops GroupStage, TStage, and MStage after calculations. Defaults to True.
            drop_dates (bool, optional): If True, drops date columns after calculations. Defaults to True.

        Returns:
            pd.DataFrame: Processed DataFrame with:
                - Categorical columns (PrimarySite, DiseaseGrade, NStage, SmokingStatus, SurgeryType)
                - Boolean column (Surgery)
                - Calculated columns:
                    * GroupStage_mod: Consolidated staging (Stage 0, I, II, III, IV, Unknown)
                    * TStage_mod: Consolidated T staging (T0-T4, Ta, Tis, TX, Unknown)
                    * MStage_mod: Consolidated M staging (M0, M1, MX, Unknown)
                    * days_diagnosis_to_advanced: Days between initial and advanced diagnosis
                    * advanced_diagnosis_year: Year of advanced diagnosis (as category)
                    * days_diagnosis_to_surgery: Days between initial diagnosis and surgery
                - Original GroupStage, TStage, and MStage columns dropped if drop_stages = True
                - Original date columns dropped if drop_dates = True

        Notes:
            - Checks for and logs duplicate PatientIDs
            - Stores processed DataFrame in self.enhanced_df
        """
        try:
            df = pd.read_csv(file_path)

            # Filter for specific PatientIDs if provided
            if patient_ids is not None:
                df = df[df['PatientID'].isin(patient_ids)]
                logging.info(f"Filtered for {len(patient_ids)} specific PatientIDs")

            logging.info(f"Successfully read Enhanced_AdvUrothelial.csv file with shape: {df.shape} and unique PatientIDs: {(df.PatientID.nunique())}")
        
            # Convert categorical columns
            categorical_cols = ['PrimarySite', 
                                'DiseaseGrade',
                                'GroupStage',
                                'TStage', 
                                'NStage',
                                'MStage', 
                                'SmokingStatus', 
                                'SurgeryType']
        
            df[categorical_cols] = df[categorical_cols].astype('category')

            # Recode stage variables using class-level mapping and create new column
            df['GroupStage_mod'] = df['GroupStage'].map(self.GROUP_STAGE_MAPPING).astype('category')
            df['TStage_mod'] = df['TStage'].map(self.T_STAGE_MAPPING).astype('category')
            df['MStage_mod'] = df['MStage'].map(self.M_STAGE_MAPPING).astype('category')

            # Drop stage variables if specified
            if drop_stages:
                df = df.drop(columns=['GroupStage', 'TStage', 'MStage'])
        
            # Convert date columns
            date_cols = ['DiagnosisDate', 'AdvancedDiagnosisDate', 'SurgeryDate']
            for col in date_cols:
                df[col] = pd.to_datetime(df[col])
            
            # Convert boolean column
            df['Surgery'] = df['Surgery'].astype(bool)

            # Generate new variables 
            df['days_diagnosis_to_advanced'] = (df['AdvancedDiagnosisDate'] - df['DiagnosisDate']).dt.days
            df['advanced_diagnosis_year'] = pd.Categorical(df['AdvancedDiagnosisDate'].dt.year)
            df['days_diagnosis_to_surgery'] = (df['SurgeryDate'] - df['DiagnosisDate']).dt.days
    
            # Drop date-based variables if specified
            if drop_dates:
                df = df.drop(columns=['AdvancedDiagnosisDate', 'DiagnosisDate', 'SurgeryDate'])

            # Check for duplicate PatientIDs
            if len(df) > df.PatientID.nunique():
                logging.error(f"Duplicate PatientIDs found")
                return None

            logging.info(f"Successfully processed Enhanced_AdvUrothelial.csv file with final shape: {df.shape} and unique PatientIDs: {(df.PatientID.nunique())}")
            self.enhanced_df = df
            return df

        except Exception as e:
            logging.error(f"Error processing Enhanced_AdvUrothelial.csv file: {e}")
            return None
    
    def process_demographics(self, 
                        file_path: str,
                        patient_ids: list = None, 
                        reference_dates_df: pd.DataFrame = None,
                        date_column: str = None,
                        drop_state: bool = True) -> pd.DataFrame:
        """
        Process Demographics.csv file by cleaning data types, calculating age, and mapping states to regions.
    
        Parameters:
            file_path (str): Path to Demographics.csv file
            patient_ids (list, optional): List of specific PatientIDs to process. If None, processes all patients. Defaults to None.
            reference_dates_df (pd.DataFrame, optional): DataFrame containing PatientID and reference dates 
                (e.g., AdvancedDiagnosisDate or 1L StartDate) for age calculation
            date_column (str, optional): Name of the date column in reference_dates_df to use for age calculation
            drop_state (bool, optional): If True, drops the State column after mapping to regions. Defaults to True.

        Returns:
            pd.DataFrame: Processed DataFrame with:
                - Categorical columns (Gender, Race, Ethnicity, Region)
                - Age calculated if reference dates provided
                - States mapped to Census Bureau regions
                - Hispanic/Latino ethnicity standardized

        Notes:
            - Checks for and logs duplicate PatientIDs
            - Stores processed DataFrame in self.demographics_df
        """
        try:
            df = pd.read_csv(file_path)

            # Filter for specific PatientIDs if provided
            if patient_ids is not None:
                df = df[df['PatientID'].isin(patient_ids)]
                logging.info(f"Filtered for {len(patient_ids)} specific PatientIDs")

            logging.info(f"Successfully read Demographics.csv file with shape: {df.shape} and unique PatientIDs: {(df.PatientID.nunique())}")

            # Initial data type conversions
            df['BirthYear'] = df['BirthYear'].astype('int64')
            df['Gender'] = df['Gender'].astype('category')
            df['State'] = df['State'].astype('category')
        
            # Age calculation block (if reference dates provided)
            if reference_dates_df is not None:
                # Validate reference data
                if 'PatientID' not in reference_dates_df.columns:
                    logging.error("reference_dates_df must contain 'PatientID' column")
                    return None
            
                if date_column is None:
                    logging.error("date_column must be specified when reference_dates_df is provided")
                    return None
                
                if date_column not in reference_dates_df.columns:
                    logging.error(f"Column '{date_column}' not found in reference_dates_df")
                    return None

                # Process dates and calculate age
                reference_dates_df[date_column] = pd.to_datetime(reference_dates_df[date_column])
                df = pd.merge(
                    df,
                    reference_dates_df[['PatientID', date_column]], 
                    on = 'PatientID',
                    how = 'left'
                )
        
                df['age'] = df[date_column].dt.year - df['BirthYear']

                # Age validation
                mask_invalid_age = (df['age'] < 18) | (df['age'] > 120)
                if mask_invalid_age.any():
                    logging.warning(f"Found {mask_invalid_age.sum()} ages outside valid range (18-120)")

                # Drop the date column and BirthYear after age calculation
                df = df.drop(columns = [date_column, 'BirthYear'])


            # Race and Ethnicity processing
            # If Race == 'Hispanic or Latino', fill 'Hispanic or Latino' for Ethnicity
            df['Ethnicity'] = np.where(df['Race'] == 'Hispanic or Latino', 'Hispanic or Latino', df['Ethnicity'])

            # If Race == 'Hispanic or Latino' replace with Nan
            df['Race'] = np.where(df['Race'] == 'Hispanic or Latino', np.nan, df['Race'])

            df[['Race', 'Ethnicity']] = df[['Race', 'Ethnicity']].astype('category')

            # Region processing
            # Group states into Census-Bureau regions  
            df['region'] = (df['State']
                            .map(self.STATE_REGIONS)
                            .fillna('unknown')
                            .astype('category'))

            # Drop State varibale if specified
            if drop_state:               
                df = df.drop(columns = ['State'])

            # Check for duplicate PatientIDs
            if len(df) > df.PatientID.nunique():
                logging.error(f"Duplicate PatientIDs found")
                return None
            
            logging.info(f"Successfully processed Demographics.csv file with final shape: {df.shape} and unique PatientIDs: {(df.PatientID.nunique())}")
            self.demographics_df = df
            return df

        except Exception as e:
            logging.error(f"Error processing demographics file: {e}")
            return None
        

    def process_practice(self,
                         file_path: str,
                         patient_ids: list = None) -> pd.DataFrame:
        """
        Process Practice.csv file by consolidating practice types per patient.
    
        Parameters:
            file_path (str): Path to Practice.csv file
            patient_ids (list, optional): List of specific PatientIDs to process. If None, processes all patients. Defaults to None.

        Returns:
            pd.DataFrame: Processed DataFrame with:
                - PracticeType_mod (ACADEMIC, COMMUNITY, BOTH)
        Notes:
            - Checks for and logs duplicate PatientIDs
            - Stores processed DataFrame in self.practice_df
        """
        try:
            df = pd.read_csv(file_path)

            # Filter for specific PatientIDs if provided
            if patient_ids is not None:
                df = df[df['PatientID'].isin(patient_ids)]
                logging.info(f"Filtered for {len(patient_ids)} specific PatientIDs")

            logging.info(f"Successfully read Practice.csv file with shape: {df.shape} and unique PatientIDs: {(df.PatientID.nunique())}")

            df = df[['PatientID', 'PracticeType']]

            # Group by PatientID and get set of unique PracticeTypes
            grouped = df.groupby('PatientID')['PracticeType'].unique()
            new_df = pd.DataFrame(grouped).reset_index()

            # Function to determine the modified practice type
            def get_practice_type(practice_types):
                if len(practice_types) > 1:
                    return 'BOTH'
                return practice_types[0]
            
            # Apply the function to the column containing sets
            new_df['PracticeType_mod'] = new_df['PracticeType'].apply(get_practice_type).astype('category')

            new_df = new_df[['PatientID', 'PracticeType_mod']]

            # Check for duplicate PatientIDs
            if len(new_df) > new_df.PatientID.nunique():
                logging.error(f"Duplicate PatientIDs found")
                return None
            
            logging.info(f"Successfully processed Practice.csv file with final shape: {new_df.shape} and unique PatientIDs: {(new_df.PatientID.nunique())}")
            self.practice_df = new_df
            return new_df

        except Exception as e:
            logging.error(f"Error processing practice file: {e}")
            return None