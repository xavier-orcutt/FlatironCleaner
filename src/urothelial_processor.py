import pandas as pd
import numpy as np
import logging
from typing import Optional

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

    SURGERY_TYPE_MAPPING = {
        'Cystoprostatectomy': 'bladder',
        'Complete (radical) cystectomy': 'bladder',
        'Partial cystectomy': 'bladder',
        'Cystectomy, NOS': 'bladder',
        'Nephroureterectomy': 'upper',
        'Nephrectomy': 'upper',
        'Ureterectomy': 'upper', 
        'Urethrectomy': 'other',
        'Other': 'other',
        'Unknown/not documented': 'unknown', 
        np.nan: 'unknown'
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
        self.mortality_df = None 
        self.biomarkers_df = None
        self.ecog_df = None
        self.vitals_df = None
        self.labs_df = None

    def process_enhanced_adv(self,
                             file_path: str,
                             patient_ids: list = None,
                             drop_stages: bool = True, 
                             drop_surgery_type: bool = True,
                             drop_dates: bool = True) -> pd.DataFrame: 
        """
        Processes Enhanced_AdvUrothelial.csv to standardize categories, consolidate 
        staging information, and calculate time-based metrics between key clinical events.

        Parameters
        ----------
        file_path : str
            Path to Enhanced_AdvUrothelial.csv file
        patient_ids : list, optional
            List of specific PatientIDs to process. If None, processes all patients
        drop_stages : bool, default=True
            If True, drops original staging columns (GroupStage, TStage, and MStage) after creating modified versions
        drop_surgery_type : bool, default=True
            If True, drops original surgery type after creating modified version
        drop_dates : bool, default=True
            If True, drops date columns after calculating durations

        Returns
        -------
        pd.DataFrame
            Processed DataFrame containing:
            - PatientID : unique patient identifier
            - PrimarySite : anatomical site of cancer
            - SmokingStatus : smoking history
            - Surgery : whether surgery was performed (boolean)
            - SurgeryType_mod : consolidated surgery type
            - days_diagnosis_to_surgery : days from diagnosis to surgery
            - DiseaseGrade : tumor grade
            - NStage : lymph node staging
            - GroupStage_mod : consolidated overall staging (0-IV, Unknown)
            - TStage_mod : consolidated tumor staging (T0-T4, Ta, Tis, TX, Unknown)
            - MStage_mod : consolidated metastasis staging (M0, M1, MX, Unknown)
            - days_diagnosis_to_advanced : days from initial to advanced diagnosis
            - adv_diagnosis_year : year of advanced diagnosis (categorical)
            - days_diagnosis_to_adv : days from diagnosis to advanced disease 
            
            Original staging and date columns retained if respective drop_* = False

        Notes
        -----
        - Duplicate PatientIDs are logged as warnings if found
        - Processed DataFrame is stored in self.enhanced_df
        """
        try:
            df = pd.read_csv(file_path)
            logging.info(f"Successfully read Enhanced_AdvUrothelial.csv file with shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")

            # Filter for specific PatientIDs if provided
            if patient_ids is not None:
                logging.info(f"Filtering for {len(patient_ids)} specific PatientIDs")
                df = df[df['PatientID'].isin(patient_ids)]
                logging.info(f"Successfully filtered Enhanced_AdvUrothelial.csv file with shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")
        
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

            # Drop original stage variables if specified
            if drop_stages:
                df = df.drop(columns=['GroupStage', 'TStage', 'MStage'])

            # Recode surgery type variable using class-level mapping and create new column
            df['SurgeryType_mod'] = df['SurgeryType'].map(self.SURGERY_TYPE_MAPPING).astype('category')

            # Drop original surgery type variable if specified
            if drop_surgery_type:
                df = df.drop(columns=['SurgeryType'])

            # Convert date columns
            date_cols = ['DiagnosisDate', 'AdvancedDiagnosisDate', 'SurgeryDate']
            for col in date_cols:
                df[col] = pd.to_datetime(df[col])
            
            # Convert boolean column
            df['Surgery'] = df['Surgery'].astype(bool)

            # Generate new variables 
            df['days_diagnosis_to_adv'] = (df['AdvancedDiagnosisDate'] - df['DiagnosisDate']).dt.days
            df['adv_diagnosis_year'] = pd.Categorical(df['AdvancedDiagnosisDate'].dt.year)
            df['days_diagnosis_to_surgery'] = (df['SurgeryDate'] - df['DiagnosisDate']).dt.days
    
            if drop_dates:
                df = df.drop(columns = ['AdvancedDiagnosisDate', 'DiagnosisDate', 'SurgeryDate'])

            # Check for duplicate PatientIDs
            if len(df) > df['PatientID'].nunique():
                logging.error(f"Duplicate PatientIDs found")
                return None

            logging.info(f"Successfully processed Enhanced_AdvUrothelial.csv file with final shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")
            self.enhanced_df = df
            return df

        except Exception as e:
            logging.error(f"Error processing Enhanced_AdvUrothelial.csv file: {e}")
            return None
    
    def process_demographics(self, 
                            file_path: str,
                            patient_ids: list = None, 
                            index_date_df: pd.DataFrame = None,
                            index_date_column: str = None,
                            drop_state: bool = True) -> pd.DataFrame:
        """
        Processes Demographics.csv by standardizing categorical variables, mapping states 
        to census regions, and calculating age at index date if provided.

        Parameters
        ----------
        file_path : str
            Path to Demographics.csv file
        patient_ids : list, optional
            List of specific PatientIDs to process. If None, processes all patients
        index_dates_df : pd.DataFrame, optional
            DataFrame containing PatientID and index dates for age calculation
        index_date_column : str, optional
            Column name in index_date_df containing dates for age calculation
        drop_state : bool, default = True
            If True, drops State column after mapping to regions

        Returns
        -------
        pd.DataFrame
            Processed DataFrame containing:
            - PatientID : unique patient identifier
            - Gender : standardized gender category
            - Race : standardized race category 
            - Ethnicity : standardized ethnicity (Hispanic/Latino status)
            - age : age at index date (if index_date_df provided)
            - region : US Census Bureau region
            - State : US state (if drop_state=False)
            
        Notes
        -----
        - Duplicate PatientIDs are logged as warnings if found
        - Processed DataFrame is stored in self.demographics_df
        """
        try:
            df = pd.read_csv(file_path)
            logging.info(f"Successfully read Demographics.csv file with shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")

            # Filter for specific PatientIDs if provided
            if patient_ids is not None:
                logging.info(f"Filtering for {len(patient_ids)} specific PatientIDs")
                df = df[df['PatientID'].isin(patient_ids)]
                logging.info(f"Successfully filtered Demographics.csv file with shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")

            # Initial data type conversions
            df['BirthYear'] = df['BirthYear'].astype('int64')
            df['Gender'] = df['Gender'].astype('category')
            df['State'] = df['State'].astype('category')
        
            # Age calculation block (if index dates provided)
            if index_date_df is not None:
                # Validate index data
                if 'PatientID' not in index_date_df.columns:
                    logging.error("index_dates_df must contain 'PatientID' column")
                    return None
            
                if index_date_column is None:
                    logging.error("index_date_column must be specified when index_date_df is provided")
                    return None
                
                if index_date_column not in index_date_df.columns:
                    logging.error(f"Column '{index_date_column}' not found in index_date_df")
                    return None

                # Process dates and calculate age
                index_date_df[index_date_column] = pd.to_datetime(index_date_df[index_date_column])
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
                            .map(self.STATE_REGIONS)
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
            self.demographics_df = df
            return df

        except Exception as e:
            logging.error(f"Error processing demographics file: {e}")
            return None
        

    def process_practice(self,
                         file_path: str,
                         patient_ids: list = None) -> pd.DataFrame:
        """
        Processes Practice.csv to consolidate practice types per patient into a single 
        categorical value indicating academic, community, or both settings.

        Parameters
        ----------
        file_path : str
            Path to Practice.csv file
        patient_ids : list, optional
            List of specific PatientIDs to process. If None, processes all patients

        Returns
        -------
        pd.DataFrame
            Processed DataFrame containing:
            - PatientID : unique patient identifier  
            - PracticeType_mod : practice setting (ACADEMIC, COMMUNITY, or BOTH)
       
        Notes
        -----
        - Duplicate PatientIDs are logged as warnings if found
        - Processed DataFrame is stored in self.practice_df
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
            if len(new_df) > new_df['PatientID'].nunique():
                logging.error(f"Duplicate PatientIDs found")
                return None
            
            logging.info(f"Successfully processed Practice.csv file with final shape: {new_df.shape} and unique PatientIDs: {(new_df['PatientID'].nunique())}")
            self.practice_df = new_df
            return new_df

        except Exception as e:
            logging.error(f"Error processing practice file: {e}")
            return None
        
    def process_mortality(self,
                          file_path: str,
                          index_date_df: pd.DataFrame,
                          index_date_column: str,
                          visit_path: str = None, 
                          telemedicine_path: str = None, 
                          biomarker_path: str = None, 
                          oral_path: str = None,
                          progression_path: str = None,
                          drop_dates: bool = True) -> pd.DataFrame:
        """
        Processes Enhanced_Mortality_V2.csv by cleaning data types, calculating time 
        from index date to death/censor, and determining mortality events. Handles
        incomplete death dates by imputing missing day/month values.

        Parameters
        ----------
        file_path : str
            Path to Enhanced_Mortality_V2.csv file
        index_date_df : pd.DataFrame
            DataFrame containing PatientID and index dates. Only mortality data for PatientIDs present in this DataFrame will be processed
        index_date_column : str
            Column name in index_date_df containing the index date
        df_merge_type : str, default='left'
            Merge type for pd.merge(index_date_df, mortality_data, on = 'PatientID', how = df_merge_type)
        visit_path : str
            Path to Visit.csv file
        telemedicine_path : str
            Path to Telemedicine.csv file
        biomarker_path : str
            Path to Enhanced_AdvUrothelialBiomarkers.csv file
        oral_path : str
            Path to Enhanced_AdvUrothelial_Orals.csv file
        progression_path : str
            Path to Enhanced_AdvUrothelial_Progression.csv file
        drop_dates : bool, default = True
            If True, drops date columns (index_date_column, DateOfDeath, last_ehr_date)   
        
        Returns
        -------
        pd.DataFrame
            Processed DataFrame containing:
            - PatientID : unique patient identifier
            - duration : days from index date to death/censor
            - event : mortality status (1 = death, 0 = censored)

        Notes
        ------
        Death date imputation:
        - Missing day : Imputed to 15th of the month
        - Missing month and day : Imputed to July 1st
    
        Duplicate PatientIDs are logged as warnings if found
        Processed DataFrame is stored in self.mortality_df
        """

        # Input validation
        if not isinstance(index_date_df, pd.DataFrame) or 'PatientID' not in index_date_df.columns:
            raise ValueError("index_date_df must be a DataFrame containing 'PatientID' column")
    
        if not index_date_column or index_date_column not in index_date_df.columns:
            raise ValueError(f"Column '{index_date_column}' not found in index_date_df")

        try:
            df = pd.read_csv(file_path)
            logging.info(f"Successfully read Enhanced_Mortality_V2.csv file with shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")

            # When only year is available: Impute to July 1st (mid-year)
            df['DateOfDeath'] = np.where(df['DateOfDeath'].str.len() == 4, df['DateOfDeath'] + '-07-01', df['DateOfDeath'])

            # When only month and year are available: Impute to the 15th day of the month
            df['DateOfDeath'] = np.where(df['DateOfDeath'].str.len() == 7, df['DateOfDeath'] + '-15', df['DateOfDeath'])

            df['DateOfDeath'] = pd.to_datetime(df['DateOfDeath'])

            # Process index dates and merge
            index_date_df[index_date_column] = pd.to_datetime(index_date_df[index_date_column])
            df_death = pd.merge(
                index_date_df[['PatientID', index_date_column]],
                df,
                on = 'PatientID',
                how = 'left'
            )
            
            logging.info(f"Successfully merged Enhanced_Mortality_V2.csv df with index_date_df resulting in shape: {df_death.shape} and unique PatientIDs: {(df_death.PatientID.nunique())}")
                
            # Create event column
            df_death['event'] = df_death['DateOfDeath'].notna().astype(int)

            # Initialize df_final
            df_final = df_death

            # Determine last EHR data
            if all(path is None for path in [visit_path, telemedicine_path, biomarker_path, oral_path, progression_path]):
                logging.info("WARNING: At least one of visit_path, telemedicine_path, biomarker_path, oral_path, or progression_path must be provided to calculate duration for those with a missing death date")

            else: 
                if visit_path is not None and telemedicine_path is not None:
                    try:
                        df_visit = pd.read_csv(visit_path)
                        df_tele = pd.read_csv(telemedicine_path)

                        df_visit_tele = (
                            pd.concat([
                                df_visit[['PatientID', 'VisitDate']],
                                df_tele[['PatientID', 'VisitDate']]
                                ]))
                        
                        df_visit_tele['VisitDate'] = pd.to_datetime(df_visit_tele['VisitDate'])

                        df_visit_tele_max = (
                            df_visit_tele
                            .query("PatientID in @index_date_df.PatientID")  
                            .groupby('PatientID', observed = True)['VisitDate']  
                            .max()
                            .to_frame(name = 'last_visit_date')          
                            .reset_index()
                            )
                    except Exception as e:
                        logging.error(f"Error reading Visit.csv and/or Telemedicine.csv files: {e}")
                        return None

                if visit_path is not None and telemedicine_path is None:
                    try: 
                        df_visit = pd.read_csv(visit_path)
                        df_visit['VisitDate'] = pd.to_datetime(df_visit['VisitDate'])

                        df_visit_max = (
                            df_visit
                            .query("PatientID in @index_date_df.PatientID")  
                            .groupby('PatientID', observed = True)['VisitDate']  
                            .max()
                            .to_frame(name = 'last_visit_date')          
                            .reset_index()
                        )
                    except Exception as e:
                        logging.error(f"Error reading Visit.csv file: {e}")
                        return None

                if telemedicine_path is not None and visit_path is None:
                    try: 
                        df_tele = pd.read_csv(telemedicine_path)
                        df_tele['VisitDate'] = pd.to_datetime(df_tele['VisitDate'])

                        df_tele_max = (
                            df_tele
                            .query("PatientID in @index_date_df.PatientID")  
                            .groupby('PatientID', observed = True)['VisitDate']  
                            .max()
                            .to_frame(name = 'last_visit_date')          
                            .reset_index()
                        )
                    except Exception as e:
                        logging.error(f"Error reading Telemedicine.csv file: {e}")
                        return None
                                            
                if biomarker_path is not None:
                    try: 
                        df_biomarker = pd.read_csv(biomarker_path)
                        df_biomarker['SpecimenCollectedDate'] = pd.to_datetime(df_biomarker['SpecimenCollectedDate'])

                        df_biomarker_max = (
                            df_biomarker
                            .query("PatientID in @index_date_df.PatientID")
                            .groupby('PatientID', observed = True)['SpecimenCollectedDate'].max()
                            .to_frame(name = 'last_biomarker_date')
                            .reset_index()
                        )
                    except Exception as e:
                        logging.error(f"Error reading Enhanced_AdvUrothelialBiomarkers.csv file: {e}")
                        return None

                if oral_path is not None:
                    try:
                        df_oral = pd.read_csv(oral_path)
                        df_oral['StartDate'] = pd.to_datetime(df_oral['StartDate'])
                        df_oral['EndDate'] = pd.to_datetime(df_oral['EndDate'])

                        df_oral_max = (
                            df_oral
                            .query("PatientID in @index_date_df.PatientID")
                            .assign(max_date = df_oral[['StartDate', 'EndDate']].max(axis = 1))
                            .groupby('PatientID', observed = True)['max_date'].max()
                            .to_frame(name = 'last_oral_date')
                            .reset_index()
                        )
                    except Exception as e:
                        logging.error(f"Error reading Enhanced_AdvUrothelial_Orals.csv file: {e}")
                        return None

                if progression_path is not None:
                    try: 
                        df_progression = pd.read_csv(progression_path)
                        df_progression['ProgressionDate'] = pd.to_datetime(df_progression['ProgressionDate'])
                        df_progression['LastClinicNoteDate'] = pd.to_datetime(df_progression['LastClinicNoteDate'])

                        df_progression_max = (
                            df_progression
                            .query("PatientID in @index_date_df.PatientID")
                            .assign(max_date = df_progression[['ProgressionDate', 'LastClinicNoteDate']].max(axis = 1))
                            .groupby('PatientID', observed = True)['max_date'].max()
                            .to_frame(name = 'last_progression_date')
                            .reset_index()
                        )
                    except Exception as e:
                        logging.error(f"Error reading Enhanced_AdvUrothelial_Progression.csv file: {e}")
                        return None

                # Create a dictionary to store all available dataframes
                dfs_to_merge = {}

                # Add dataframes to dictionary if they exist
                if visit_path is not None and telemedicine_path is not None:
                    dfs_to_merge['visit_tele'] = df_visit_tele_max
                elif visit_path is not None:
                    dfs_to_merge['visit'] = df_visit_max
                elif telemedicine_path is not None:
                    dfs_to_merge['tele'] = df_tele_max

                if biomarker_path is not None:
                    dfs_to_merge['biomarker'] = df_biomarker_max
                if oral_path is not None:
                    dfs_to_merge['oral'] = df_oral_max
                if progression_path is not None:
                    dfs_to_merge['progression'] = df_progression_max

                # Merge all available dataframes
                if dfs_to_merge:
                    df_last_ehr_activity = None
                    for name, df in dfs_to_merge.items():
                        if df_last_ehr_activity is None:
                            df_last_ehr_activity = df
                        else:
                            df_last_ehr_activity = pd.merge(df_last_ehr_activity, df, on = 'PatientID', how = 'outer')

                if df_last_ehr_activity is not None:
                    # Get the available date columns that exist in our merged dataframe
                    last_date_columns = [col for col in ['last_visit_date', 'last_oral_date', 'last_biomarker_date', 'last_progression_date']
                                        if col in df_last_ehr_activity.columns]
                    logging.info(f"The follwing columns {last_date_columns} are used to calculate the last EHR date")
                    
                    if last_date_columns:
                        single_date = (
                            df_last_ehr_activity
                            .assign(last_ehr_activity = df_last_ehr_activity[last_date_columns].max(axis = 1))
                            .filter(items = ['PatientID', 'last_ehr_activity'])
                        )

                        df_final = pd.merge(df_death, single_date, on = 'PatientID', how = 'left')

            # Calculate duration
            if 'last_ehr_activity' in df_final.columns:
                df_final['duration'] = np.where(df_final['event'] == 0, 
                                                (df_final['last_ehr_activity'] - df_final[index_date_column]).dt.days, 
                                                (df_final['DateOfDeath'] - df_final[index_date_column]).dt.days)
                
                # Drop date varibales if specified
                if drop_dates:               
                    df_final = df_final.drop(columns = [index_date_column, 'DateOfDeath', 'last_ehr_activity'])

                # Check for duplicate PatientIDs
                if len(df_final) > df_final.PatientID.nunique():
                    logging.error(f"Duplicate PatientIDs found")

                logging.info(f"Successfully processed Enhanced_Mortality_V2.csv file with final shape: {df_final.shape} and unique PatientIDs: {(df_final['PatientID'].nunique())}. There are {df_final['duration'].isna().sum()} out of {df_final['PatientID'].nunique()} patients with missing duration values")
                self.mortality_df = df_final
                return df_final
                       
            else: 
                df_final['duration'] = (df_final['DateOfDeath'] - df_final[index_date_column]).dt.days

                # Drop date varibales if specified
                if drop_dates:               
                    df_final = df_final.drop(columns = [index_date_column, 'DateOfDeath'])

                # Check for duplicate PatientIDs
                if len(df_final) > df_final.PatientID.nunique():
                    logging.error(f"Duplicate PatientIDs found")

                logging.info(f"Successfully processed Enhanced_Mortality_V2.csv file with final shape: {df_final.shape} and unique PatientIDs: {(df_final['PatientID'].nunique())}. There are {df_final['duration'].isna().sum()} out of {df_final['PatientID'].nunique()} patients with missing duration values")
                self.mortality_df = df_final
                return df_final

        except Exception as e:
            logging.error(f"Error processing Enhanced_Mortality_V2.csv file: {e}")
            return None
        
    def process_biomarkers(self,
                           file_path: str,
                           index_date_df: pd.DataFrame,
                           index_date_column: str, 
                           days_before: Optional[int] = None,
                           days_after: int = 0) -> pd.DataFrame:
        """
        Processes Enhanced_AdvUrothelialBiomarkers.csv by determining FGFR and PDL1 status for each patient within a specified time window relative to an index date
        For each biomarker:
        - FGFR status is classified as:
            - 'positive' if any test result is positive (ever-positive)
            - 'negative' if any test is negative without positives (only-negative) 
            - 'unknown' if all results are indeterminate
        - PDL1 status follows the same classification logic
        - PDL1 staining percentage is also captured

        Parameters
        ----------
        file_path : str
            Path to Enhanced_AdvUrothelialBiomarkers.csv file
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
            Processed DataFrame containing:
            - PatientID : unique patient identifier
            - fgfr_status : positive if ever-positive, negative if only-negative, otherwise unknown
            - pdl1_status : positive if ever-positive, negative if only-negative, otherwise unknown
            - pdl1_staining : returns a patient's maximum percent staining for PDL1

        Notes
        ------
        Missing ResultDate is imputed with SpecimenReceivedDate.
        Duplicate PatientIDs are logged as warnings if found
        Processed DataFrame is stored in self.biomarkers_df
        """

        # Input validation
        if not isinstance(index_date_df, pd.DataFrame) or 'PatientID' not in index_date_df.columns:
            raise ValueError("index_date_df must be a DataFrame containing 'PatientID' column")
        if not index_date_column or index_date_column not in index_date_df.columns:
            raise ValueError(f"Column '{index_date_column}' not found in index_date_df")
        
        if days_before is not None:
            if not isinstance(days_before, int) or days_before < 0:
                raise ValueError("days_before must be a non-negative integer or None")
        if not isinstance(days_after, int) or days_after < 0:
            raise ValueError("days_after must be a non-negative integer")

        try:
            df = pd.read_csv(file_path)
            logging.info(f"Successfully read Enhanced_AdvUrothelialBiomarkers.csv file with shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")

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
            logging.info(f"Successfully merged Enhanced_AdvUrothelialBiomarkers.csv df with index_date_df resulting in shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")
            
            # Create new variable 'index_to_result' that notes difference in days between resulted specimen and index date
            df['index_to_result'] = (df['ResultDate'] - df[index_date_column]).dt.days
            
            # Select biomarkers that fall within desired before and after index date
            if days_before is None:
                # Only filter for days after
                df_filtered = df[df['index_to_result'] <= days_after].copy()
                window_desc = f"negative infinity to +{days_after} days from index date"
            else:
                # Filter for both before and after
                df_filtered = df[
                    (df['index_to_result'] <= days_after) & 
                    (df['index_to_result'] >= -days_before)
                ].copy()
                window_desc = f"-{days_before} to +{days_after} days from index date"

            logging.info(f"After applying window period of {window_desc}, "f"remaining records: {df_filtered.shape}, "f"unique PatientIDs: {df_filtered['PatientID'].nunique()}")

            # Process FGFR status
            fgfr_df = (
                df_filtered
                .query('BiomarkerName == "FGFR"')
                .groupby('PatientID')['BiomarkerStatus']
                .agg(lambda x: 'positive' if any ('Positive' in val for val in x)
                    else ('negative' if any('Negative' in val for val in x)
                        else 'unknown'))
                .reset_index()
                .rename(columns={'BiomarkerStatus': 'fgfr_status'})
                )
            
            # Process PDL1 status
            pdl1_df = (
                df_filtered
                .query('BiomarkerName == "PDL1"')
                .groupby('PatientID')['BiomarkerStatus']
                .agg(lambda x: 'positive' if any ('PD-L1 positive' in val for val in x)
                    else ('negative' if any('PD-L1 negative/not detected' in val for val in x)
                        else 'unknown'))
                .reset_index()
                .rename(columns={'BiomarkerStatus': 'pdl1_status'})
                )

            # Process PDL1 staining 
            pdl1_staining_df = (
                df_filtered
                .query('BiomarkerName == "PDL1"')
                .query('BiomarkerStatus == "PD-L1 positive"')
                .groupby('PatientID')['PercentStaining']
                .apply(lambda x: x.map(self.PDL1_PERCENT_STAINING_MAPPING))
                .groupby('PatientID')
                .agg('max')
                .to_frame(name = 'pdl1_ordinal_value')
                .reset_index()
                )
            
            # Create reverse mapping to convert back to percentage strings
            reverse_pdl1_dict = {v: k for k, v in self.PDL1_PERCENT_STAINING_MAPPING.items()}
            pdl1_staining_df['pdl1_percent_staining'] = pdl1_staining_df['pdl1_ordinal_value'].map(reverse_pdl1_dict)
            pdl1_staining_df = pdl1_staining_df.drop(columns = ['pdl1_ordinal_value'])

            # Merge dataframes
            final_df = pd.merge(pdl1_df, pdl1_staining_df, on = 'PatientID', how = 'left')
            final_df = pd.merge(final_df, fgfr_df, on = 'PatientID', how = 'outer')

            final_df['pdl1_status'] = final_df['pdl1_status'].astype('category')
            final_df['fgfr_status'] = final_df['fgfr_status'].astype('category')


            staining_dtype = pd.CategoricalDtype(
                categories = ['0%', '< 1%', '1%', '2% - 4%', '5% - 9%', '10% - 19%',
                              '20% - 29%', '30% - 39%', '40% - 49%', '50% - 59%',
                              '60% - 69%', '70% - 79%', '80% - 89%', '90% - 99%', '100%'],
                              ordered = True
                              )
            
            final_df['pdl1_percent_staining'] = final_df['pdl1_percent_staining'].astype(staining_dtype)

            # Check for duplicate PatientIDs
            if len(final_df) > final_df['PatientID'].nunique():
                logging.error(f"Duplicate PatientIDs found")
                return None

            logging.info(f"Successfully processed Enhanced_AdvUrothelialBiomarkers.csv file with final shape: {final_df.shape} and unique PatientIDs: {(final_df['PatientID'].nunique())}")
            self.biomarkers_df = final_df
            return final_df

        except Exception as e:
            logging.error(f"Error processing Enhanced_AdvUrothelialBiomarkers.csv file: {e}")
            return None
        
    def process_ecog(self,
                     file_path: str,
                     index_date_df: pd.DataFrame,
                     index_date_column: str, 
                     days_before: int = 90,
                     days_after: int = 0) -> pd.DataFrame:
        """
        Processes ECOG.csv to determine patient ECOG scores relative to a reference index date.
        For each patient, finds:
        1. The ECOG score closest to index date (selecting higher score in case of ties)
        2. Whether ECOG newly increased to ≥2 from 0-1 in the prior 6 months

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
        
        Returns
        -------
        pd.DataFrame
            Processed DataFrame containing:
            - PatientID : unique patient identifier
            - ecog_index : ECOG score (0-5) closest to index date, categorical
            - ecog_newly_gte2 : boolean indicating if ECOG increased from 0-1 to ≥2 in 6 months before index

        Notes
        ------
        When multiple ECOG scores are equidistant to index date, selects higher score
        Uses fixed 6-month lookback for newly_gte2 calculation regardless of days_before
        Duplicate PatientIDs are logged as warnings if found
        Processed DataFrame is stored in self.ecog_df
        """

        # Input validation
        if not isinstance(index_date_df, pd.DataFrame) or 'PatientID' not in index_date_df.columns:
            raise ValueError("index_date_df must be a DataFrame containing 'PatientID' column")
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
            df_filtered = df[
                (df['index_to_ecog'] <= days_after) & 
                (df['index_to_ecog'] >= -days_before)].copy()
            window_desc = f"-{days_before} to +{days_after} days from index date"

            logging.info(f"After applying window period of {window_desc}, "f"remaining records: {df_filtered.shape}, "f"unique PatientIDs: {df_filtered['PatientID'].nunique()}")

            # Find EcogValue closest to index date within specified window periods
            ecog_index_df = (
                df_filtered
                .assign(abs_days_to_index = lambda x: abs(x['index_to_ecog']))
                .sort_values(
                    by=['PatientID', 'abs_days_to_index', 'EcogValue'], 
                    ascending=[True, True, False])
                .groupby('PatientID')
                .first()
                .reset_index()
                [['PatientID', 'EcogValue']]
                .rename(columns = {'EcogValue': 'ecog_index'})
                .assign(
                    ecog_index = lambda x: x['ecog_index'].astype(pd.CategoricalDtype(categories = [0, 1, 2, 3, 4, 5], ordered = True))
                    )
                )
            
            # Find EcogValue newly greater than or equal to 2 by time of index date with 6 month look back using pre-specified days_after
            # First get 6-month window data 
            df_6month = df[
                    (df['index_to_ecog'] <= days_after) & 
                    (df['index_to_ecog'] >= -180)].copy()
            
            # Create flag for ECOG newly greater than or equal to 2
            ecog_newly_gte2_df = (
                df_6month
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

            # Merge back with df_filtered
            unique_patient_df = df[['PatientID']].drop_duplicates()
            final_df = pd.merge(unique_patient_df, ecog_index_df, on = 'PatientID', how = 'left')
            final_df = pd.merge(final_df, ecog_newly_gte2_df, on = 'PatientID', how = 'left')
            final_df['ecog_newly_gte2'] = final_df['ecog_newly_gte2'].astype('boolean')


            # Check for duplicate PatientIDs
            if len(final_df) > final_df['PatientID'].nunique():
                logging.error(f"Duplicate PatientIDs found")
                return None
                
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
                       weight_days_after: int = 0,
                       vital_summary_lookback: int = 180) -> pd.DataFrame:
        """
        Processes Vitals.csv to determine patient BMI, weight, change in weight, and vital sign abnormalities
        within a specified time window relative to an index date. 

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
        weight_days_after : int, optional
            Number of days after the index date to include for weight and BMI calculations. Also used as the end point for 
            vital sign abnormalities and weight change calculations. Must be >= 0. Default: 0
        vital_summary_lookback : int, optional
            Number of days before index date to assess for weight change, hypotension, tachycardia, and fever. Must be >= 0. Default: 180
        
        Returns
        -------
        pd.DataFrame
            Processed DataFrame containing:
            - PatientID : unique patient identifier
            - weight : float, weight in kg closest to index date within specified window (index_date - weight_days_before) 
              to (index_date + weight_days_after)
            - bmi : float, BMI closest to index date within specified window (index_date - weight_days_before) 
              to (index_date + weight_days_after)
            - percent_change_weight : float, percentage change in weight over period from (index_date - vital_summary_lookback) 
              to (index_date + weight_days_after), calculated as ((end_weight - start_weight) / start_weight) * 100
            - hypotension : bool, True if ystolic blood pressure <90 mmHg on ≥2 separate readings between (index_date - vital_summary_lookback) 
              and (index_date + weight_days_after)
            - tachycardia : bool, True if heart rate >100 bpm on ≥2 separate readings between (index_date - vital_summary_lookback) 
              and (index_date + weight_days_after)
            - fevers : bool, True if temperature >38°C on ≥2 separate readings between (index_date - vital_summary_lookback) 
              and (index_date + weight_days_after)

        Notes
        -----
        BMI is calculated using weight closest to index date within specified window while height outside the specified window may be used. The equation used: weight (kg)/height (m)^2
        Vital sign thresholds:
            * Hypotension: systolic BP <90 mmHg
            * Tachycardia: HR >100 bpm
            * Fever: temperature >38°C
        Duplicate PatientIDs are logged as warnings but retained in output
        Results are stored in self.vitals_df attribute
        """

        # Input validation
        if not isinstance(index_date_df, pd.DataFrame) or 'PatientID' not in index_date_df.columns:
            raise ValueError("index_date_df must be a DataFrame containing 'PatientID' column")
        if not index_date_column or index_date_column not in index_date_df.columns:
            raise ValueError(f"Column '{index_date_column}' not found in index_date_df")
        
        if not isinstance(weight_days_before, int) or weight_days_before < 0:
                raise ValueError("weight_days_before must be a non-negative integer")
        if not isinstance(weight_days_after, int) or weight_days_after < 0:
            raise ValueError("weight_days_after must be a non-negative integer")
        if not isinstance(vital_summary_lookback, int) or vital_summary_lookback < 0:
            raise ValueError("vital_summary_lookback must be a non-negative integer")

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
            weight_df = df.query('Test == "body weight"')
            mask_needs_imputation = weight_df['TestResultCleaned'].isna() & weight_df['TestResult'].notna()
            
            imputed_weights = weight_df.loc[mask_needs_imputation, 'TestResult'].apply(
                lambda x: x/2.2046 if x > 100  # Convert to kg if likely lbs (>100)
                else x if x < 60  # Keep as is if likely kg (<60)
                else None  # Leave as null if ambiguous
                )
            
            weight_df.loc[mask_needs_imputation, 'TestResultCleaned'] = imputed_weights
            weight_df = weight_df.query('TestResultCleaned > 0')
            
            df_weight_filtered = weight_df[
                (weight_df['index_to_vital'] <= weight_days_after) & 
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
                (weight_df['index_to_vital'] <= weight_days_after) & 
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

            # Calculate hypotension 
            df_summary_filtered = df[
                (df['index_to_vital'] <= weight_days_after) & 
                (df['index_to_vital'] >= -vital_summary_lookback)].copy()
            
            hypotension_df = (
                df_summary_filtered
                .query("Test == 'systolic blood pressure'")
                .sort_values(['PatientID', 'TestDate'])
                .groupby('PatientID')
                .agg({
                    'TestResult': lambda x: (
                        sum(x < 90) >= 2) # True if 2 readings of systolic <90
                })
                .reset_index()
                .rename(columns = {'TestResult': 'hypotension'})
                )

            # Calculate tachycardia
            tachycardia_df = (
                df_summary_filtered 
                .query("Test == 'heart rate'")
                .sort_values(['PatientID', 'TestDate'])
                .groupby('PatientID')
                .agg({
                    'TestResult': lambda x: (
                        sum(x > 100) >= 2) # True if 2 readings of heart rate >100 
                })
                .reset_index()
                .rename(columns = {'TestResult': 'tachycardia'})
                )

            # Calculate fevers 
            fevers_df = (
                df_summary_filtered 
                .query("Test == 'body temperature'")
            )

            # Any temperature >45 is presumed F, otherwise C
            fevers_df.loc[:, 'TestResult'] = np.where(fevers_df['TestResult'] > 45,
                                                    (fevers_df['TestResult'] - 32) * 5/9,
                                                    fevers_df['TestResult'])

            fevers_df = (
                fevers_df
                .sort_values(['PatientID', 'TestDate'])
                .groupby('PatientID')
                .agg({
                    'TestResult': lambda x: sum(x >= 38) >= 2 # True if 2 readings of temperature >38C
                })
                .reset_index()
                .rename(columns={'TestResult': 'fevers'})
            )

            # Merge dataframes 
            final_df = pd.merge(weight_index_df, change_weight_df, on = 'PatientID', how = 'outer')
            final_df = pd.merge(final_df, hypotension_df, on = 'PatientID', how = 'outer')
            final_df = pd.merge(final_df, tachycardia_df, on = 'PatientID', how = 'outer')
            final_df = pd.merge(final_df, fevers_df, on = 'PatientID', how = 'outer')

            boolean_columns = ['hypotension', 'tachycardia', 'fevers']
            for col in boolean_columns:
                final_df[col] = final_df[col].astype('boolean')
            
            final_df = final_df.fillna({'hypotension': False,
                                        'tachycardia': False, 
                                        'fevers': False})
            
            # Check for duplicate PatientIDs
            if len(final_df) > final_df['PatientID'].nunique():
                logging.error(f"Duplicate PatientIDs found")
                return None

            logging.info(f"Successfully processed Vitals.csv file with final shape: {final_df.shape} and unique PatientIDs: {(final_df['PatientID'].nunique())}")
            self.vitals_df = final_df
            return final_df

        except Exception as e:
            logging.error(f"Error processing Vitals.csv file: {e}")
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
            Processed DataFrame containing:
            - PatientID : unique patient identifier

            Baseline values (closest to index date within days_before/days_after window):
            - hemoglobin : float, g/dL
            - wbc : float, K/uL
            - platelet : float, 10^9/L
            - creatinine : float, mg/dL
            - bun : float, mg/dL
            - chloride : float, mmol/L
            - bicarb : float, mmol/L
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
        Duplicate PatientIDs are logged as warnings but retained in output
        Results are stored in self.labs_df attribute
        """

        # Input validation
        if not isinstance(index_date_df, pd.DataFrame) or 'PatientID' not in index_date_df.columns:
            raise ValueError("index_date_df must be a DataFrame containing 'PatientID' column")
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
            df = df[df['LOINC'].isin(all_loinc_codes)].copy()

            # Map LOINC codes to lab names
            for lab_name, loinc_codes in self.LOINC_MAPPINGS.items():
                mask = df['LOINC'].isin(loinc_codes)
                df.loc[mask, 'lab_name'] = lab_name

            # Impute missing hemoglobin 
            mask = df.query('lab_name == "hemoglobin" and TestResultCleaned.isna() and TestResult.notna()').index
            df.loc[mask, 'TestResultCleaned'] = pd.to_numeric(
                df.loc[mask, 'TestResult']
                .str.replace('L', '')
                .str.strip(),
                errors = 'coerce'
                )

            # Impute missing wbc
            mask = df.query('lab_name == "wbc" and TestResultCleaned.isna() and TestResult.notna()').index
            df.loc[mask, 'TestResultCleaned'] = (
                pd.to_numeric(df.loc[mask, 'TestResult'], errors = 'coerce')
                .where(lambda x: (x >= 2) & (x <= 15))
                )
            
            # Impute missing platelets 
            mask = df.query('lab_name == "platelet" and TestResultCleaned.isna() and TestResult.notna()').index
            df.loc[mask, 'TestResultCleaned'] = (
                pd.to_numeric(df.loc[mask, 'TestResult'], errors = 'coerce')
                .where(lambda x: (x >= 50) & (x <= 450))
                )
            
            # Correct units for hemoglobin, WBC, and platelets
            # Convert 10*3/L values
            mask = (
                (df['TestUnits'] == '10*3/L') & 
                (df['lab_name'].isin(['wbc', 'platelet']))
            )
            df.loc[mask, 'TestResultCleaned'] = df.loc[mask, 'TestResultCleaned'] * 1000000

            # Convert g/uL values 
            mask = (
                (df['TestUnits'] == 'g/uL') & 
                (df['lab_name'] == 'hemoglobin')
            )
            df.loc[mask, 'TestResultCleaned'] = df.loc[mask, 'TestResultCleaned'] / 100000   

            # Impute missing creatinine 
            mask = df.query('lab_name == "creatinine" and TestResultCleaned.isna() and TestResult.notna()').index
            df.loc[mask, 'TestResultCleaned'] = (
                pd.to_numeric(df.loc[mask, 'TestResult'], errors = 'coerce')
                .where(lambda x: (x >= 0.3) & (x <= 3))
                )
            
            # Impute missing bun
            mask = df.query('lab_name == "bun" and TestResultCleaned.isna() and TestResult.notna()').index
            df.loc[mask, 'TestResultCleaned'] = (
                pd.to_numeric(df.loc[mask, 'TestResult'], errors = 'coerce')
                .where(lambda x: (x >= 5) & (x <= 50))
                )
            
            # Impute missing chloride 
            mask = df.query('lab_name == "chloride" and TestResultCleaned.isna() and TestResult.notna()').index
            df.loc[mask, 'TestResultCleaned'] = (
                pd.to_numeric(df.loc[mask, 'TestResult'], errors = 'coerce')
                .where(lambda x: (x >= 80) & (x <= 120))
                )
            
            # Impute missing bicarb 
            mask = df.query('lab_name == "bicarb" and TestResultCleaned.isna() and TestResult.notna()').index
            df.loc[mask, 'TestResultCleaned'] = (
                pd.to_numeric(df.loc[mask, 'TestResult'], errors = 'coerce')
                .where(lambda x: (x >= 15) & (x <= 35))
                )

            # Impute missing potassium 
            mask = df.query('lab_name == "potassium" and TestResultCleaned.isna() and TestResult.notna()').index
            df.loc[mask, 'TestResultCleaned'] = (
                pd.to_numeric(df.loc[mask, 'TestResult'], errors = 'coerce')
                .where(lambda x: (x >= 2.5) & (x <= 6))
                )
            
            # Impute missing calcium 
            mask = df.query('lab_name == "calicum" and TestResultCleaned.isna() and TestResult.notna()').index
            df.loc[mask, 'TestResultCleaned'] = (
                pd.to_numeric(df.loc[mask, 'TestResult'], errors = 'coerce')
                .where(lambda x: (x >= 7) & (x <= 14))
                )
            
            # Impute missing alp
            mask = df.query('lab_name == "alp" and TestResultCleaned.isna() and TestResult.notna()').index
            df.loc[mask, 'TestResultCleaned'] = (
                pd.to_numeric(df.loc[mask, 'TestResult'], errors = 'coerce')
                .where(lambda x: (x >= 40) & (x <= 500))
                )
            
            # Impute missing ast
            mask = df.query('lab_name == "ast" and TestResultCleaned.isna() and TestResult.notna()').index
            df.loc[mask, 'TestResultCleaned'] = pd.to_numeric(
                df.loc[mask, 'TestResult']
                .str.replace('<', '')
                .str.strip(),
                errors = 'coerce'
                )
            
            # Impute missing alt
            mask = df.query('lab_name == "alt" and TestResultCleaned.isna() and TestResult.notna()').index
            df.loc[mask, 'TestResultCleaned'] = pd.to_numeric(
                df.loc[mask, 'TestResult']
                .str.replace('<', '')
                .str.strip(),
                errors = 'coerce'
                )
            
            # Impute missing total_bilirbuin
            mask = df.query('lab_name == "total_bilirubin" and TestResultCleaned.isna() and TestResult.notna()').index
            df.loc[mask, 'TestResultCleaned'] = pd.to_numeric(
                df.loc[mask, 'TestResult']
                .str.replace('<', '')
                .str.strip(),
                errors = 'coerce'
                )
            
            # Impute missing albumin
            mask = df.query('lab_name == "albumin" and TestResultCleaned.isna() and TestResult.notna()').index
            df.loc[mask, 'TestResultCleaned'] = (
                pd.to_numeric(df.loc[mask, 'TestResult'], errors = 'coerce')
                .where(lambda x: (x >= 1) & (x <= 6)) * 10
                )

            # Filter for desired window period for baseline labs
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
                                        1)[0]
                    if (x['TestResultCleaned'].notna().sum() > 1 and # at least 2 non-NaN lab values
                        x['index_to_lab'].notna().sum() > 1 and # at least 2 non-NaN time points
                        len(x['index_to_lab'].unique()) > 1)  # time points are not all the same
                    else np.nan)
                .reset_index()
                .pivot(index = 'PatientID', columns = 'lab_name', values = 0)
                .rename_axis(columns = None)
                .rename(columns = lambda x: f'{x}_slope')
                .reset_index()
                )
            
            # Merging lab dataframes 
            final_df = pd.merge(lab_df, max_df, on = 'PatientID', how = 'outer')
            final_df = pd.merge(final_df, min_df, on = 'PatientID', how = 'outer')
            final_df = pd.merge(final_df, std_df, on = 'PatientID', how = 'outer')
            final_df = pd.merge(final_df, slope_df, on = 'PatientID', how = 'outer')

            # Check for duplicate PatientIDs
            if len(final_df) > final_df['PatientID'].nunique():
                logging.error(f"Duplicate PatientIDs found")
                return None

            logging.info(f"Successfully processed Lab.csv file with final shape: {final_df.shape} and unique PatientIDs: {(final_df['PatientID'].nunique())}")
            return final_df

        except Exception as e:
            logging.error(f"Error processing Lab.csv file: {e}")
            return None
            

        