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
            - SurgeryType : type of surgery performed
            - days_diagnosis_to_surgery : days from diagnosis to surgery
            - DiseaseGrade : tumor grade
            - NStage : lymph node staging
            - GroupStage_mod : consolidated overall staging (0-IV, Unknown)
            - TStage_mod : consolidated tumor staging (T0-T4, Ta, Tis, TX, Unknown)
            - MStage_mod : consolidated metastasis staging (M0, M1, MX, Unknown)
            - days_diagnosis_to_advanced : days from initial to advanced diagnosis
            - advanced_diagnosis_year : year of advanced diagnosis (categorical)
            
            Original staging and date columns retained if respective drop_* = False

        Notes
        -----
        - Duplicate PatientIDs are logged as warnings if found
        - Processed DataFrame is stored in self.enhanced_df
        """
        try:
            df = pd.read_csv(file_path)
            logging.info(f"Successfully read Enhanced_AdvUrothelial.csv file with shape: {df.shape} and unique PatientIDs: {(df.PatientID.nunique())}")

            # Filter for specific PatientIDs if provided
            if patient_ids is not None:
                logging.info(f"Filtering for {len(patient_ids)} specific PatientIDs")
                df = df[df['PatientID'].isin(patient_ids)]
                logging.info(f"Successfully filtered Enhanced_AdvUrothelial.csv file with shape: {df.shape} and unique PatientIDs: {(df.PatientID.nunique())}")
        
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
        Processes Demographics.csv by standardizing categorical variables, mapping states 
        to census regions, and calculating age at reference date if provided.

        Parameters
        ----------
        file_path : str
            Path to Demographics.csv file
        patient_ids : list, optional
            List of specific PatientIDs to process. If None, processes all patients
        reference_dates_df : pd.DataFrame, optional
            DataFrame containing PatientID and reference dates for age calculation
        date_column : str, optional
            Column name in reference_dates_df containing dates for age calculation
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
            - age : age at reference date (if reference_dates_df provided)
            - region : US Census Bureau region
            - State : US state (if drop_state=False)
            
        Notes
        -----
        - Duplicate PatientIDs are logged as warnings if found
        - Processed DataFrame is stored in self.demographics_df
        """
        try:
            df = pd.read_csv(file_path)
            logging.info(f"Successfully read Demographics.csv file with shape: {df.shape} and unique PatientIDs: {(df.PatientID.nunique())}")

            # Filter for specific PatientIDs if provided
            if patient_ids is not None:
                logging.info(f"Filtering for {len(patient_ids)} specific PatientIDs")
                df = df[df['PatientID'].isin(patient_ids)]
                logging.info(f"Successfully filtered Demographics.csv file with shape: {df.shape} and unique PatientIDs: {(df.PatientID.nunique())}")

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
            logging.info(f"Successfully read Practice.csv file with shape: {df.shape} and unique PatientIDs: {(df.PatientID.nunique())}")

            # Filter for specific PatientIDs if provided
            if patient_ids is not None:
                logging.info(f"Filtering for {len(patient_ids)} specific PatientIDs")
                df = df[df['PatientID'].isin(patient_ids)]
                logging.info(f"Successfully filtered Practice.csv file with shape: {df.shape} and unique PatientIDs: {(df.PatientID.nunique())}")

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
        
    def process_mortality(self,
                          file_path: str,
                          index_date_df: pd.DataFrame,
                          index_date_column: str,
                          df_merge_type: str = 'left',
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
            DataFrame containing PatientID and reference dates for duration calculation
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
            logging.info(f"Successfully read Enhanced_Mortality_V2.csv file with shape: {df.shape} and unique PatientIDs: {(df.PatientID.nunique())}")

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
                how = df_merge_type
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
        

    