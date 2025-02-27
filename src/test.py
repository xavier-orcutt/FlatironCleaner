import pandas as pd
import numpy as np
import logging
from IPython import embed
from typing import Optional
import re 

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

def process_mortality(file_path: str,
                      index_date_df: pd.DataFrame,
                      index_date_column: str,
                      visit_path: str = None, 
                      telemedicine_path: str = None, 
                      biomarkers_path: str = None, 
                      oral_path: str = None,
                      progression_path: str = None,
                      drop_dates: bool = True) -> pd.DataFrame:
        """
        Processes Enhanced_Mortality_V2.csv by cleaning data types, calculating time from index date to death/censor, and determining mortality events. 

        Parameters
        ----------
        file_path : str
            Path to Enhanced_Mortality_V2.csv file
        index_date_df : pd.DataFrame
            DataFrame containing PatientID and index dates. Only mortality data for PatientIDs present in this DataFrame will be processed
        index_date_column : str
            Column name in index_date_df containing the index date
        visit_path : str
            Path to Visit.csv file, used to determine last EHR activity date for censored patients
        telemedicine_path : str
            Path to Telemedicine.csv file, used to determine last EHR activity date for censored patients
        biomarkers_path : str
            Path to Enhanced_AdvNSCLCBiomarkers.csv file, used to determine last EHR activity date for censored patients
        oral_path : str
            Path to Enhanced_AdvNSCLC_Orals.csv file, used to determine last EHR activity date for censored patients
        progression_path : str
            Path to Enhanced_AdvNSCLC_Progression.csv file, used to determine last EHR activity date for censored patients
        drop_dates : bool, default = True
            If True, drops date columns (index_date_column, DateOfDeath, last_ehr_date)   
        
        Returns
        -------
        pd.DataFrame
            - PatientID : object
                unique patient identifier
            - duration : float
                days from index date to death or censor 
            - event : Int64
                mortality status (1 = death, 0 = censored)

        If drop_dates=False, the DataFrame will also include:
            - {index_date_column} : datetime64
                The index date for each patient
            - DateOfDeath : datetime64
                Date of death (if available)
            - last_ehr_activity : datetime64
                Most recent EHR activity date (if available from supplementary files)
                
        Notes
        ------
        Death date handling:
        - Known death date: 'event' = 1, 'duration' = days from index to death
        - No death date: 'event' = 0, 'duration' = days from index to last EHR activity
        
        Death date imputation for incomplete dates:
        - Missing day: Imputed to 15th of the month
        - Missing month and day: Imputed to July 1st of the year
    
        Censoring logic:
        - Patients without death dates are censored at their last EHR activity
        - Last EHR activity is determined as the maximum date across all provided
          supplementary files (visit, telemedicine, biomarkers, oral, progression)
        - If no supplementary files are provided or a patient has no activity in 
          supplementary files, duration may be null for censored patients
        
        Duplicate PatientIDs are logged as warnings if found but retained in output
        Processed DataFrame is stored in self.mortality_df
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
            logging.info(f"Successfully read Enhanced_Mortality_V2.csv file with shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")

            # When only year is available: Impute to July 1st (mid-year)
            df['DateOfDeath'] = np.where(df['DateOfDeath'].str.len() == 4, df['DateOfDeath'] + '-07-01', df['DateOfDeath'])

            # When only month and year are available: Impute to the 15th day of the month
            df['DateOfDeath'] = np.where(df['DateOfDeath'].str.len() == 7, df['DateOfDeath'] + '-15', df['DateOfDeath'])

            df['DateOfDeath'] = pd.to_datetime(df['DateOfDeath'])

            # Process index dates and merge
            index_date_df[index_date_column] = pd.to_datetime(index_date_df[index_date_column])
            df = pd.merge(
                index_date_df[['PatientID', index_date_column]],
                df,
                on = 'PatientID',
                how = 'left'
            )
            logging.info(f"Successfully merged Enhanced_Mortality_V2.csv df with index_date_df resulting in shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")
                
            # Create event column
            df['event'] = df['DateOfDeath'].notna().astype('Int64')

            # Initialize final dataframe
            final_df = df.copy()

            # Create a list to store all last activity date dataframes
            patient_last_dates = []

            # Determine last EHR data
            if all(path is None for path in [visit_path, telemedicine_path, biomarkers_path, oral_path, progression_path]):
                logging.info("WARNING: At least one of visit_path, telemedicine_path, biomarkers_path, oral_path, or progression_path must be provided to calculate duration for those with a missing death date")
            else: 
                # Process visit and telemedicine data
                if visit_path is not None or telemedicine_path is not None:
                    visit_dates = []
                    try:
                        if visit_path is not None:
                            df_visit = pd.read_csv(visit_path)
                            df_visit['VisitDate'] = pd.to_datetime(df_visit['VisitDate'])
                            visit_dates.append(df_visit[['PatientID', 'VisitDate']])
                            
                        if telemedicine_path is not None:
                            df_tele = pd.read_csv(telemedicine_path)
                            df_tele['VisitDate'] = pd.to_datetime(df_tele['VisitDate'])
                            visit_dates.append(df_tele[['PatientID', 'VisitDate']])
                        
                        if visit_dates:
                            df_visit_combined = pd.concat(visit_dates)
                            df_visit_max = (
                                df_visit_combined
                                .query("PatientID in @index_date_df.PatientID")
                                .groupby('PatientID')['VisitDate']
                                .max()
                                .to_frame(name='last_visit_date')
                                .reset_index()
                            )
                            patient_last_dates.append(df_visit_max)
                    except Exception as e:
                        logging.error(f"Error processing Visit.csv or Telemedicine.csv: {e}")
                                            
                # Process biomarkers data
                if biomarkers_path is not None:
                    try: 
                        df_biomarkers = pd.read_csv(biomarkers_path)
                        df_biomarkers['SpecimenCollectedDate'] = pd.to_datetime(df_biomarkers['SpecimenCollectedDate'])

                        df_biomarkers_max = (
                            df_biomarkers
                            .query("PatientID in @index_date_df.PatientID")
                            .groupby('PatientID')['SpecimenCollectedDate']
                            .max()
                            .to_frame(name='last_biomarker_date')
                            .reset_index()
                        )
                        patient_last_dates.append(df_biomarkers_max)
                    except Exception as e:
                        logging.error(f"Error reading Enhanced_AdvNSCLCBiomarkers.csv file: {e}")

                # Process oral medication data
                if oral_path is not None:
                    try:
                        df_oral = pd.read_csv(oral_path)
                        df_oral['StartDate'] = pd.to_datetime(df_oral['StartDate'])
                        df_oral['EndDate'] = pd.to_datetime(df_oral['EndDate'])

                        df_oral_max = (
                            df_oral
                            .query("PatientID in @index_date_df.PatientID")
                            .assign(max_date=lambda x: x[['StartDate', 'EndDate']].max(axis=1))
                            .groupby('PatientID')['max_date']
                            .max()
                            .to_frame(name='last_oral_date')
                            .reset_index()
                        )
                        patient_last_dates.append(df_oral_max)
                    except Exception as e:
                        logging.error(f"Error reading Enhanced_AdvNSCLC_Orals.csv file: {e}")

                # Process progression data
                if progression_path is not None:
                    try: 
                        df_progression = pd.read_csv(progression_path)
                        df_progression['ProgressionDate'] = pd.to_datetime(df_progression['ProgressionDate'])
                        df_progression['LastClinicNoteDate'] = pd.to_datetime(df_progression['LastClinicNoteDate'])

                        df_progression_max = (
                            df_progression
                            .query("PatientID in @index_date_df.PatientID")
                            .assign(max_date=lambda x: x[['ProgressionDate', 'LastClinicNoteDate']].max(axis=1))
                            .groupby('PatientID')['max_date']
                            .max()
                            .to_frame(name='last_progression_date')
                            .reset_index()
                        )
                        patient_last_dates.append(df_progression_max)
                    except Exception as e:
                        logging.error(f"Error reading Enhanced_AdvNSCLC_Progression.csv file: {e}")

                # Combine all last activity dates
                if patient_last_dates:
                    # Start with the first dataframe
                    combined_dates = patient_last_dates[0]
                    
                    # Merge with any additional dataframes
                    for date_df in patient_last_dates[1:]:
                        combined_dates = pd.merge(combined_dates, date_df, on = 'PatientID', how = 'outer')
                    
                    # Calculate the last activity date across all columns
                    date_columns = [col for col in combined_dates.columns if col != 'PatientID']
                    if date_columns:
                        logging.info(f"The following columns {date_columns} are used to calculate the last EHR date")
                        combined_dates['last_ehr_activity'] = combined_dates[date_columns].max(axis=1)
                        single_date = combined_dates[['PatientID', 'last_ehr_activity']]
                        
                        # Merge with the main dataframe
                        final_df = pd.merge(final_df, single_date, on='PatientID', how='left')
     
            # Calculate duration
            if 'last_ehr_activity' in final_df.columns:
                final_df['duration'] = np.where(
                    final_df['event'] == 0, 
                    (final_df['last_ehr_activity'] - final_df[index_date_column]).dt.days, 
                    (final_df['DateOfDeath'] - final_df[index_date_column]).dt.days
                )
                
                # Drop date variables if specified
                if drop_dates:               
                    final_df = final_df.drop(columns=[index_date_column, 'DateOfDeath', 'last_ehr_activity'])
                       
            else: 
                final_df['duration'] = (final_df['DateOfDeath'] - final_df[index_date_column]).dt.days
            
                # Drop date variables if specified
                if drop_dates:               
                    final_df = final_df.drop(columns=[index_date_column, 'DateOfDeath'])
                
            # Check for duplicate PatientIDs
            if len(final_df) > final_df['PatientID'].nunique():
                duplicate_ids = final_df[final_df.duplicated(subset=['PatientID'], keep=False)]['PatientID'].unique()
                logging.warning(f"Duplicate PatientIDs found: {duplicate_ids}")

            logging.info(f"Successfully processed Enhanced_Mortality_V2.csv file with final shape: {final_df.shape} and unique PatientIDs: {final_df['PatientID'].nunique()}. There are {final_df['duration'].isna().sum()} out of {final_df['PatientID'].nunique()} patients with missing duration values")
            return final_df

        except Exception as e:
            logging.error(f"Error processing Enhanced_Mortality_V2.csv file: {e}")
            return None

# TESTING
df = pd.read_csv('data_nsclc/Enhanced_AdvancedNSCLC.csv')
a = process_mortality(file_path="data_nsclc/Enhanced_Mortality_V2.csv",
                       index_date_df= df,
                       index_date_column= 'AdvancedDiagnosisDate')

embed()