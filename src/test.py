import pandas as pd
import numpy as np
import logging
from IPython import embed

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

def process_mortality(
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
            Column name in index_date_df containing the reference dates
        df_merge_type : str, default='left'
            Merge type for pd.merge(index_date_df, mortality_data)
        visit_path : str
            Path to Visit.csv file
        telemedicine_path : str
            Path to Telemedicine.csv file
        biomarker_path : str
            Path to Enhanced_AdvUrothelialBiomarkers.csv file
        oral_path : str
            Path to Enhanced_AdvUrothelial_Orals.csv file
        progression_path : str
            Path to Progression.csv file
        drop_dates : bool, default = True
            If True, drops date columns (DeathDate, index_date_column, last_ehr_date)   
        
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

            # Calculate last activity 
            if all(path is None for path in [visit_path, telemedicine_path, biomarker_path, oral_path, progression_path]):
                logging.info("WARNING: At least one of visit_path, telemedicine_path, biomarker_path, oral_path, or progression_path must be provided to calculate duration for those with a missing death date")
                return df_death

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

                    final_df = pd.merge(df_death, single_date, on = 'PatientID', how = 'left')
                    return final_df

        except Exception as e:
            logging.error(f"Error processing mortality file: {e}")
            return None
        

index_date_df = pd.read_csv("data/Enhanced_AdvUrothelial.csv")



a = process_mortality(file_path="data/Enhanced_Mortality_V2.csv",
                          index_date_df=index_date_df,
                           index_date_column='AdvancedDiagnosisDate',
                           df_merge_type='left',
                           drop_dates = True)

embed()