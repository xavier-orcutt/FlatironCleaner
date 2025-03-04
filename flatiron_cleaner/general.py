import pandas as pd
import numpy as np
import logging 
import re 
from typing import Optional

logging.basicConfig(
    level = logging.INFO,                                 
    format = '%(asctime)s - %(levelname)s - %(message)s'  
)

class DataProcessorGeneral:

    def __init__(self):
        self.mortality_df = None

    def process_mortality(self, 
                          file_path: str,
                          index_date_df: pd.DataFrame,
                          index_date_column: str,
                          supplementary_files: dict = None,
                          drop_dates: bool = True) -> Optional[pd.DataFrame]:
        """
        Processes Enhanced_Mortality_V2.csv by cleaning data types, calculating time from index date to death/censor, and determining mortality events. 

        Parameters
        ----------
        file_path : str
            Path to Enhanced_Mortality_V2.csv file
        index_date_df : pd.DataFrame
            DataFrame containing unique PatientIDs and their corresponding index dates. Only mortality data for PatientIDs present in this DataFrame will be processed
        index_date_column : str
            Column name in index_date_df containing the index date
        supplementary_files : dict, optional
            Dictionary of paths to additional supplementary files and the names of the columns with the dates of interest for calculating last EHR activity.
            Example: {'path/to/your/Visit.csv': ['VisitDate'], 'path/to/your/Progression.csv': ['ProgresionDate', 'LastClinicNoteDate']}
        drop_dates : bool, default = True
            If True, drops date columns (index_date_column, DateOfDeath, last_ehr_date)   
        
        Returns
        -------
        pd.DataFrame or None
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
          supplementary files 
        - If no supplementary files are provided or a patient has no activity in 
          supplementary files, duration may be null for censored patients

        Supplementary files: 
        - Supplementary files can include any csv file with a date that could be used to calculate 
        last EHR activity. Common files recommended by Flatiron include visit, biomarkers, orals, 
        progression, and sites of metastasis. See Flatiron documentation for file recomendations by
        cancer type. 
        - The supplementary files must include a PatientID column, but don't have to have a unique PatientID
        per row. 

        Output handling: 
        - Duplicate PatientIDs are logged as warnings if found but retained in output
        - Processed DataFrame is stored in self.mortality_df
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
        
        index_date_df = index_date_df.copy()
        # Rename all columns from index_date_df except PatientID to avoid conflicts with merging and processing 
        for col in index_date_df.columns:
            if col != 'PatientID':  # Keep PatientID unchanged for merging
                index_date_df.rename(columns={col: f'imported_{col}'}, inplace=True)

        # Update index_date_column name
        index_date_column = f'imported_{index_date_column}'

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
            if supplementary_files is None:
                logging.info("WARNING: At least one supplementary file (eg., visit, biomarkers, progression, etc) is needed to calculate duration for those with a missing death date")
            else: 
                file_counter = 0 
        
                # For loop to go through each file in the supplementary file dictionary 
                for file_path, date_columns in supplementary_files.items():
                    try:
                        file_counter += 1
                        supp_df = pd.read_csv(file_path)

                        if not isinstance(supp_df, pd.DataFrame):
                            raise ValueError("supp_df must be a pandas DataFrame")
                        if 'PatientID' not in supp_df.columns:
                            raise ValueError("supp_df must contain a 'PatientID' column")

                        logging.info(f"Successfully read supplementary file: {file_path} with shape: {supp_df.shape}")

                       # Convert all date columns to datetime
                        valid_date_columns = []

                        # Process each date column in this file
                        for date_column in date_columns:
                            if date_column in supp_df.columns:
                                # Convert date column to datetime
                                supp_df[date_column] = pd.to_datetime(supp_df[date_column], errors='coerce')
                                valid_date_columns.append(date_column)
                            else:
                                logging.warning(f"Date column '{date_column}' not found in {file_path}")

                        # Skip if no valid date columns found
                        if not valid_date_columns:
                            logging.warning(f"No valid date columns found in {file_path} - skipping")
                            continue
                        
                        max_date_df = (
                            supp_df
                            .query("PatientID in @index_date_df.PatientID")
                            .assign(max_date=lambda x: x[valid_date_columns].max(axis=1))
                            .groupby('PatientID')['max_date']
                            .max()
                            .to_frame(name=f'max_date_{file_counter}')
                            .reset_index()
                        )
                        patient_last_dates.append(max_date_df)
                
                    except Exception as e:
                        logging.error(f"Error processing supplementary file {file_path}: {str(e)}")

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
            self.mortality_df = final_df
            return final_df
        
        except Exception as e:
            logging.error(f"Error processing Enhanced_Mortality_V2.csv file: {e}")
            return None