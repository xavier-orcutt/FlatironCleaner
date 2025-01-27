import pandas as pd
import numpy as np
import logging

logging.basicConfig(
    level = logging.INFO,                                 
    format = '%(asctime)s - %(levelname)s - %(message)s'  
)

class DataProcessorUrothelial:
    
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

    def process_enhanced_adv(self, file_path: str, drop_dates: bool = True) -> pd.DataFrame: 
        """
        Process Enhanced_AdvUrothelial.csv file by cleaning data types and calculating time-based variables.

        Parameters:
            file_path (str): Path to Enhanced_AdvUrothelial.csv file
            drop_dates (bool, optional): If True, drops date columns after calculations. Defaults to True.

        Returns:
            pd.DataFrame: Processed DataFrame with:
                - Categorical columns (PrimarySite, DiseaseGrade, GroupStage, TStage, NStage, 
                  MStage, SmokingStatus, SurgeryType)
                - Boolean column (Surgery)
                - Calculated columns:
                    * days_diagnosis_to_advanced: Days between initial and advanced diagnosis
                    * advanced_diagnosis_year: Year of advanced diagnosis (as category)
                    * days_diagnosis_to_surgery: Days between initial diagnosis and surgery
                - Original date columns dropped if drop_dates = True

        Notes:
            - Checks for and logs duplicate PatientIDs
            - Stores processed DataFrame in self.enhanced_df
        """
        try:
            df = pd.read_csv(file_path)
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
                        reference_dates_df: pd.DataFrame = None,
                        date_column: str = None,
                        drop_state: bool = True) -> pd.DataFrame:
        """
        Process Demographics.csv file by cleaning data types, calculating age, and mapping states to regions.
    
        Parameters:
            file_path (str): Path to demographics CSV file
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
            - Stores processed DataFrame in self.enhanced_df
        """
        try:
            df = pd.read_csv(file_path)
            logging.info(f"Successfully read Demographics.csv file with shape: {df.shape} and unique PatientIDs: {(df.PatientID.nunique())}")

            # Convert BirthYear to int64
            df['BirthYear'] = df['BirthYear'].astype('int64')

            # Convert categorical columns; Race, Ethnicity, and State will be converted later
            df['Gender'] = df['Gender'].astype('category')
        
            # Calculate age if reference dates are provided
            if reference_dates_df is not None and date_column is not None:

                # Convert date column to datetime if it's not already
                reference_dates_df[date_column] = pd.to_datetime(reference_dates_df[date_column])

                # Merge Demographics.csv with reference_dates_df
                df = pd.merge(
                    df,
                    reference_dates_df[['PatientID', date_column]], 
                    on = 'PatientID',
                    how = 'left'
                )
            
                # Calculate age
                df['age'] = df[date_column].dt.year - df['BirthYear']

                # Add age validation
                mask_invalid_age = (df['age'] < 18) | (df['age'] > 120)
                if mask_invalid_age.any():
                    logging.warning(f"Found {mask_invalid_age.sum()} ages outside valid range (18-120)")

                # Drop the date column and BirthYear after age calculation
                df = df.drop(columns = [date_column, 'BirthYear'])

            # If Race == 'Hispanic or Latino', fill 'Hispanic or Latino' for Ethnicity
            df['Ethnicity'] = np.where(df['Race'] == 'Hispanic or Latino', 'Hispanic or Latino', df['Ethnicity'])

            # If Race == 'Hispanic or Latino' replace with Nan
            df['Race'] = np.where(df['Race'] == 'Hispanic or Latino', np.nan, df['Race'])

            df[['Race', 'Ethnicity']] = df[['Race', 'Ethnicity']].astype('category')

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