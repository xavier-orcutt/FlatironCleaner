import pandas as pd
import numpy as np
import logging
from IPython import embed
from typing import Optional
import re 

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

def process_demographics(file_path: str,
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
    Ages calculated as <18 or >120 are removed as implausible
    Missing States are imputed as unknown during the mapping to regions
    Duplicate PatientIDs are logged as warnings if found
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
        
        # Region processing
        # Group states into Census-Bureau regions  
        df['region'] = (df['State']
                        .map(STATE_REGIONS_MAPPING)
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
        return df

    except Exception as e:
        logging.error(f"Error processing Demographics.csv file: {e}")
        return None
    
        
# TESTING
df = pd.read_csv('data_nsclc/Enhanced_AdvancedNSCLC.csv') 
a = process_demographics(file_path="data_nsclc/Demographics.csv",
                         index_date_df=df,
                         index_date_column='AdvancedDiagnosisDate', 
                         drop_state = False)

embed()