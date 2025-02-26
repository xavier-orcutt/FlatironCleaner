import pandas as pd
import numpy as np
import logging
from IPython import embed
from typing import Optional
import re 

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

def process_labs(file_path: str,
                 index_date_df: pd.DataFrame,
                 index_date_column: str, 
                 additional_loinc_mappings: dict = None,
                 days_before: int = 90,
                 days_after: int = 0,
                 summary_lookback: int = 180) -> pd.DataFrame:
    """
    Processes Lab.csv to determine patient lab values within a specified time window relative to an index date. Returns CBC and CMP values 
    nearest to index date, along with summary statistics (max, min, standard deviation, and slope) calculated over the summary period. 
    Additional lab tests can be included by providing corresponding LOINC code mappings.

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
        - PatientID : object
            unique patient identifier

        Baseline values (closest to index date within days_before/days_after window):
        - hemoglobin : float, g/dL
        - wbc : float, K/uL
        - platelet : float, 10^9/L
        - creatinine : float, mg/dL
        - bun : float, mg/dL
        - sodium : float, mmol/L
        - chloride : float, mmol/L
        - bicarbonate : float, mmol/L
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
    Missing ResultDate is imputed with TestDate
    
    Imputation strategy for lab values:
    - For each lab, missing TestResultCleaned values are imputed from TestResult after removing flags (L, H, <, >)
    - Values outside physiological ranges for each lab are filtered out

    Unit conversion corrections:
    - Hemoglobin: Values in g/uL are divided by 100,000 to convert to g/dL
    - WBC/Platelet: Values in 10*3/L are multiplied by 1,000,000; values in /mm3 or 10*3/mL are multiplied by 1,000
    - Creatinine/BUN/Calcium: Values in mg/L are multiplied by 10 to convert to mg/dL
    - Albumin: Values in mg/dL are multiplied by 100 to convert to g/L; values 1-6 are assumed to be g/dL and multiplied by 10
    
    All PatientIDs from index_date_df are included in the output and values will be NaN for patients without lab values 
    Duplicate PatientIDs are logged as warnings but retained in output 
    Results are stored in self.labs_df attribute
    """
    LOINC_MAPPINGS = {
        'hemoglobin': ['718-7', '20509-6'],
        'wbc': ['26464-8', '6690-2'],
        'platelet': ['26515-7', '777-3', '778-1', '49497-1'],
        'creatinine': ['2160-0', '38483-4'],
        'bun': ['3094-0'],
        'sodium': ['2947-0', '2951-2'],
        'bicarbonate': ['1963-8', '1959-6', '14627-4', '1960-4', '2028-9'],
        'chloride': ['2075-0'],
        'potassium': ['6298-4', '2823-3'],
        'albumin': ['1751-7'],
        'calcium': ['17861-6', '49765-1'],
        'total_bilirubin': ['42719-5', '1975-2'],
        'ast': ['1920-8'],
        'alt': ['1742-6', '1743-4', '1744-2'],
        'alp': ['6768-6']
    }

    # Input validation
    if not isinstance(index_date_df, pd.DataFrame):
        raise ValueError("index_date_df must be a pandas DataFrame")
    if 'PatientID' not in index_date_df.columns:
        raise ValueError("index_date_df must contain a 'PatientID' column")
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
        LOINC_MAPPINGS.update(additional_loinc_mappings)

    try:
        df = pd.read_csv(file_path)
        logging.info(f"Successfully read Lab.csv file with shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")

        df['ResultDate'] = pd.to_datetime(df['ResultDate'])
        df['TestDate'] = pd.to_datetime(df['TestDate'])

        # Impute TestDate for missing ResultDate. 
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
        all_loinc_codes = sum(LOINC_MAPPINGS.values(), [])

        # Filter for LOINC codes 
        df = df[df['LOINC'].isin(all_loinc_codes)]

        # Map LOINC codes to lab names
        for lab_name, loinc_codes in LOINC_MAPPINGS.items():
            mask = df['LOINC'].isin(loinc_codes)
            df.loc[mask, 'lab_name'] = lab_name

        ## CBC PROCESSING ##

        # Hemoglobin: Convert to TestResult to numeric after removing L, H, <, and >; filter for ranges from 3-20; and impute to TestResultCleaned
        mask = df.query('lab_name == "hemoglobin" and TestResultCleaned.isna() and TestResult.notna()').index
        df.loc[mask, 'TestResultCleaned'] = (
            pd.to_numeric(df.loc[mask, 'TestResult'].str.replace(r'[LH<>]', '', regex=True).str.strip(), errors='coerce')
            .where(lambda x: (x >= 3) & (x <= 20))
        )

        # WBC: Convert to TestResult to numeric after removing L, H, <, and >; filter for ranges from 0-40; and impute to TestResultCleaned
        mask = df.query('lab_name == "wbc" and TestResultCleaned.isna() and TestResult.notna()').index
        df.loc[mask, 'TestResultCleaned'] = (
            pd.to_numeric(df.loc[mask, 'TestResult'].str.replace(r'[LH<>]', '', regex=True).str.strip(), errors='coerce')
            .where(lambda x: (x >= 0) & (x <= 40))
        )
        
        # Platelet: Convert to TestResult to numeric after removing L, H, <, and >; filter for ranges from 0-750; and impute to TestResultCleaned
        mask = df.query('lab_name == "platelet" and TestResultCleaned.isna() and TestResult.notna()').index
        df.loc[mask, 'TestResultCleaned'] = (
            pd.to_numeric(df.loc[mask, 'TestResult'].str.replace(r'[LH<>]', '', regex=True).str.strip(), errors='coerce')
            .where(lambda x: (x >= 0) & (x <= 750))
        )
        
        # Hemoglobin conversion correction
        # TestResultCleaned incorrectly stored g/uL values 
        # Example: 12 g/uL was stored as 1,200,000 g/dL instead of 12 g/dL
        # Need to divide by 100,000 to restore correct value
        mask = (
            (df['lab_name'] == 'hemoglobin') & 
            (df['TestUnits'] == 'g/uL')
        )
        df.loc[mask, 'TestResultCleaned'] = df.loc[mask, 'TestResultCleaned'] / 100000 

        # WBC and Platelet conversion correction
        # TestResultCleaned incorrectly stored 10*3/L values 
        # Example: 9 10*3/L was stored as 0.000009 10*9/L instead of 9 10*9/L
        # Need to multipley 1,000,000 to restore correct value
        mask = (
            ((df['lab_name'] == 'wbc') | (df['lab_name'] == 'platelet')) & 
            (df['TestUnits'] == '10*3/L')
        )
        df.loc[mask, 'TestResultCleaned'] = df.loc[mask, 'TestResultCleaned'] * 1000000

        # WBC and Platelet conversion correction
        # TestResultCleaned incorrectly stored /mm3 and 10*3/mL values
        # Example: 9 /mm3 and 9 10*3/mL was stored as 0.009 10*9/L instead of 9 10*9/L
        # Need to multipley 1,000 to restore correct value
        mask = (
            ((df['lab_name'] == 'wbc') | (df['lab_name'] == 'platelet')) & 
            ((df['TestUnits'] == '/mm3') | (df['TestUnits'] == '10*3/mL'))
        )
        df.loc[mask, 'TestResultCleaned'] = df.loc[mask, 'TestResultCleaned'] * 1000

        ## CMP PROCESSING ##

        # Creatinine: Convert to TestResult to numeric after removing L, H, <, and >; filter for ranges from 0-5; and impute to TestResultCleaned 
        mask = df.query('lab_name == "creatinine" and TestResultCleaned.isna() and TestResult.notna()').index
        df.loc[mask, 'TestResultCleaned'] = (
            pd.to_numeric(df.loc[mask, 'TestResult'].str.replace(r'[LH<>]', '', regex=True).str.strip(), errors='coerce')
            .where(lambda x: (x >= 0) & (x <= 5))
        )
        
        # BUN: Convert to TestResult to numeric after removing L, H, <, and >; filter for ranges from 0-100; and impute to TestResultCleaned 
        mask = df.query('lab_name == "bun" and TestResultCleaned.isna() and TestResult.notna()').index
        df.loc[mask, 'TestResultCleaned'] = (
            pd.to_numeric(df.loc[mask, 'TestResult'].str.replace(r'[LH<>]', '', regex=True).str.strip(), errors='coerce')
            .where(lambda x: (x >= 0) & (x <= 100))
        )

        # Sodium: Convert to TestResult to numeric after removing L, H, <, and >; filter for ranges from 110-160; and impute to TestResultCleaned 
        mask = df.query('lab_name == "sodium" and TestResultCleaned.isna() and TestResult.notna()').index
        df.loc[mask, 'TestResultCleaned'] = (
            pd.to_numeric(df.loc[mask, 'TestResult'].str.replace(r'[LH<>]', '', regex=True).str.strip(), errors='coerce')
            .where(lambda x: (x >= 110) & (x <= 160))
        )
        
        # Chloride: Convert to TestResult to numeric after removing L, H, <, and >; filter for ranges from 70-140; and impute to TestResultCleaned 
        mask = df.query('lab_name == "chloride" and TestResultCleaned.isna() and TestResult.notna()').index
        df.loc[mask, 'TestResultCleaned'] = (
            pd.to_numeric(df.loc[mask, 'TestResult'].str.replace(r'[LH<>]', '', regex=True).str.strip(), errors='coerce')
            .where(lambda x: (x >= 70) & (x <= 140))
        )
        
        # Bicarbonate: Convert to TestResult to numeric after removing L, H, <, and >; filter for ranges from 5-50; and impute to TestResultCleaned 
        mask = df.query('lab_name == "bicarbonate" and TestResultCleaned.isna() and TestResult.notna()').index
        df.loc[mask, 'TestResultCleaned'] = (
            pd.to_numeric(df.loc[mask, 'TestResult'].str.replace(r'[LH<>]', '', regex=True).str.strip(), errors='coerce')
            .where(lambda x: (x >= 5) & (x <= 50))
        )

        # Potassium: Convert to TestResult to numeric after removing L, H, <, and >; filter for ranges from 2-8; and impute to TestResultCleaned  
        mask = df.query('lab_name == "potassium" and TestResultCleaned.isna() and TestResult.notna()').index
        df.loc[mask, 'TestResultCleaned'] = (
            pd.to_numeric(df.loc[mask, 'TestResult'].str.replace(r'[LH<>]', '', regex=True).str.strip(), errors='coerce')
            .where(lambda x: (x >= 2) & (x <= 8))
        )
        
        # Calcium: Convert to TestResult to numeric after removing L, H, <, and >; filter for ranges from 5-15; and impute to TestResultCleaned 
        mask = df.query('lab_name == "calcium" and TestResultCleaned.isna() and TestResult.notna()').index
        df.loc[mask, 'TestResultCleaned'] = (
            pd.to_numeric(df.loc[mask, 'TestResult'].str.replace(r'[LH<>]', '', regex=True).str.strip(), errors='coerce')
            .where(lambda x: (x >= 5) & (x <= 15))
        )
        
        # ALP: Convert to TestResult to numeric after removing L, H, <, and >; filter for ranges from 20-3000; and impute to TestResultCleaned
        mask = df.query('lab_name == "alp" and TestResultCleaned.isna() and TestResult.notna()').index
        df.loc[mask, 'TestResultCleaned'] = (
            pd.to_numeric(df.loc[mask, 'TestResult'].str.replace(r'[LH<>]', '', regex=True).str.strip(), errors='coerce')
            .where(lambda x: (x >= 20) & (x <= 3000))
        )
        
        # AST: Convert to TestResult to numeric after removing L, H, <, and >; filter for ranges from 5-2000; and impute to TestResultCleaned
        mask = df.query('lab_name == "ast" and TestResultCleaned.isna() and TestResult.notna()').index
        df.loc[mask, 'TestResultCleaned'] = (
            pd.to_numeric(df.loc[mask, 'TestResult'].str.replace(r'[LH<>]', '', regex=True).str.strip(), errors='coerce')
            .where(lambda x: (x >= 5) & (x <= 2000))
        )
        
        # ALT: Convert to TestResult to numeric after removing L, H, <, and >; filter for ranges from 5-2000; and impute to TestResultCleaned
        mask = df.query('lab_name == "alt" and TestResultCleaned.isna() and TestResult.notna()').index
        df.loc[mask, 'TestResultCleaned'] = (
            pd.to_numeric(df.loc[mask, 'TestResult'].str.replace(r'[LH<>]', '', regex=True).str.strip(), errors='coerce')
            .where(lambda x: (x >= 5) & (x <= 2000))
        )
        
        # Total bilirubin: Convert to TestResult to numeric after removing L, H, <, and >; filter for ranges from 0-40; and impute to TestResultCleaned
        mask = df.query('lab_name == "total_bilirubin" and TestResultCleaned.isna() and TestResult.notna()').index
        df.loc[mask, 'TestResultCleaned'] = (
            pd.to_numeric(df.loc[mask, 'TestResult'].str.replace(r'[LH<>]', '', regex=True).str.strip(), errors='coerce')
            .where(lambda x: (x >= 0) & (x <= 40))
        )
        
        # Albumin
        mask = df.query('lab_name == "albumin" and TestResultCleaned.isna() and TestResult.notna()').index
        # First get the cleaned numeric values
        cleaned_alb_values = pd.to_numeric(df.loc[mask, 'TestResult'].str.replace(r'[LH<>]', '', regex=True).str.strip(), errors='coerce')

        # Identify which values are likely in which unit system
        # Values 1-6 are likely g/dL and need to be converted to g/L
        gdl_mask = (cleaned_alb_values >= 1) & (cleaned_alb_values <= 6)
        # Values 10-60 are likely already in g/L
        gl_mask = (cleaned_alb_values >= 10) & (cleaned_alb_values <= 60)

        # Convert g/dL values to g/L (multiply by 10)
        df.loc[mask[gdl_mask], 'TestResultCleaned'] = cleaned_alb_values[gdl_mask] * 10

        # Keep g/L values as they are
        df.loc[mask[gl_mask], 'TestResultCleaned'] = cleaned_alb_values[gl_mask]

        # Creatinine, BUN, and calcium conversion correction
        # TestResultCleaned incorrectly stored mg/L values 
        # Example: 1.6 mg/L was stored as 0.16 mg/dL instead of 1.6 mg/dL
        # Need to divide by 10 to restore correct value
        mask = (
            ((df['lab_name'] == 'creatinine') | (df['lab_name'] == 'bun') | (df['lab_name'] == 'calcium')) & 
            (df['TestUnits'] == 'mg/L')
        )
        df.loc[mask, 'TestResultCleaned'] = df.loc[mask, 'TestResultCleaned'] * 10 

        # Albumin conversion correction
        # TestResultCleaned incorrectly stored mg/dL values 
        # Example: 3.7 mg/dL was stored as 0.037 g/L instead of 37 g/L
        # Need to multiply 100 to restore correct value
        mask = (
            (df['lab_name'] == 'albumin') & 
            (df['TestUnits'] == 'mg/dL')
        )
        df.loc[mask, 'TestResultCleaned'] = df.loc[mask, 'TestResultCleaned'] * 100         

        # Filter for desired window period for baseline labs after removing missing values after above imputation
        df = df.query('TestResultCleaned.notna()')
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
                                    1)[0] # Extract slope coefficient from linear fit
                if (x['TestResultCleaned'].notna().sum() > 1 and # Need at least 2 valid measurements
                    x['index_to_lab'].notna().sum() > 1 and      # Need at least 2 valid time points
                    len(x['index_to_lab'].unique()) > 1)         # Time points must not be identical
                else np.nan) # Return NaN if conditions for valid slope calculation aren't met
            .reset_index()
            .pivot(index = 'PatientID', columns = 'lab_name', values = 0)
            .rename_axis(columns = None)
            .rename(columns = lambda x: f'{x}_slope')
            .reset_index()
        )
        
        # Merge dataframes - start with index_date_df to ensure all PatientIDs are included
        final_df = index_date_df[['PatientID']].copy()
        final_df = pd.merge(final_df, lab_df, on = 'PatientID', how = 'left')
        final_df = pd.merge(final_df, max_df, on = 'PatientID', how = 'left')
        final_df = pd.merge(final_df, min_df, on = 'PatientID', how = 'left')
        final_df = pd.merge(final_df, std_df, on = 'PatientID', how = 'left')
        final_df = pd.merge(final_df, slope_df, on = 'PatientID', how = 'left')

        # Check for duplicate PatientIDs
        if len(final_df) > final_df['PatientID'].nunique():
            duplicate_ids = final_df[final_df.duplicated(subset = ['PatientID'], keep = False)]['PatientID'].unique()
            logging.warning(f"Duplicate PatientIDs found: {duplicate_ids}")

        logging.info(f"Successfully processed Lab.csv file with final shape: {final_df.shape} and unique PatientIDs: {(final_df['PatientID'].nunique())}")
        labs_df = final_df
        return final_df

    except Exception as e:
        logging.error(f"Error processing Lab.csv file: {e}")
        return None
        
# TESTING
df = pd.read_csv('data_nsclc/Enhanced_AdvancedNSCLC.csv')
a = process_labs(file_path="data_nsclc/Lab.csv",
                       index_date_df= df,
                       index_date_column= 'AdvancedDiagnosisDate',
                       days_before=90,
                       days_after=30,
                       summary_lookback = 180)

embed()