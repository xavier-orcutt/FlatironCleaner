import pandas as pd
import numpy as np
import logging
from IPython import embed

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
    
    Parameters
    ----------
    file_path : str
        Path to Labs.csv file
    index_date_df : pd.DataFrame
        DataFrame containing PatientID and index dates. Only labs for PatientIDs present in this DataFrame will be processed
    index_date_column : str
        Column name in index_date_df containing the index date
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

    # Filter for LOINC codes
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
        LOINC_MAPPINGS.update(additional_loinc_mappings)

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
        all_loinc_codes = sum(LOINC_MAPPINGS.values(), [])

        # Filter for LOINC codes 
        df = df[df['LOINC'].isin(all_loinc_codes)].copy()

        # Map LOINC codes to lab names
        for lab_name, loinc_codes in LOINC_MAPPINGS.items():
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
        
# TESTING 
index_date_df = pd.read_csv("data/Enhanced_AdvUrothelial.csv")
a = process_labs(file_path="data/Lab.csv",
                      index_date_df=index_date_df,
                      index_date_column='AdvancedDiagnosisDate',
                      days_before = 90,
                      days_after = 0)

embed()