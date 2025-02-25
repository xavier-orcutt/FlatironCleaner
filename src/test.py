import pandas as pd
import numpy as np
import logging
from IPython import embed
from typing import Optional
import re 

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

def process_vitals(file_path: str,
                   index_date_df: pd.DataFrame,
                   index_date_column: str, 
                   weight_days_before: int = 90,
                   days_after: int = 0,
                   vital_summary_lookback: int = 180, 
                   abnormal_reading_threshold: int = 2) -> pd.DataFrame:
    """
    Processes Vitals.csv to determine patient BMI, weight, change in weight, and vital sign abnormalities
    within a specified time window relative to an index date. Two different time windows are used for distinct 
    clinical purposes:
    
    1. A smaller window near the index date to find weight and BMI at that time point
    2. A larger lookback window to detect clinically significant vital sign abnormalities 
    suggesting possible deterioration

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
    days_after : int, optional
        Number of days after the index date to include for weight and BMI calculations. Also used as the end point for 
        vital sign abnormalities and weight change calculations. Must be >= 0. Default: 0
    vital_summary_lookback : int, optional
        Number of days before index date to assess for weight change, hypotension, tachycardia, and fever. Must be >= 0. Default: 180
    abnormal_reading_threshold: int, optional 
        Number of abnormal readings required to flag a patient with a vital sign abnormality (hypotension, tachycardia, 
        fevers, hypoxemia). Must be >= 1. Default: 2

    Returns
    -------
    pd.DataFrame
        - PatientID : object 
            unique patient identifier
        - weight : float
            weight in kg closest to index date within specified window (index_date - weight_days_before) to (index_date + weight_days_after)
        - bmi : float
            BMI closest to index date within specified window (index_date - weight_days_before) to (index_date + days_after)
        - percent_change_weight : float
            percentage change in weight over period from (index_date - vital_summary_lookback) to (index_date + days_after)
        - hypotension : Int64
            binary indicator (0/1) for systolic blood pressure <90 mmHg on ≥{abnormal_reading_threshold} separate readings between (index_date - vital_summary_lookback) and (index_date + days_after)
        - tachycardia : Int64
            binary indicator (0/1) for heart rate >100 bpm on ≥{abnormal_reading_threshold} separate readings between (index_date - vital_summary_lookback) and (index_date + days_after)
        - fevers : Int64
            binary indicator (0/1) for temperature >=38°C on ≥{abnormal_reading_threshold} separate readings between (index_date - vital_summary_lookback) and (index_date + days_after)
        - hypoxemia : Int64
            binary indicator (0/1) for SpO2 <=88% on ≥{abnormal_reading_threshold} separate readings between (index_date - vital_summary_lookback) and (index_date + days_after)

    Notes
    -----
    Missing TestResultCleaned values are imputed using TestResult. For those where units are ambiguous, unit conversion is based on thresholds:
        - For weight: 
            Values >140 are presumed to be in pounds and converted to kg (divided by 2.2046)
            Values <70 are presumed to be already in kg and kept as is
            Values between 70-140 are considered ambiguous and not imputed
        - For height: 
            Values between 55-80 are presumed to be in inches and converted to cm (multiplied by 2.54)
            Values between 140-220 are presumed to be already in cm and kept as is
            Values outside these ranges are considered ambiguous and not imputed
        - For temperature: 
            Values >45 are presumed to be in Fahrenheit and converted to Celsius using (F-32)*5/9
            Values ≤45 are presumed to be already in Celsius
    
    BMI is calculated using weight closest to index date within specified window while height outside the specified window may be used. The equation used: weight (kg)/height (m)^2
    BMI calucalted as <13 are considered implausible and removed
    Percent change in weight is calculated as ((end_weight - start_weight) / start_weight) * 100
    TestDate rather than ResultDate is used since TestDate is always populated and, for vital signs, the measurement date (TestDate) and result date (ResultDate) should be identical since vitals are recorded in real-time
    All PatientIDs from index_date_df are included in the output and values will be NaN for patients without weight, BMI, or percent_change_weight, but set to 0 for hypotension, tachycardia, and fevers
    Duplicate PatientIDs are logged as warnings but retained in output
    Results are stored in self.vitals_df attribute
    """
    # Input validation
    if not isinstance(index_date_df, pd.DataFrame):
        raise ValueError("index_date_df must be a pandas DataFrame")
    if 'PatientID' not in index_date_df.columns:
        raise ValueError("index_date_df must contain a 'PatientID' column")
    if not index_date_column or index_date_column not in index_date_df.columns:
        raise ValueError(f"Column '{index_date_column}' not found in index_date_df")
    
    if not isinstance(weight_days_before, int) or weight_days_before < 0:
            raise ValueError("weight_days_before must be a non-negative integer")
    if not isinstance(days_after, int) or days_after < 0:
        raise ValueError("days_after must be a non-negative integer")
    if not isinstance(vital_summary_lookback, int) or vital_summary_lookback < 0:
        raise ValueError("vital_summary_lookback must be a non-negative integer")
    if not isinstance(abnormal_reading_threshold, int) or abnormal_reading_threshold < 1:
            raise ValueError("abnormal_reading_threshold must be an integer ≥1")

    try:
        df = pd.read_csv(file_path, low_memory = False)
        logging.info(f"Successfully read Vitals.csv file with shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")

        df['TestDate'] = pd.to_datetime(df['TestDate'])
        df['TestResult'] = pd.to_numeric(df['TestResult'], errors = 'coerce').astype('float')
        
        # Remove all rows with missing TestResult
        df = df.query('TestResult.notna()')

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
        weight_df = df.query('Test == "body weight"').copy()
        mask_needs_imputation = weight_df['TestResultCleaned'].isna() & weight_df['TestResult'].notna()
        
        imputed_weights = weight_df.loc[mask_needs_imputation, 'TestResult'].apply(
            lambda x: x/2.2046 if x > 140  # Convert to kg since likely lbs 
            else x if x < 70  # Keep as is if likely kg 
            else None  # Leave as null if ambiguous
        )
        
        weight_df.loc[mask_needs_imputation, 'TestResultCleaned'] = imputed_weights
        weight_df = weight_df.query('TestResultCleaned > 0')
        
        df_weight_filtered = weight_df[
            (weight_df['index_to_vital'] <= days_after) & 
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
            (weight_df['index_to_vital'] <= days_after) & 
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

        # Create new window period for vital sign abnormalities 
        df_summary_filtered = df[
            (df['index_to_vital'] <= days_after) & 
            (df['index_to_vital'] >= -vital_summary_lookback)].copy()
        
        # Calculate hypotension indicator 
        bp_df = df_summary_filtered.query("Test == 'systolic blood pressure'").copy()

        bp_df['TestResultCleaned'] = np.where(bp_df['TestResultCleaned'].isna(),
                                              bp_df['TestResult'],
                                              bp_df['TestResultCleaned'])

        hypotension_df = (
            bp_df
            .sort_values(['PatientID', 'TestDate'])
            .groupby('PatientID')
            .agg({
                'TestResultCleaned': lambda x: (
                    sum(x < 90) >= abnormal_reading_threshold) 
            })
            .reset_index()
            .rename(columns = {'TestResultCleaned': 'hypotension'})
        )

        # Calculate tachycardia indicator
        hr_df = df_summary_filtered.query("Test == 'heart rate'").copy()

        hr_df['TestResultCleaned'] = np.where(hr_df['TestResultCleaned'].isna(),
                                              hr_df['TestResult'],
                                              hr_df['TestResultCleaned'])

        tachycardia_df = (
            hr_df 
            .sort_values(['PatientID', 'TestDate'])
            .groupby('PatientID')
            .agg({
                'TestResultCleaned': lambda x: (
                    sum(x > 100) >= abnormal_reading_threshold) 
            })
            .reset_index()
            .rename(columns = {'TestResultCleaned': 'tachycardia'})
        )

        # Calculate fevers indicator
        temp_df = df_summary_filtered.query("Test == 'body temperature'").copy()
        
        mask_needs_imputation = temp_df['TestResultCleaned'].isna() & temp_df['TestResult'].notna()
        
        imputed_temps = temp_df.loc[mask_needs_imputation, 'TestResult'].apply(
            lambda x: (x - 32) * 5/9 if x > 45  # Convert to C since likely F
            else x # Leave as C
        )

        temp_df.loc[mask_needs_imputation, 'TestResultCleaned'] = imputed_temps

        fevers_df = (
            temp_df
            .sort_values(['PatientID', 'TestDate'])
            .groupby('PatientID')
            .agg({
                'TestResultCleaned': lambda x: sum(x >= 38) >= abnormal_reading_threshold 
            })
            .reset_index()
            .rename(columns={'TestResultCleaned': 'fevers'})
        )

        # Calculate hypoxemia indicator 
        oxygen_df = df_summary_filtered.query("Test == 'oxygen saturation in arterial blood by pulse oximetry'").copy()

        oxygen_df['TestResultCleaned'] = np.where(oxygen_df['TestResultCleaned'].isna(),
                                                  oxygen_df['TestResult'],
                                                  oxygen_df['TestResultCleaned'])
        
        hypoxemia_df = (
            oxygen_df
            .sort_values(['PatientID', 'TestDate'])
            .groupby('PatientID')
            .agg({
                'TestResultCleaned': lambda x: sum(x <= 88) >= abnormal_reading_threshold 
            })
            .reset_index()
            .rename(columns={'TestResultCleaned': 'hypoxemia'})
        )

        # Merge dataframes - start with index_date_df to ensure all PatientIDs are included
        final_df = index_date_df[['PatientID']].copy()
        final_df = pd.merge(final_df, weight_index_df, on = 'PatientID', how = 'left')
        final_df = pd.merge(final_df, change_weight_df, on = 'PatientID', how = 'left')
        final_df = pd.merge(final_df, hypotension_df, on = 'PatientID', how = 'left')
        final_df = pd.merge(final_df, tachycardia_df, on = 'PatientID', how = 'left')
        final_df = pd.merge(final_df, fevers_df, on = 'PatientID', how = 'left')
        final_df = pd.merge(final_df, hypoxemia_df, on = 'PatientID', how = 'left')

        boolean_columns = ['hypotension', 'tachycardia', 'fevers', 'hypoxemia']
        for col in boolean_columns:
            final_df[col] = final_df[col].fillna(0).astype('Int64')
        
        if len(final_df) > final_df['PatientID'].nunique():
            duplicate_ids = final_df[final_df.duplicated(subset=['PatientID'], keep=False)]['PatientID'].unique()
            logging.warning(f"Duplicate PatientIDs found: {duplicate_ids}")

        logging.info(f"Successfully processed Vitals.csv file with final shape: {final_df.shape} and unique PatientIDs: {(final_df['PatientID'].nunique())}")
        return final_df

    except Exception as e:
        logging.error(f"Error processing Vitals.csv file: {e}")
        return None
        
# TESTING
df = pd.read_csv('data_nsclc/Enhanced_AdvancedNSCLC.csv')
a = process_vitals(file_path="data_nsclc/Vitals.csv",
                       index_date_df= df,
                       index_date_column= 'AdvancedDiagnosisDate',
                       weight_days_before=90,
                       days_after=14,
                       vital_summary_lookback = 180, 
                       abnormal_reading_threshold = 1)

embed()