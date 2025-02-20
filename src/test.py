import pandas as pd
import numpy as np
import logging
from IPython import embed
from typing import Optional
import re 

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

def process_diagnosis(file_path: str,
                      index_date_df: pd.DataFrame,
                      index_date_column: str,
                      days_before: Optional[int] = None,
                      days_after: int = 0) -> pd.DataFrame:
    """
    Processes Diagnosis.csv by mapping ICD 9 and 10 codes to Elixhauser comorbidity index and calculates a van Walraven score. 
    It also determines site of metastases based on ICD 9 and 10 codes. 
    See "Coding algorithms for defining comorbidities in ICD-9-CM and ICD-10 administrative data" by Quan et al for details 
    on ICD mapping to comorbidities. 
    See "A modification of the Elixhauser comorbidity measures into a point system for hospital death using administrative data"
    by van Walraven et al for details on van Walraven score. 
    
    Parameters
    ----------
    file_path : str
        Path to Diagnosis.csv file
    index_date_df : pd.DataFrame
        DataFrame containing PatientID and index dates. Only diagnoses for PatientIDs present in this DataFrame will be processed
    index_date_column : str
        Column name in index_date_df containing the index date
    days_before : int | None, optional
        Number of days before the index date to include for window period. Must be >= 0 or None. If None, includes all prior results. Default: None
    days_after : int, optional
        Number of days after the index date to include for window period. Must be >= 0. Default: 0
    
    Returns
    -------
    pd.DataFrame
        Processed DataFrame containing:
        - PatientID : unique patient identifier
        - CHF : binary indicator for Congestive Heart Failure
        - Arrhythmia : binary indicator for cardiac arrhythmias
        - Valvular : binary indicator for valvular disease
        - Pulmonary : binary indicator for pulmonary circulation disorders
        - PVD : binary indicator for peripheral vascular disease
        - HTN : binary indicator for hypertension (uncomplicated)
        - HTNcx : binary indicator for hypertension (complicated)
        - Paralysis : binary indicator for paralysis
        - Neuro : binary indicator for other neurological disorders
        - COPD : binary indicator for chronic pulmonary disease
        - DM : binary indicator for diabetes (uncomplicated)
        - DMcx : binary indicator for diabetes (complicated)
        - Hypothyroid : binary indicator for hypothyroidism
        - Renal : binary indicator for renal failure
        - Liver : binary indicator for liver disease
        - PUD : binary indicator for peptic ulcer disease
        - AIDS : binary indicator for AIDS/HIV
        - Lymphoma : binary indicator for lymphoma
        - Rheumatic : binary indicator for rheumatoid arthritis
        - Coagulopathy : binary indicator for coagulopathy
        - Obesity : binary indicator for obesity
        - WeightLoss : binary indicator for weight loss
        - Fluid : binary indicator for fluid and electrolyte disorders
        - BloodLoss : binary indicator for blood loss anemia
        - DefAnemia : binary indicator for deficiency anemia
        - Alcohol : binary indicator for alcohol abuse
        - DrugAbuse : binary indicator for drug abuse
        - Psychoses : binary indicator for psychoses
        - Depression : binary indicator for depression
        - van_walraven_score : weighted composite of the binary Elixhauser comorbidities 
        - lymph_mets : binary indicator for lymph node metastases
        - lung_mets : binary indicator for lung metastases
        - pleura_mets : binary indicator for pleural metastases
        - liver_mets : binary indicator for liver metastases
        - bone_mets : binary indicator for bone metastases
        - brain_mets : binary indicator for brain/CNS metastases
        - adrenal_mets : binary indicator for adrenal metastases
        - peritoneum_mets : binary indicator for peritoneal/retroperitoneal metastases
        - gi_mets : binary indicator for gastrointestinal tract metastases
        - other_mets : binary indicator for other sites of metastases
        - unspecified_mets : binary indicator for unspecified metastases

    Notes
    -----
    Maps both ICD-9-CM and ICD-10-CM codes to Elixhauser comorbidities and metastatic sites
    Metastatic cancer and tumor categories are excluded in the Elxihauser comorbidities and van Walravane score as this is intended for an advanced cancer population
    For patients with both ICD-9 and ICD-10 codes, comorbidities and metastatic sites are combined (if a comorbidity is present in either coding system, it is marked as present)
    Duplicate PatientIDs are logged as warnings but retained in output
    Results are stored in self.diagnoses_df attribute
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
        logging.info(f"Successfully read Diagnosis.csv file with shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")

        df['DiagnosisDate'] = pd.to_datetime(df['DiagnosisDate'])
        index_date_df[index_date_column] = pd.to_datetime(index_date_df[index_date_column])

        # Select PatientIDs that are included in the index_date_df the merge on 'left'
        df = df[df.PatientID.isin(index_date_df.PatientID)]
        df = pd.merge(
             df,
             index_date_df[['PatientID', index_date_column]],
             on = 'PatientID',
             how = 'left'
             )
        logging.info(f"Successfully merged Diagnosis.csv df with index_date_df resulting in shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")

        # Filter for desired window period for baseline labs
        df['index_to_diagnosis'] = (df['DiagnosisDate'] - df[index_date_column]).dt.days
        
        # Select biomarkers that fall within desired before and after index date
        if days_before is None:
            # Only filter for days after
            df_filtered = df[df['index_to_diagnosis'] <= days_after].copy()
        else:
            # Filter for both before and after
            df_filtered = df[
                (df['index_to_diagnosis'] <= days_after) & 
                (df['index_to_diagnosis'] >= -days_before)
            ].copy()

        df9_elix = (
            df_filtered
            .query('DiagnosisCodeSystem == "ICD-9-CM"')
            .assign(diagnosis_code = lambda x: x['DiagnosisCode'].replace(r'\.', '', regex=True)) # Remove decimal points from ICD-9 codes to make mapping easier 
            .drop_duplicates(subset = ['PatientID', 'diagnosis_code'], keep = 'first')
            .assign(comorbidity=lambda x: x['diagnosis_code'].map(
                lambda code: next((comorb for pattern, comorb in ICD_9_EXLIXHAUSER_MAPPING.items() 
                                if re.match(pattern, code)), 'Other')))
            .query('comorbidity != "Other"') 
            .drop_duplicates(subset=['PatientID', 'comorbidity'], keep = 'first')
            .assign(value=1)  # Add a column of 1s to use for pivot
            .pivot(index = 'PatientID', columns = 'comorbidity', values = 'value')
            .fillna(0) 
            .astype(int)  
            .rename_axis(columns = None)
            .reset_index()
        )

        df10_elix = (
            df_filtered
            .query('DiagnosisCodeSystem == "ICD-10-CM"')
            .assign(diagnosis_code = lambda x: x['DiagnosisCode'].replace(r'\.', '', regex=True)) # Remove decimal points from ICD-10 codes to make mapping easier 
            .drop_duplicates(subset = ['PatientID', 'diagnosis_code'], keep = 'first')
            .assign(comorbidity=lambda x: x['diagnosis_code'].map(
                lambda code: next((comorb for pattern, comorb in ICD_10_ELIXHAUSER_MAPPING.items() 
                                if re.match(pattern, code)), 'Other')))
            .query('comorbidity != "Other"') 
            .drop_duplicates(subset=['PatientID', 'comorbidity'], keep = 'first')
            .assign(value=1)  # Add a column of 1s to use for pivot
            .pivot(index = 'PatientID', columns = 'comorbidity', values = 'value')
            .fillna(0) 
            .astype(int)
            .rename_axis(columns = None)
            .reset_index()  
        )

        all_columns_elix = ['PatientID'] + list(set(ICD_9_EXLIXHAUSER_MAPPING.values()) - {'Metastatic', 'Tumor'})
        
        # Reindex both dataframes to have all columns, filling missing ones with 0
        df9_elix_aligned = df9_elix.reindex(columns = all_columns_elix, fill_value = 0)
        df10_elix_aligned = df10_elix.reindex(columns = all_columns_elix, fill_value = 0)

        df_elix_combined = pd.concat([df9_elix_aligned, df10_elix_aligned]).groupby('PatientID').max().reset_index()

        van_walraven_score = df_elix_combined.drop('PatientID', axis=1).mul(VAN_WALRAVEN_WEIGHTS).sum(axis=1)
        df_elix_combined['van_walraven_score'] = van_walraven_score


        df9_mets = (
            df_filtered
            .query('DiagnosisCodeSystem == "ICD-9-CM"')
            .assign(diagnosis_code = lambda x: x['DiagnosisCode'].replace(r'\.', '', regex=True)) # Remove decimal points from ICD-9 codes to make mapping easier 
            .drop_duplicates(subset = ['PatientID', 'diagnosis_code'], keep = 'first')
            .assign(met_site=lambda x: x['diagnosis_code'].map(
                lambda code: next((site for pattern, site in ICD_9_METS_MAPPING.items()
                                   if re.match(pattern, code)), 'no_met')))
            .query('met_site != "no_met"') 
            .drop_duplicates(subset=['PatientID', 'met_site'], keep = 'first')
            .assign(value=1)  # Add a column of 1s to use for pivot
            .pivot(index = 'PatientID', columns = 'met_site', values = 'value')
            .fillna(0) 
            .astype(int)  
            .rename_axis(columns = None)
            .reset_index()
        )

        df10_mets = (
            df_filtered
            .query('DiagnosisCodeSystem == "ICD-10-CM"')
            .assign(diagnosis_code = lambda x: x['DiagnosisCode'].replace(r'\.', '', regex=True)) # Remove decimal points from ICD-9 codes to make mapping easier 
            .drop_duplicates(subset = ['PatientID', 'diagnosis_code'], keep = 'first')
            .assign(met_site=lambda x: x['diagnosis_code'].map(
                lambda code: next((site for pattern, site in ICD_10_METS_MAPPING.items()
                                   if re.match(pattern, code)), 'no_met')))
            .query('met_site != "no_met"') 
            .drop_duplicates(subset=['PatientID', 'met_site'], keep = 'first')
            .assign(value=1)  # Add a column of 1s to use for pivot
            .pivot(index = 'PatientID', columns = 'met_site', values = 'value')
            .fillna(0) 
            .astype(int)  
            .rename_axis(columns = None)
            .reset_index()
        )

        all_columns_mets = ['PatientID'] + list(set(ICD_9_METS_MAPPING.values())) 
        
        # Reindex both dataframes to have all columns, filling missing ones with 0
        df9_mets_aligned = df9_mets.reindex(columns = all_columns_mets, fill_value = 0)
        df10_mets_aligned = df10_mets.reindex(columns = all_columns_mets, fill_value = 0)

        df_mets_combined = pd.concat([df9_mets_aligned, df10_mets_aligned]).groupby('PatientID').max().reset_index()

        # Merging dataframes for Elixhauser comorbidities and metastatic sites
        final_df = pd.merge(df_elix_combined, df_mets_combined, on = 'PatientID', how = 'outer')
        final_df = final_df.fillna(0)

        binary_columns = [col for col in final_df.columns 
                 if col not in ['PatientID', 'van_walraven_score']]
        final_df[binary_columns] = final_df[binary_columns].astype(int)

        # Check for duplicate PatientIDs
        if len(final_df) > final_df['PatientID'].nunique():
            logging.error(f"Duplicate PatientIDs found")
            return None

        logging.info(f"Successfully processed Diagnosis.csv file with final shape: {final_df.shape} and unique PatientIDs: {(final_df['PatientID'].nunique())}")
        return final_df

    except Exception as e:
        logging.error(f"Error processing Diagnosis.csv file: {e}")
        return None
        
# TESTING 
index_date_df = pd.read_csv("data/Enhanced_AdvUrothelial.csv")
a = process_diagnosis(file_path="data/Diagnosis.csv",
                      index_date_df=index_date_df,
                      index_date_column='AdvancedDiagnosisDate',
                      days_before = None,
                      days_after = 0)

embed()