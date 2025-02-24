import pandas as pd
import numpy as np
import logging
from IPython import embed
from typing import Optional
import re 

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

def process_biomarkers(file_path: str,
                       index_date_df: pd.DataFrame,
                       index_date_column: str, 
                       days_before: Optional[int] = None,
                       days_after: int = 0) -> pd.DataFrame:
    """
    Processes Enhanced_AdvNSCLCBiomarkers.csv by determining FGFR and PDL1 status for each patient within a specified time window relative to an index date. 

    Parameters
    ----------
    file_path : str
        Path to Enhanced_AdvNSCLCBiomarkers.csv file
    index_date_df : pd.DataFrame
        DataFrame containing PatientID and index dates. Only biomarkers for PatientIDs present in this DataFrame will be processed
    index_date_column : str
        Column name in index_date_df containing the index date
    days_before : int | None, optional
        Number of days before the index date to include. Must be >= 0 or None. If None, includes all prior results. Default: None
    days_after : int, optional
        Number of days after the index date to include. Must be >= 0. Default: 0
    
    Returns
    -------
    pd.DataFrame
        - PatientID : object
            unique patient identifier
        - EGFR_status : category
            positive if ever-positive, negative if only-negative, otherwise unknown
        - KRAS_status : cateogory
            positive if ever-positive, negative if only-negative, otherwise unknown
        - BRAF_status : cateogory
            positive if ever-positive, negative if only-negative, otherwise unknown
        - ALK_status : cateogory
            positive if ever-positive, negative if only-negative, otherwise unknown
        - ROS1_status : cateogory
            positive if ever-positive, negative if only-negative, otherwise unknown
        - MET_status : category
            positive if ever-positive, negative if only-negative, otherwise unknown
        - RET_status : category
            positive if ever-positive, negative if only-negative, otherwise unknown
        - NTRK_status : category
            positive if ever-positive, negative if only-negative, otherwise unknown
        - PDL1_status : cateogry 
            positive if ever-positive, negative if only-negative, otherwise unknown
        - PDL1_percent_staining : category, ordered 
            returns a patient's maximum percent staining for PDL1

    Notes
    ------
    Missing ResultDate is imputed with SpecimenReceivedDate.
    All PatientIDs from index_date_df are included in the output and values will be NaN for patients without any biomarker tests
    NTRK genes (NTRK1, NTRK2, NTRK3, NTRK - other, NTRK - unknown) are grouped given that gene type does not impact treatment decisions
    For each biomarker, status is classified as:
        - 'positive' if any test result is positive (ever-positive)
        - 'negative' if any test is negative without positives (only-negative) 
        - 'unknown' if all results are indeterminate
    Duplicate PatientIDs are logged as warnings if found
    Processed DataFrame is stored in self.biomarkers_df
    """
    # Input validation
    if not isinstance(index_date_df, pd.DataFrame):
        raise ValueError("index_date_df must be a pandas DataFrame")
    if 'PatientID' not in index_date_df.columns:
        raise ValueError("index_date_df must contain a 'PatientID' column")
    if not index_date_column or index_date_column not in index_date_df.columns:
        raise ValueError(f"Column '{index_date_column}' not found in index_date_df")
    
    if days_before is not None:
        if not isinstance(days_before, int) or days_before < 0:
            raise ValueError("days_before must be a non-negative integer or None")
    if not isinstance(days_after, int) or days_after < 0:
        raise ValueError("days_after must be a non-negative integer")

    try:
        df = pd.read_csv(file_path)
        logging.info(f"Successfully read Enhanced_AdvNSCLCBiomarkers.csv file with shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")

        df['ResultDate'] = pd.to_datetime(df['ResultDate'])
        df['SpecimenReceivedDate'] = pd.to_datetime(df['SpecimenReceivedDate'])

        # Impute missing ResultDate with SpecimenReceivedDate
        df['ResultDate'] = np.where(df['ResultDate'].isna(), df['SpecimenReceivedDate'], df['ResultDate'])

        index_date_df[index_date_column] = pd.to_datetime(index_date_df[index_date_column])

        # Select PatientIDs that are included in the index_date_df the merge on 'left'
        df = df[df.PatientID.isin(index_date_df.PatientID)]
        df = pd.merge(
                df,
                index_date_df[['PatientID', index_date_column]],
                on = 'PatientID',
                how = 'left'
                )
        logging.info(f"Successfully merged Enhanced_AdvNSCLCBiomarkers.csv df with index_date_df resulting in shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")
        
        # Create new variable 'index_to_result' that notes difference in days between resulted specimen and index date
        df['index_to_result'] = (df['ResultDate'] - df[index_date_column]).dt.days
        
        # Select biomarkers that fall within desired before and after index date
        if days_before is None:
            # Only filter for days after
            df_filtered = df[df['index_to_result'] <= days_after].copy()
        else:
            # Filter for both before and after
            df_filtered = df[
                (df['index_to_result'] <= days_after) & 
                (df['index_to_result'] >= -days_before)
            ].copy()

        # Group NTRK genes
        df_filtered['BiomarkerName'] = (
            np.where(df_filtered['BiomarkerName'].isin(["NTRK1", "NTRK2", "NTRK3", "NTRK - unknown gene type","NTRK - other"]),
                     "NTRK",
                     df_filtered['BiomarkerName'])
        )

        # Create an empty dictionary to store the dataframes
        biomarker_dfs = {}

        # Process EGFR, KRAS, and BRAF 
        for biomarker in ['EGFR', 'KRAS', 'BRAF']:
            biomarker_dfs[biomarker] = (
                df_filtered
                .query(f'BiomarkerName == "{biomarker}"')
                .groupby('PatientID')['BiomarkerStatus']
                .agg(lambda x: 'positive' if any('Mutation positive' in val for val in x)
                    else ('negative' if any('Mutation negative' in val for val in x)
                        else 'unknown'))
                .reset_index()
                .rename(columns={'BiomarkerStatus': f'{biomarker}_status'})  # Rename for clarity
        )
            
        # Process ALK and ROS1
        for biomarker in ['ALK', 'ROS1']:
            biomarker_dfs[biomarker] = (
                df_filtered
                .query(f'BiomarkerName == "{biomarker}"')
                .groupby('PatientID')['BiomarkerStatus']
                .agg(lambda x: 'positive' if any('Rearrangement present' in val for val in x)
                    else ('negative' if any('Rearrangement not present' in val for val in x)
                        else 'unknown'))
                .reset_index()
                .rename(columns={'BiomarkerStatus': f'{biomarker}_status'})  # Rename for clarity
        )
            
        # Process MET, RET, and NTRK
        positive_values = {
            "Protein expression positive",
            "Mutation positive",
            "Amplification positive",
            "Rearrangement positive",
            "Other result type positive",
            "Unknown result type positive"
        }

        for biomarker in ['MET', 'RET', 'NTRK']:
            biomarker_dfs[biomarker] = (
                df_filtered
                .query(f'BiomarkerName == "{biomarker}"')
                .groupby('PatientID')['BiomarkerStatus']
                .agg(lambda x: 'positive' if any(val in positive_values for val in x)
                    else ('negative' if any('Negative' in val for val in x)
                        else 'unknown'))
                .reset_index()
                .rename(columns={'BiomarkerStatus': f'{biomarker}_status'})  # Rename for clarity
        )
        
        # Process PDL1 and add to biomarker_dfs
        biomarker_dfs['PDL1'] = (
            df_filtered
            .query('BiomarkerName == "PDL1"')
            .groupby('PatientID')['BiomarkerStatus']
            .agg(lambda x: 'positive' if any ('PD-L1 positive' in val for val in x)
                else ('negative' if any('PD-L1 negative/not detected' in val for val in x)
                    else 'unknown'))
            .reset_index()
            .rename(columns={'BiomarkerStatus': 'PDL1_status'})
        )

        PDL1_PERCENT_STAINING_MAPPING = {
            np.nan: 0,
            '0%': 1, 
            '< 1%': 2,
            '1%': 3, 
            '2% - 4%': 4,
            '5% - 9%': 5,
            '10% - 19%': 6,  
            '20% - 29%': 7, 
            '30% - 39%': 8, 
            '40% - 49%': 9, 
            '50% - 59%': 10, 
            '60% - 69%': 11, 
            '70% - 79%': 12, 
            '80% - 89%': 13, 
            '90% - 99%': 14,
            '100%': 15
        }

        # Process PDL1 staining 
        PDL1_staining_df = (
            df_filtered
            .query('BiomarkerName == "PDL1"')
            .query('BiomarkerStatus == "PD-L1 positive"')
            .groupby('PatientID')['PercentStaining']
            .apply(lambda x: x.map(PDL1_PERCENT_STAINING_MAPPING))
            .groupby('PatientID')
            .agg('max')
            .to_frame(name = 'PDL1_ordinal_value')
            .reset_index()
        )
        
        # Create reverse mapping to convert back to percentage strings
        reverse_pdl1_dict = {v: k for k, v in PDL1_PERCENT_STAINING_MAPPING.items()}
        PDL1_staining_df['PDL1_percent_staining'] = PDL1_staining_df['PDL1_ordinal_value'].map(reverse_pdl1_dict)
        PDL1_staining_df = PDL1_staining_df.drop(columns = ['PDL1_ordinal_value'])

        # Merge dataframes -- start with index_date_df to ensure all PatientIDs are included
        final_df = index_date_df[['PatientID']].copy()

        for biomarker in ['EGFR', 'KRAS', 'BRAF', 'ALK', 'ROS1', 'MET', 'RET', 'NTRK', 'PDL1']:
            final_df = pd.merge(final_df, biomarker_dfs[biomarker], on = 'PatientID', how = 'left')

        final_df = pd.merge(final_df, PDL1_staining_df, on = 'PatientID', how = 'left')


        for biomarker_status in ['EGFR_status', 'KRAS_status', 'BRAF_status', 'ALK_status', 'ROS1_status', 'MET_status', 'RET_status', 'NTRK_status', 'PDL1_status']:
            final_df[biomarker_status] = final_df[biomarker_status].astype('category')

        staining_dtype = pd.CategoricalDtype(
            categories = ['0%', '< 1%', '1%', '2% - 4%', '5% - 9%', '10% - 19%',
                            '20% - 29%', '30% - 39%', '40% - 49%', '50% - 59%',
                            '60% - 69%', '70% - 79%', '80% - 89%', '90% - 99%', '100%'],
                            ordered = True
                            )
        
        final_df['PDL1_percent_staining'] = final_df['PDL1_percent_staining'].astype(staining_dtype)

        # Check for duplicate PatientIDs
        if len(final_df) > final_df['PatientID'].nunique():
            logging.error(f"Duplicate PatientIDs found")
            return None

        logging.info(f"Successfully processed Enhanced_AdvNSCLCBiomarkers.csv file with final shape: {final_df.shape} and unique PatientIDs: {(final_df['PatientID'].nunique())}")
        return final_df

    except Exception as e:
        logging.error(f"Error processing Enhanced_AdvNSCLCBiomarkers.csv file: {e}")
        return None
    
        
# TESTING
df = pd.read_csv('data_nsclc/Enhanced_AdvancedNSCLC.csv')
ids = df.sample(n=1000).PatientID.to_list() 
a = process_biomarkers(file_path="data_nsclc/Enhanced_AdvNSCLCBiomarkers.csv",
                       index_date_df= df,
                       index_date_column= 'AdvancedDiagnosisDate',
                       days_before=None,
                       days_after=30)

embed()