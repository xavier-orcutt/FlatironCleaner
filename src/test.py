import pandas as pd
import numpy as np
import logging
from IPython import embed
from typing import Optional
import re 

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

def process_medications(file_path: str,
                        index_date_df: pd.DataFrame,
                        index_date_column: str,
                        days_before: int = 90,
                        days_after: int = 0) -> pd.DataFrame:
    """
    Processes MedicationAdministration.csv to determine clinically relevant medicines received by patients within a specified time window 
    relative to an index date. 
    
    Parameters
    ----------
    file_path : str
        Path to MedicationAdministration.csv file
    index_date_df : pd.DataFrame
        DataFrame containing PatientID and index dates. Only medicines for PatientIDs present in this DataFrame will be processed
    index_date_column : str
        Column name in index_date_df containing the index date
    days_before : int, optional
        Number of days before the index date to include for window period. Must be >= 0. Default: 90
    days_after : int, optional
        Number of days after the index date to include for window period. Must be >= 0. Default: 0
    
    Returns
    -------
    pd.DataFrame
        - PatientID : ojbect
            unique patient identifier
        - anticoagulant : Int64
            binary indicator (0/1) for therapeutic anticoagulation (heparin IV with specific units, enoxaparin >40mg, dalteparin >5000u, fondaparinux >2.5mg, or any DOAC/warfarin) 
        - opioid : Int64
            binary indicator (0/1) for oral, transdermal, sublingual, or enteral opioids
        - steroid : Int64
            binary indicator (0/1) for oral steroids
        - antibiotic : Int64
            binary indicator (0/1) for oral/IV antibiotics (excluding antifungals/antivirals)
        - diabetic : Int64
            binary indicator (0/1) for antihyperglycemic medication 
        - antidepressant : Int64
            binary indicator (0/1) for antidepressant
        - bone_therapy : Int64
            binary indicator (0/1) for bone-targeted therapy (e.g., bisphosphonates, denosumab)
        - immunosuppressed : Int64
            binary indicator (0/1) for immunosuppressive medications

    Notes
    -----
    All PatientIDs from index_date_df are included in the output
    Duplicate PatientIDs are logged as warnings but retained in output 
    Results are stored in self.medicines_df attribute
    """
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

    try:
        df = pd.read_csv(file_path)
        logging.info(f"Successfully read MedicationAdministration.csv file with shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")

        df['AdministeredDate'] = pd.to_datetime(df['AdministeredDate'])
        df['AdministeredAmount'] = df['AdministeredAmount'].astype(float)
        df = df.query('CommonDrugName != "Clinical study drug"')
                                      
        index_date_df[index_date_column] = pd.to_datetime(index_date_df[index_date_column])

        # Select PatientIDs that are included in the index_date_df the merge on 'left'
        df = df[df.PatientID.isin(index_date_df.PatientID)]
        df = pd.merge(
            df,
            index_date_df[['PatientID', index_date_column]],
            on = 'PatientID',
            how = 'left'
        )
        logging.info(f"Successfully merged MedicationAdministration.csv df with index_date_df resulting in shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")
        
        # Filter for desired window period for baseline labs
        df['index_to_med'] = (df['AdministeredDate'] - df[index_date_column]).dt.days
        
        df_filtered = df[
            (df['index_to_med'] <= days_after) & 
            (df['index_to_med'] >= -days_before)].copy()
        
        anticoagulated_IDs = pd.concat([
            # Heparin patients
            (
                df_filtered
                .query('CommonDrugName == "heparin (porcine)"')
                .query('AdministeredUnits in ["unit/kg/hr", "U/hr", "U/kg"]')
                .PatientID
            ),
            # Enoxaparin patients
            (
                df_filtered
                .query('CommonDrugName == "enoxaparin"')
                .query('AdministeredAmount > 40')
                .PatientID
            ),
        
            # Dalteparin patients
            (
                df_filtered
                .query('CommonDrugName == "dalteparin,porcine"')
                .query('AdministeredAmount > 5000')
                .PatientID
            ),
        
            # Fondaparinux patients
            (
                df_filtered
                .query('CommonDrugName == "fondaparinux"')
                .query('AdministeredAmount > 2.5')
                .PatientID
            ),
        
            # Warfarin and DOAC patients
            (
                df_filtered
                .query('CommonDrugName in ["warfarin", "apixaban", "rivaroxaban", "dabigatran etexilate", "edoxaban"]')
                .PatientID
            )
            ]).unique()
        
        opioid_IDs = (
            df_filtered
            .query('DrugCategory == "pain agent"')
            .query('Route in ["Oral", "Transdermal", "Sublingual", "enteral", "Subcutaneous"]')
            .query('CommonDrugName in ["oxycodone", "morphine", "hydromorphone", "acetaminophen/oxycodone", "tramadol", "methadone", "fentanyl", "acetaminophen/hydrocodone", "acetaminophen/codeine", "codeine", "oxymorphone", "tapentadol", "buprenorphine", "acetaminophen/tramadol", "hydrocodone", "levorphanol", "acetaminophen/tramadol"]')
            .PatientID
        ).unique()
        
        steroid_IDs = (
            df_filtered
            .query('DrugCategory == "steroid"')
            .query('Route == "Oral"')
            .PatientID
        ).unique()
        
        antibiotics = [
            # Glycopeptides
            "vancomycin",
            
            # Beta-lactams
            "piperacillin/tazobactam", "cefazolin", "ceftriaxone",
            "cefepime", "meropenem", "cefoxitin", "ampicillin/sulbactam",
            "ampicillin", "amoxicillin/clavulanic acid", "ertapenem", 
            "dextrose, iso-osmotic/piperacillin/tazobactam", "ceftazidime", 
            "cephalexin", "cefuroxime", "amoxicillin", "oxacillin", 
            "cefdinir", "cefpodoxime", "cefadroxil", "penicillin g",
            
            # Fluoroquinolones
            "ciprofloxacin", "levofloxacin", "moxifloxacin",
            
            # Nitroimidazoles
            "metronidazole",
            
            # Sulfonamides
            "sulfamethoxazole/trimethoprim",
            
            # Tetracyclines
            "doxycycline", "minocycline", "tigecycline",
            
            # Lincosamides
            "clindamycin",
            
            # Aminoglycosides
            "gentamicin", "neomycin",
            
            # Macrolides
            "azithromycin", "erythromycin base",
            
            # Oxazolidinones
            "linezolid",
            
            # Other classes
            "daptomycin",  
            "aztreonam",  
            "nitrofurantoin",  
            "fosfomycin",  
            "rifaximin"   
        ]

        antibiotic_IDs = (
            df_filtered
            .query('DrugCategory == "anti-infective"')
            .query('Route in ["Oral", "Intravenous"]')
            .query('CommonDrugName in @antibiotics')
            .PatientID
        ).unique()

        diabetic_IDs = ( 
            df_filtered 
            .query('DrugCategory == "antihyperglycemic"') 
            .PatientID 
        ).unique()

        antidepressant_IDs = (
            df_filtered 
            .query('DrugCategory == "antidepressant"') 
            .PatientID 
        ).unique()

        bta_IDs = (
            df_filtered
            .query('DrugCategory == "bone therapy agent (bta)"')
            .PatientID
        ).unique()

        immunosuppressed_IDs = (
            df_filtered
            .query('DrugCategory == "immunosuppressive"')
            .PatientID
        ).unique()

        # Create dictionary of medication categories and their respective IDs
        med_categories = {
            'anticoagulant': anticoagulated_IDs,
            'opioid': opioid_IDs,
            'steroid': steroid_IDs,
            'antibiotic': antibiotic_IDs,
            'diabetic': diabetic_IDs,
            'antidepressant': antidepressant_IDs,
            'bone_therapy': bta_IDs,
            'immunosuppressed': immunosuppressed_IDs
        }

        # Start with index_date_df to ensure all PatientIDs are included
        final_df = index_date_df[['PatientID']].copy()

        # Add binary (0/1) columns for each medication category
        for category, ids in med_categories.items():
            final_df[category] = final_df['PatientID'].isin(ids).astype('Int64')

        # Check for duplicate PatientIDs
        if len(final_df) > final_df['PatientID'].nunique():
            duplicate_ids = final_df[final_df.duplicated(subset = ['PatientID'], keep = False)]['PatientID'].unique()
            logging.warning(f"Duplicate PatientIDs found: {duplicate_ids}")

        logging.info(f"Successfully processed MedicationAdministration.csv file with final shape: {final_df.shape} and unique PatientIDs: {(final_df['PatientID'].nunique())}")
        return final_df

    except Exception as e:
        logging.error(f"Error processing MedicationAdministration.csv file: {e}")
        return None

# TESTING
df = pd.read_csv('data_nsclc/Enhanced_AdvancedNSCLC.csv')
a = process_medications(file_path="data_nsclc/MedicationAdministration.csv",
                       index_date_df= df,
                       index_date_column= 'AdvancedDiagnosisDate',
                       days_before=180,
                       days_after=30)

embed()