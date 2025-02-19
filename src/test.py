import pandas as pd
import numpy as np
import logging
from IPython import embed

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
        Processed DataFrame containing:
        - PatientID : unique patient identifier
        - anticoagulated : True if patient received therapeutic anticoagulation (heparin IV with specific units, 
          enoxaparin >40mg, dalteparin >5000u, fondaparinux >2.5mg, or any DOAC/warfarin) 
        - opioid : True if patient received oral, transdermal, or sublingual opioids
        - steroid : True if patient received oral steroids
        - antibiotic : True if patient received oral/IV antibiotics (excluding antifungals/antivirals)
        - diabetic : True if patient received any antihyperglycemic medication 
        - antidepressant : True if patient received any antidepressant
        - bone_therapy : True if patient received bone-targeted therapy (e.g., bisphosphonates, denosumab)
        - immunosuppressed : True if patient received immunosuppressive medications

    Notes
    -----
    Returns boolean indicators for each medication class, where False indicates either no medication or medication outside the specified time window
    Duplicate PatientIDs are logged as warnings but retained in output
    Results are stored in self.medicines_df attribute
    """

    # Input validation
    if not isinstance(index_date_df, pd.DataFrame) or 'PatientID' not in index_date_df.columns:
        raise ValueError("index_date_df must be a DataFrame containing 'PatientID' column")
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
                .query('Route == "Intravenous"')
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
            .query('Route in ["Oral", "Transdermal", "Sublingual"]')
            .query('DrugName in ["oxycodone hcl", "hydromorphone hcl", "oxycodone hcl/acetaminophen", "morphine sulfate", "methadone hcl", "hydrocodone bitartrate/acetaminophen", "fentanyl", "levorphanol tartrate", "oxymorphone hcl", "tapentadol hcl", "hydrocodone bitartrate"]')
            .PatientID
        ).unique()
        
        steroid_IDs = (
            df_filtered
            .query('DrugCategory == "steroid"')
            .query('Route == "Oral"')
            .query('DrugName != "Clinical study drug"')
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
            "cefdinir", "cefpodoxime",
            
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

        # Merge arrays
        # Create the dataframe
        final_df = pd.DataFrame(index = pd.Series(list(index_date_df.PatientID), name = 'PatientID'))

        # Add boolean columns for each medication category
        final_df['anticoagulated'] = final_df.index.isin(anticoagulated_IDs)
        final_df['opioid'] = final_df.index.isin(opioid_IDs)
        final_df['steroid'] = final_df.index.isin(steroid_IDs)
        final_df['antibiotic'] = final_df.index.isin(antibiotic_IDs)
        final_df['diabetic'] = final_df.index.isin(diabetic_IDs)
        final_df['antidepressant'] = final_df.index.isin(antidepressant_IDs)
        final_df['bone_therapy'] = final_df.index.isin(bta_IDs)
        final_df['immunosuppressed'] = final_df.index.isin(immunosuppressed_IDs)

        final_df = final_df.reset_index()

        # Check for duplicate PatientIDs
        if len(final_df) > final_df['PatientID'].nunique():
            logging.error(f"Duplicate PatientIDs found")
            return None

        logging.info(f"Successfully processed MedicationAdministration.csv file with final shape: {final_df.shape} and unique PatientIDs: {(final_df['PatientID'].nunique())}")
        return final_df

    except Exception as e:
        logging.error(f"Error processing MedicationAdministration.csv file: {e}")
        return None
        
# TESTING 
index_date_df = pd.read_csv("data/Enhanced_AdvUrothelial.csv")
a = process_medications(file_path="data/MedicationAdministration.csv",
                      index_date_df=index_date_df,
                      index_date_column='AdvancedDiagnosisDate',
                      days_before = 90,
                      days_after = 0)

embed()