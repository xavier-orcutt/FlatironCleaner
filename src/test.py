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
        - PatientID : object unique patient identifier
        - chf : binary indicator for congestive heart failure
        - cardiac_arrhythmia : binary indicator for cardiac arrhythmias
        - valvular_disease : binary indicator for valvular disease
        - pulm_circulation : binary indicator for pulmonary circulation disorders
        - pvd : binary indicator for peripheral vascular disease
        - htn_uncomplicated : binary indicator for hypertension, uncomplicated
        - htn_complicated : binary indicator for hypertension, complicated
        - paralysis : binary indicator for paralysis
        - other_neuro : binary indicator for other neurological disorders
        - chronic_pulm_disease : binary indicator for chronic pulmonary disease
        - diabetes_uncomplicated : binary indicator for diabetes, uncomplicated
        - diabetes_complicated : binary indicator for diabetes, complicated
        - hypothyroid : binary indicator for hypothyroidism
        - renal_failuare : binary indicator for renal failure
        - liver_disease : binary indicator for liver disease
        - PUD : binary indicator for peptic ulcer disease
        - aids : binary indicator for AIDS/HIV
        - lymphoma : binary indicator for lymphoma
        - rheumatic : binary indicator for rheumatoid arthritis/collagen vascular diseases
        - coagulopathy : binary indicator for coagulopathy
        - obesity : binary indicator for obesity
        - weight_loss : binary indicator for weight loss
        - fluid : binary indicator for fluid and electrolyte disorders
        - blood_loss_anemia : binary indicator for blood loss anemia
        - deficiency_anemia : binary indicator for deficiency anemia
        - alcohol : binary indicator for alcohol abuse
        - drug_abuse : binary indicator for drug abuse
        - psychoses : binary indicator for psychoses
        - depression : binary indicator for depression
        - van_walraven_score : weighted composite of the binary Elixhauser comorbidities 

    Notes
    -----
    Maps both ICD-9-CM and ICD-10-CM codes to Elixhauser comorbidities and metastatic sites
    Metastatic cancer and tumor categories are excluded in the Elxihauser comorbidities and van Walraven score as this is intended for an advanced cancer population
    All PatientIDs from index_date_df are included in the output and values will be set to 0 for patients with misisng Elixhauser comorbidities or metastasis sites, but NaN for van_walraven_score
    Duplicate PatientIDs are logged as warnings but retained in output
    Results are stored in self.diagnoses_df attribute
    """
    ICD_9_EXLIXHAUSER_MAPPING = {
        # Congestive heart failure
        r'^39891|^40201|^40211|^40291|^40401|^40403|^40411|^40413|^40491|^40493|^4254|^4255|^4256|^4257|^4258|^4259|^428': 'chf',
        
        # Cardiac arrhythmias
        r'^4260|^42613|^4267|^4269|^42610|^42612|^4270|^4271|^4272|^4273|^4274|^4276|^4277|^4278|^4279|^7850|^99601|^99604|^V450|^V533': 'cardiac_arrhythmias',
        
        # Valvular disease
        r'^0932|^394|^395|^396|^397|^424|^7463|^7464|^7465|^7466|^V422|^V433': 'valvular_disease',
        
        # Pulmonary circulation disorders
        r'^4150|^4151|^416|^4170|^4178|^4179': 'pulm_circulation',
        
        # Peripheral vascular disorders
        r'^0930|^4373|^440|^441|^4431|^4432|^4433|^4434|^4435|^4436|^4437|^4438|^4439|^4471|^5571|^5579|^V434': 'pvd',
        
        # Hypertension, uncomplicated
        r'^401': 'htn_uncomplicated',
        
        # Hypertension, complicated
        r'^402|^403|^404|^405': 'htn_complicated',
        
        # Paralysis
        r'^3341|^342|^343|^3440|^3441|^3442|^3443|^3444|^3445|^3446|^3449': 'paralysis',
        
        # Other neurological disorders
        r'^3319|^3320|^3321|^3334|^3335|^33392|^334|^335|^3362|^340|^341|^345|^3481|^3483|^7803|^7843': 'other_neuro',
        
        # Chronic pulmonary disease
        r'^4168|^4169|^490|^491|^492|^493|^494|^495|^496|^497|^498|^499|^500|^501|^502|^503|^504|^505|^5064|^5081|^5088': 'chronic_pulm_disease',
        
        # Diabetes, uncomplicated
        r'^2500|^2501|^2502|^2503': 'diabetes_uncomplicated',
        
        # Diabetes, complicated
        r'^2504|^2505|^2506|^2507|^2508|^2509': 'diabetes_complicated',
        
        # Hypothyroidism
        r'^2409|^243|^244|^2461|^2468': 'hypothyroid',
        
        # Renal failure
        r'^40301|^40311|^40391|^40402|^40403|^40412|^40413|^40492|^40493|^585|^586|^5880|^V420|^V451|^V56': 'renal_failure',
        
        # Liver disease
        r'^07022|^07023|^07032|^07033|^07044|^07054|^0706|^0709|^4560|^4561|^4562|^570|^571|^5722|^5723|^5724|^5725|^5726|^5727|^5728|^5733|^5734|^5738|^5739|^V427': 'liver_disease',
        
        # Peptic ulcer disease excluding bleeding
        r'^5317|^5319|^5327|^5329|^5337|^5339|^5347|^5349': 'pud',
        
        # AIDS/HIV
        r'^042|^043|^044': 'aids',
        
        # Lymphoma
        r'^200|^201|^202|^2030|^2386': 'lymphoma',
        
        # Rheumatoid arthritis/collagen vascular diseases
        r'^446|^7010|^7100|^7101|^7102|^7103|^7104|^7108|^7109|^7112|^714|^7193|^720|^725|^7285|^72889|^72930': 'rheumatic',
        
        # Coagulopathy
        r'^286|^2871|^2873|^2874|^2875': 'coagulopathy',
        
        # Obesity
        r'^2780': 'obesity',
        
        # Weight loss
        r'^260|^261|^262|^263|^7832|^7994': 'weight_loss',
        
        # Fluid and electrolyte disorders
        r'^2536|^276': 'fluid',
        
        # Blood loss anemia
        r'^2800': 'blood_loss_anemia',
        
        # Deficiency anemia
        r'^2801|^2802|^2803|^2804|^2805|^2806|^2807|^2808|^2809|^281': 'deficiency_anemia',
        
        # Alcohol abuse
        r'^2652|^2911|^2912|^2913|^2915|^2916|^2917|^2918|^2919|^3030|^3039|^3050|^3575|^4255|^5353|^5710|^5711|^5712|^5713|^980|^V113': 'alcohol',
        
        # Drug abuse
        r'^292|^304|^3052|^3053|^3054|^3055|^3056|^3057|^3058|^3059|^V6542': 'drug_abuse',
        
        # Psychoses
        r'^2938|^295|^29604|^29614|^29644|^29654|^297|^298': 'psychoses',
        
        # Depression
        r'^2962|^2963|^2965|^3004|^309|^311': 'depression'
    }

    ICD_10_ELIXHAUSER_MAPPING = {
        # Congestive heart failure
        r'^I099|^I110|^I130|^I132|^I255|^I420|^I425|^I426|^I427|^I428|^I429|^I43|^I50|^P290': 'chf',
        
        # Cardiac arrhythmias
        r'^I441|^I442|^I443|^I456|^I459|^I47|^I48|^I49|^R000|^R001|^R008|^T821|^Z450|^Z950': 'cardiac_arrhythmias',
        
        # Valvular disease
        r'^A520|^I05|^I06|^I07|^I08|^I091|^I098|^I34|^I35|^I36|^I37|^I38|^I39|^Q230|^Q231|^Q232|^Q233|^Z952|^Z953|^Z954': 'valvular_disease',
        
        # Pulmonary circulation disorders
        r'^I26|^I27|^I280|^I288|^I289': 'pulm_circulation',
        
        # Peripheral vascular disorders
        r'^I70|^I71|^I731|^I738|^I739|^I771|^I790|^I792|^K551|^K558|^K559|^Z958|^Z959': 'pvd',
        
        # Hypertension, uncomplicated
        r'^I10': 'htn_uncomplicated',
        
        # Hypertension, complicated
        r'^I11|^I12|^I13|^I15': 'htn_complicated',
        
        # Paralysis
        r'^G041|^G114|^G801|^G802|^G81|^G82|^G830|^G831|^G832|^G833|^G834|^G839': 'paralysis',
        
        # Other neurological disorders
        r'^G10|^G11|^G12|^G13|^G20|^G21|^G22|^G254|^G255|^G312|^G318|^G319|^G32|^G35|^G36|^G37|^G40|^G41|^G931|^G934|^R470|^R56': 'other_neuro',
        
        # Chronic pulmonary disease
        r'^I278|^I279|^J40|^J41|^J42|^J43|^J44|^J45|^J46|^J47|^J60|^J61|^J62|^J63|^J64|^J65|^J66|^J67|^J684|^J701|^J703': 'chronic_pulm_disease',
        
        # Diabetes, uncomplicated
        r'^E100|^E101|^E109|^E110|^E111|^E119|^E120|^E121|^E129|^E130|^E131|^E139|^E140|^E141|^E149': 'diabetes_uncomplicated',
        
        # Diabetes, complicated
        r'^E102|^E103|^E104|^E105|^E106|^E107|^E108|^E112|^E113|^E114|^E115|^E116|^E117|^E118|^E122|^E123|^E124|^E125|^E126|^E127|^E128|^E132|^E133|^E134|^E135|^E136|^E137|^E138|^E142|^E143|^E144|^E145|^E146|^E147|^E148': 'diabetes_complicated',
        
        # Hypothyroidism
        r'^E00|^E01|^E02|^E03|^E890': 'hypothyroid',
        
        # Renal failure
        r'^I120|^I131|^N18|^N19|^N250|^Z490|^Z491|^Z492|^Z940|^Z992': 'renal_failure',
        
        # Liver disease
        r'^B18|^I85|^I864|^I982|^K70|^K711|^K713|^K714|^K715|^K717|^K72|^K73|^K74|^K760|^K762|^K763|^K764|^K765|^K766|^K767|^K768|^K769|^Z944': 'liver_disease',
        
        # Peptic ulcer disease excluding bleeding
        r'^K257|^K259|^K267|^K269|^K277|^K279|^K287|^K289': 'pud',
        
        # AIDS/HIV
        r'^B20|^B21|^B22|^B24': 'aids',
        
        # Lymphoma
        r'^C81|^C82|^C83|^C84|^C85|^C88|^C96|^C900|^C902': 'lymphoma',
        
        # Rheumatoid arthritis/collagen vascular diseases
        r'^L940|^L941|^L943|^M05|^M06|^M08|^M120|^M123|^M30|^M310|^M311|^M312|^M313|^M32|^M33|^M34|^M35|^M45|^M461|^M468|^M469': 'rheumatic',
        
        # Coagulopathy
        r'^D65|^D66|^D67|^D68|^D691|^D693|^D694|^D695|^D696': 'coagulopathy',
        
        # Obesity
        r'^E66': 'obesity',
        
        # Weight loss
        r'^E40|^E41|^E42|^E43|^E44|^E45|^E46|^R634|^R64': 'weight_loss',
        
        # Fluid and electrolyte Disorders
        r'^E222|^E86|^E87': 'fluid',
        
        # Blood loss anemia
        r'^D500': 'blood_loss_anemia',
        
        # Deficiency anemia
        r'^D508|^D509|^D51|^D52|^D53': 'deficiency_anemia',
        
        # Alcohol abuse
        r'^F10|^E52|^G621|^I426|^K292|^K700|^K703|^K709|^T51|^Z502|^Z714|^Z721': 'alcohol',
        
        # Drug abuse
        r'^F11|^F12|^F13|^F14|^F15|^F16|^F18|^F19|^Z715|^Z722': 'drug_abuse',
        
        # Psychoses
        r'^F20|^F22|^F23|^F24|^F25|^F28|^F29|^F302|^F312|^F315': 'psychoses',
        
        # Depression
        r'^F204|^F313|^F314|^F315|^F32|^F33|^F341|^F412|^F432': 'depression'
    }
    
    VAN_WALRAVEN_WEIGHTS = {
        'chf': 7,
        'cardiac_arrhythmias': 5,
        'valvular_disease': -1,
        'pulm_circulation': 4,
        'pvd': 2,
        'htn_uncomplicated': 0,
        'htn_complicated': 0,
        'paralysis': 7,
        'other_neuro': 6,
        'chronic_pulm_disease': 3,
        'diabetes_uncomplicated': 0,
        'diabetes_complicated': 0,
        'hypothyroid': 0,
        'renal_failure': 5,
        'liver_disease': 11,
        'pud': 0,
        'aids': 0,
        'lymphoma': 9,
        'rheumatic': 0,
        'coagulopathy': 3,
        'obesity': -4,
        'weight_loss': 6,
        'fluid': 5,
        'blood_loss_anemia': -2,
        'deficiency_anemia': -2,
        'alcohol': 0,
        'drug_abuse': -7,
        'psychoses': 0,
        'depression': -3
    }
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

        # Elixhauser comorbidities based on ICD-9 codes
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
            .astype('Int64')  
            .rename_axis(columns = None)
            .reset_index()
        )

        # Elixhauser comorbidities based on ICD-10 codes
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
            .astype('Int64')
            .rename_axis(columns = None)
            .reset_index()  
        )

        all_columns_elix = ['PatientID'] + list(ICD_9_EXLIXHAUSER_MAPPING.values())
        
        # Reindex both dataframes to have all columns, filling missing ones with 0
        df9_elix_aligned = df9_elix.reindex(columns = all_columns_elix, fill_value = 0)
        df10_elix_aligned = df10_elix.reindex(columns = all_columns_elix, fill_value = 0)

        # Combine Elixhauser comorbidity dataframes for ICD-9 and ICD-10
        df_elix_combined = pd.concat([df9_elix_aligned, df10_elix_aligned]).groupby('PatientID').max().reset_index()

        # Calculate van Walraven score
        van_walraven_score = df_elix_combined.drop('PatientID', axis=1).mul(VAN_WALRAVEN_WEIGHTS).sum(axis=1)
        df_elix_combined['van_walraven_score'] = van_walraven_score

        # Start with index_date_df to ensure all PatientIDs are included
        final_df = index_date_df[['PatientID']].copy()
        final_df = pd.merge(final_df, df_elix_combined, on = 'PatientID', how = 'left')

        binary_columns = [col for col in final_df.columns 
                if col not in ['PatientID', 'van_walraven_score']]
        final_df[binary_columns] = final_df[binary_columns].fillna(0).astype('Int64')

        # Check for duplicate PatientIDs
        if len(final_df) > final_df['PatientID'].nunique():
            duplicate_ids = final_df[final_df.duplicated(subset = ['PatientID'], keep = False)]['PatientID'].unique()
            logging.warning(f"Duplicate PatientIDs found: {duplicate_ids}")

        logging.info(f"Successfully processed Diagnosis.csv file with final shape: {final_df.shape} and unique PatientIDs: {(final_df['PatientID'].nunique())}")
        return final_df

    except Exception as e:
        logging.error(f"Error processing Diagnosis.csv file: {e}")
        return None

# TESTING
df = pd.read_csv('data_nsclc/Enhanced_AdvancedNSCLC.csv')
a = process_diagnosis(file_path="data_nsclc/Diagnosis.csv",
                       index_date_df= df,
                       index_date_column= 'AdvancedDiagnosisDate',
                       days_before=None,
                       days_after=30)

embed()