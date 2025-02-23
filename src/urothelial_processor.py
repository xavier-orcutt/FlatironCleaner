import pandas as pd
import numpy as np
import logging
import re 
from typing import Optional

logging.basicConfig(
    level = logging.INFO,                                 
    format = '%(asctime)s - %(levelname)s - %(message)s'  
)

class DataProcessorUrothelial:
    
    GROUP_STAGE_MAPPING = {
        'Stage IV': 'Stage IV',
        'Stage IVA': 'Stage IV',
        'Stage IVB': 'Stage IV',
        'Stage III': 'Stage III',
        'Stage IIIA': 'Stage III',
        'Stage IIIB': 'Stage III',
        'Stage II': 'Stage II',
        'Stage I': 'Stage I',
        'Stage 0is': 'Stage 0',
        'Stage 0a': 'Stage 0',
        'Unknown/not documented': 'unknown'
    }

    T_STAGE_MAPPING = {
        'T4': 'T4',
        'T4a': 'T4',
        'T4b': 'T4',
        'T3': 'T3',
        'T3a': 'T3',
        'T3b': 'T3',
        'T2': 'T2',
        'T2a': 'T2',
        'T2b': 'T2',
        'T1': 'T1',
        'T0': 'T0',
        'Ta': 'Ta',
        'Tis': 'Tis',
        'TX': 'TX',
        'Unknown/not documented': 'unknown'
    }

    M_STAGE_MAPPING = {
        'M1': 'M1',
        'M1a': 'M1',
        'M1b': 'M1',
        'M0': 'M0',
        'MX': 'MX',
        'Unknown/not documented': 'unknown'
    }

    SURGERY_TYPE_MAPPING = {
        'Cystoprostatectomy': 'bladder',
        'Complete (radical) cystectomy': 'bladder',
        'Partial cystectomy': 'bladder',
        'Cystectomy, NOS': 'bladder',
        'Nephroureterectomy': 'upper',
        'Nephrectomy': 'upper',
        'Ureterectomy': 'upper', 
        'Urethrectomy': 'other',
        'Other': 'other',
        'Unknown/not documented': 'unknown', 
        np.nan: 'unknown'
    }

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

    ICD_9_EXLIXHAUSER_MAPPING = {
        # Congestive Heart Failure
        r'^39891|^40201|^40211|^40291|^40401|^40403|^40411|^40413|^40491|^40493|^4254|^4255|^4257|^4258|^4259|^428': 'CHF',
        
        # Cardiac Arrhythmias
        r'^4260|^42613|^4267|^4269|^4261|^427|^7850|^99601|^V450|^V533': 'Arrhythmia',
        
        # Valvular Disease
        r'^0932|^394|^395|^396|^397|^424|^7463|^7464|^7465|^7466|^V422|^V433': 'Valvular',
        
        # Pulmonary Circulation Disorders
        r'^4150|^4151|^416|^4170|^4178|^4179': 'Pulmonary',
        
        # Peripheral Vascular Disorders
        r'^440|^441|^442|^443|^4471|^7854|^V434': 'PVD',
        
        # Hypertension Uncomplicated
        r'^401': 'HTN',
        
        # Hypertension Complicated
        r'^402|^403|^404|^405': 'HTNcx',
        
        # Paralysis
        r'^3341|^342|^343|^3440|^3441|^3442|^3443|^3444|^3445|^3446|^3449': 'Paralysis',
        
        # Other Neurological Disorders
        r'^3319|^332|^333|^334|^335|^336|^337|^340|^341|^345|^3481|^3483|^7803|^7843': 'Neuro',
        
        # Chronic Pulmonary Disease
        r'^4168|^4169|^490|^491|^492|^493|^494|^495|^496|^497|^498|^499|^500|^501|^502|^503|^504|^505|^5064|^5081|^5088': 'COPD',
        
        # Diabetes Uncomplicated
        r'^2500|^2501|^2502|^2503': 'DM',
        
        # Diabetes Complicated
        r'^2504|^2505|^2506|^2507|^2508|^2509': 'DMcx',
        
        # Hypothyroidism
        r'^243|^244|^2461|^2468': 'Hypothyroid',
        
        # Renal Failure
        r'^40301|^40311|^40391|^40402|^40403|^40412|^40413|^40492|^40493|^585|^586|^V420|^V451|^V56': 'Renal',
        
        # Liver Disease
        r'^07022|^07023|^07032|^07033|^07044|^07054|^0706|^0709|^4560|^4561|^4562|^571|^5722|^5723|^5724|^5728|^5730|^5731|^5732|^5733|^5738|^5739|^V427': 'Liver',
        
        # Peptic Ulcer Disease excluding bleeding
        r'^5317|^5319|^5327|^5329|^5337|^5339|^5347|^5349': 'PUD',
        
        # AIDS/HIV
        r'^042|^043|^044': 'AIDS',
        
        # Lymphoma
        r'^200|^201|^202|^2030|^2386': 'Lymphoma',
        
        # Rheumatoid Arthritis/collagen
        r'^446|^7010|^710|^7112|^714|^7193|^720|^725|^7285|^72889|^72930': 'Rheumatic',
        
        # Coagulopathy
        r'^286|^2871|^2873|^2874|^2875': 'Coagulopathy',
        
        # Obesity
        r'^2780': 'Obesity',
        
        # Weight Loss
        r'^260|^261|^262|^263|^7832|^7994': 'WeightLoss',
        
        # Fluid and Electrolyte Disorders
        r'^2536|^276': 'Fluid',
        
        # Blood Loss Anemia
        r'^2800': 'BloodLoss',
        
        # Deficiency Anemia
        r'^2801|^2808|^2809|^281|^2859': 'DefAnemia',
        
        # Alcohol Abuse
        r'^2911|^2912|^2915|^2918|^2919|^303|^3050|^3575|^4255|^5353|^5710|^5711|^5712|^5713|^980|^V113': 'Alcohol',
        
        # Drug Abuse
        r'^292|^304|^3052|^3053|^3054|^3055|^3056|^3057|^3058|^3059|^V6542': 'DrugAbuse',
        
        # Psychoses
        r'^293|^294|^295|^296|^297|^298': 'Psychoses',
        
        # Depression
        r'^2962|^2963|^2965|^3004|^309|^311': 'Depression'
    }

    ICD_10_ELIXHAUSER_MAPPING = {
        # Congestive Heart Failure
        r'^I099|^I110|^I130|^I132|^I255|^I420|^I425|^I426|^I427|^I428|^I429|^I43|^I50|^P290': 'CHF',
        
        # Cardiac Arrhythmias
        r'^I441|^I442|^I443|^I456|^I459|^I47|^I48|^I49|^R000|^R001|^R008|^T821|^Z450|^Z950': 'Arrhythmia',
        
        # Valvular Disease
        r'^A520|^I05|^I06|^I07|^I08|^I091|^I098|^I34|^I35|^I36|^I37|^I38|^I39|^Q230|^Q231|^Q232|^Q233|^Z952|^Z953|^Z954': 'Valvular',
        
        # Pulmonary Circulation Disorders
        r'^I26|^I27|^I280|^I288|^I289': 'Pulmonary',
        
        # Peripheral Vascular Disorders
        r'^I70|^I71|^I731|^I738|^I739|^I771|^I790|^I792|^K551|^K558|^K559|^Z958|^Z959': 'PVD',
        
        # Hypertension Uncomplicated
        r'^I10': 'HTN',
        
        # Hypertension Complicated
        r'^I11|^I12|^I13|^I15': 'HTNcx',
        
        # Paralysis
        r'^G041|^G114|^G801|^G802|^G81|^G82|^G830|^G831|^G832|^G833|^G834|^G839': 'Paralysis',
        
        # Other Neurological Disorders
        r'^G10|^G11|^G12|^G13|^G20|^G21|^G22|^G25|^G31|^G32|^G35|^G36|^G37|^G40|^G41|^G931|^G934|^R470': 'Neuro',
        
        # Chronic Pulmonary Disease
        r'^I278|^I279|^J40|^J41|^J42|^J43|^J44|^J45|^J46|^J47|^J60|^J61|^J62|^J63|^J64|^J65|^J66|^J67|^J684|^J701|^J703': 'COPD',
        
        # Diabetes Uncomplicated
        r'^E100|^E101|^E109|^E110|^E111|^E119|^E120|^E121|^E129|^E130|^E131|^E139|^E140|^E141|^E149': 'DM',
        
        # Diabetes Complicated
        r'^E102|^E103|^E104|^E105|^E106|^E107|^E108|^E112|^E113|^E114|^E115|^E116|^E117|^E118|^E122|^E123|^E124|^E125|^E126|^E127|^E128|^E132|^E133|^E134|^E135|^E136|^E137|^E138|^E142|^E143|^E144|^E145|^E146|^E147|^E148': 'DMcx',
        
        # Hypothyroidism
        r'^E00|^E01|^E02|^E03|^E890': 'Hypothyroid',
        
        # Renal Failure
        r'^I120|^I131|^N18|^N19|^N250|^Z490|^Z491|^Z492|^Z940|^Z992': 'Renal',
        
        # Liver Disease
        r'^B18|^I85|^I864|^I982|^K70|^K711|^K713|^K714|^K715|^K717|^K72|^K73|^K74|^K760|^K762|^K763|^K764|^K765|^K766|^K767|^K768|^K769|^Z944': 'Liver',
        
        # Peptic Ulcer Disease excluding bleeding
        r'^K257|^K259|^K267|^K269|^K277|^K279|^K287|^K289': 'PUD',
        
        # AIDS/HIV
        r'^B20|^B21|^B22|^B24': 'AIDS',
        
        # Lymphoma
        r'^C81|^C82|^C83|^C84|^C85|^C88|^C96|^C900|^C902': 'Lymphoma',
        
        # Rheumatoid Arthritis/collagen
        r'^L940|^L941|^L943|^M05|^M06|^M08|^M120|^M123|^M30|^M310|^M311|^M312|^M313|^M32|^M33|^M34|^M35|^M45|^M461|^M468|^M469': 'Rheumatic',
        
        # Coagulopathy
        r'^D65|^D66|^D67|^D68|^D691|^D693|^D694|^D695|^D696': 'Coagulopathy',
        
        # Obesity
        r'^E66': 'Obesity',
        
        # Weight Loss
        r'^E40|^E41|^E42|^E43|^E44|^E45|^E46|^R634|^R64': 'WeightLoss',
        
        # Fluid and Electrolyte Disorders
        r'^E222|^E86|^E87': 'Fluid',
        
        # Blood Loss Anemia
        r'^D500': 'BloodLoss',
        
        # Deficiency Anemia
        r'^D508|^D509|^D51|^D52|^D53': 'DefAnemia',
        
        # Alcohol Abuse
        r'^F10|^E52|^G621|^I426|^K292|^K700|^K703|^K709|^T51|^Z502|^Z714|^Z721': 'Alcohol',
        
        # Drug Abuse
        r'^F11|^F12|^F13|^F14|^F15|^F16|^F18|^F19|^Z715|^Z722': 'DrugAbuse',
        
        # Psychoses
        r'^F20|^F22|^F23|^F24|^F25|^F28|^F29|^F30|^F31': 'Psychoses',
        
        # Depression
        r'^F204|^F313|^F314|^F315|^F32|^F33|^F341|^F412|^F432': 'Depression'
    }
    
    VAN_WALRAVEN_WEIGHTS = {
        'CHF': 7,
        'Arrhythmia': 5,
        'Valvular': -1,
        'Pulmonary': 4,
        'PVD': 2,
        'HTN': 0,
        'HTNcx': 0,
        'Paralysis': 7,
        'Neuro': 6,
        'COPD': 3,
        'DM': 0,
        'DMcx': 0,
        'Hypothyroid': 0,
        'Renal': 5,
        'Liver': 11,
        'PUD': 0,
        'AIDS': 0,
        'Lymphoma': 9,
        'Rheumatic': 0,
        'Coagulopathy': 3,
        'Obesity': -4,
        'WeightLoss': 6,
        'Fluid': 5,
        'BloodLoss': -2,
        'DefAnemia': -2,
        'Alcohol': 0,
        'DrugAbuse': -7,
        'Psychoses': 0,
        'Depression': -3
    }

    ICD_9_METS_MAPPING = {
        # Lymph Nodes
        r'^196|^1960|^1961|^1962|^1963|^1965|^1966|^1968|^1969': 'lymph_mets',
        
        # Respiratory
        r'^1970|^1971|^1972|^1973': 'lung_mets',
        r'^1972': 'pleura_mets',
        
        # Liver
        r'^1977': 'liver_mets',
        
        # Bone
        r'^1985': 'bone_mets',
        
        # Brain/CNS
        r'^1983|^1984': 'brain_mets',
        
        # Adrenal
        r'^1987': 'adrenal_mets',
        
        # Peritoneum/Retroperitoneum
        r'^1976': 'peritoneum_mets',
        
        # Other GI tract
        r'^1974|^1975|^1978': 'gi_mets',
        
        # Other Sites
        r'^1980|^1981|^1982|^1986|^1988|^19889': 'other_mets',
        
        # Generalized/Unspecified
        r'^1990|^1991': 'unspecified_mets'
    }

    ICD_10_METS_MAPPING = {
        # Lymph Nodes
        r'^C770|^C771|^C772|^C773|^C774|^C775|^C778|^C779': 'lymph_mets',

        # Respiratory
        r'^C780|^C781|^C782': 'lung_mets',
        r'^C782': 'pleura_mets',

        # Liver
        r'^C787': 'liver_mets',

        # Bone
        r'^C795': 'bone_mets',

        # Brain/CNS
        r'^C793|^C794': 'brain_mets',

        # Adrenal
        r'^C797': 'adrenal_mets',

        # Peritoneum/Retroperitoneum
        r'^C786': 'peritoneum_mets',

        # Other GI tract
        r'^C784|^C785|^C788': 'gi_mets',

        # Other Sites
        r'^C780|^C781|^C786|^C798|^C799': 'other_mets',

        # Generalized/Unspecified
        r'^C800|^C801': 'unspecified_mets'
    }

    INSURANCE_MAPPING = {
        'Commercial Health Plan': 'commercial',
        'Medicare': 'medicare',
        'Medicaid': 'medicaid',
        'Other Payer - Type Unknown': 'other_insurance',
        'Other Government Program': 'other_insurance',
        'Patient Assistance Program': 'other_insurance',
        'Self Pay': 'other_insurance',
        'Workers Compensation': 'other_insurance'
    }

    def __init__(self):
        self.enhanced_df = None
        self.demographics_df = None
        self.practice_df = None
        self.mortality_df = None 
        self.biomarkers_df = None
        self.ecog_df = None
        self.vitals_df = None
        self.labs_df = None
        self.medications_df = None
        self.diagnosis_df = None
        self.insurance_df = None

    def process_enhanced_adv(self,
                             file_path: str,
                             patient_ids: list = None,
                             drop_stages: bool = True, 
                             drop_surgery_type: bool = True,
                             drop_dates: bool = True) -> pd.DataFrame: 
        """
        Processes Enhanced_AdvUrothelial.csv to standardize categories, consolidate 
        staging information, and calculate time-based metrics between key clinical events.

        Parameters
        ----------
        file_path : str
            Path to Enhanced_AdvUrothelial.csv file
        patient_ids : list, optional
            List of specific PatientIDs to process. If None, processes all patients
        drop_stages : bool, default=True
            If True, drops original staging columns (GroupStage, TStage, and MStage) after creating modified versions
        drop_surgery_type : bool, default=True
            If True, drops original surgery type after creating modified version
        drop_dates : bool, default=True
            If True, drops date columns (DiagnosisDate, AdvancedDiagnosisDate, and SurgeryDate) after calculating durations

        Returns
        -------
        pd.DataFrame
            - PatientID : object
                unique patient identifier
            - PrimarySite : category
                anatomical site of cancer
            - SmokingStatus : category
                smoking history
            - Surgery : Int64
                binary indicator (0/1) for whether patient had surgery
            - SurgeryType_mod : category
                consolidated surgery type
            - days_diagnosis_to_surgery : float
                days from diagnosis to surgery
            - DiseaseGrade : category
                tumor grade
            - NStage : category
                lymph node staging
            - GroupStage_mod : category
                consolidated overall staging (0-IV, Unknown)
            - TStage_mod : category
                consolidated tumor staging (T0-T4, Ta, Tis, TX, Unknown)
            - MStage_mod : category
                consolidated metastasis staging (M0, M1, MX, Unknown)
            - days_diagnosis_to_adv : float
                days from diagnosis to advanced disease 
            - adv_diagnosis_year : category
                year of advanced diagnosis 
            - days_diagnosis_to_surgery : float
                days from diagnosis to surgery 
            
            Original staging and date columns retained if respective drop_* = False

        Notes
        -----
        Duplicate PatientIDs are logged as warnings if found
        Processed DataFrame is stored in self.enhanced_df
        """
        try:
            df = pd.read_csv(file_path)
            logging.info(f"Successfully read Enhanced_AdvUrothelial.csv file with shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")

            # Filter for specific PatientIDs if provided
            if patient_ids is not None:
                logging.info(f"Filtering for {len(patient_ids)} specific PatientIDs")
                df = df[df['PatientID'].isin(patient_ids)]
                logging.info(f"Successfully filtered Enhanced_AdvUrothelial.csv file with shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")
        
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

            # Recode stage variables using class-level mapping and create new column
            df['GroupStage_mod'] = df['GroupStage'].map(self.GROUP_STAGE_MAPPING).astype('category')
            df['TStage_mod'] = df['TStage'].map(self.T_STAGE_MAPPING).astype('category')
            df['MStage_mod'] = df['MStage'].map(self.M_STAGE_MAPPING).astype('category')

            # Drop original stage variables if specified
            if drop_stages:
                df = df.drop(columns=['GroupStage', 'TStage', 'MStage'])

            # Recode surgery type variable using class-level mapping and create new column
            df['SurgeryType_mod'] = df['SurgeryType'].map(self.SURGERY_TYPE_MAPPING).astype('category')

            # Drop original surgery type variable if specified
            if drop_surgery_type:
                df = df.drop(columns=['SurgeryType'])

            # Convert date columns
            date_cols = ['DiagnosisDate', 'AdvancedDiagnosisDate', 'SurgeryDate']
            for col in date_cols:
                df[col] = pd.to_datetime(df[col])
            
            # Convert boolean column to binary (0/1)
            df['Surgery'] = df['Surgery'].astype(int)

            # Generate new variables 
            df['days_diagnosis_to_adv'] = (df['AdvancedDiagnosisDate'] - df['DiagnosisDate']).dt.days
            df['adv_diagnosis_year'] = pd.Categorical(df['AdvancedDiagnosisDate'].dt.year)
            df['days_diagnosis_to_surgery'] = (df['SurgeryDate'] - df['DiagnosisDate']).dt.days
    
            if drop_dates:
                df = df.drop(columns = ['AdvancedDiagnosisDate', 'DiagnosisDate', 'SurgeryDate'])

            # Check for duplicate PatientIDs
            if len(df) > df['PatientID'].nunique():
                logging.error(f"Duplicate PatientIDs found")
                return None

            logging.info(f"Successfully processed Enhanced_AdvUrothelial.csv file with final shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")
            self.enhanced_df = df
            return df

        except Exception as e:
            logging.error(f"Error processing Enhanced_AdvUrothelial.csv file: {e}")
            return None
    
    def process_demographics(self, 
                             file_path: str,
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
        Imputation:
            - if Race='Hispanic or Latino', value is replaced with NaN
            - if Race='Hispanic or Latino', Ethnicity is set to 'Hispanic or Latino'
        Ages calculated as <18 or >120 are removed as implausible
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

            # Region processing
            # Group states into Census-Bureau regions  
            df['region'] = (df['State']
                            .map(self.STATE_REGIONS)
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
            self.demographics_df = df
            return df

        except Exception as e:
            logging.error(f"Error processing demographics file: {e}")
            return None
        

    def process_practice(self,
                         file_path: str,
                         patient_ids: list = None) -> pd.DataFrame:
        """
        Processes Practice.csv to consolidate practice types per patient into a single categorical value indicating academic, community, or both settings.

        Parameters
        ----------
        file_path : str
            Path to Practice.csv file
        patient_ids : list, optional
            List of specific PatientIDs to process. If None, processes all patients

        Returns
        -------
        pd.DataFrame
            - PatientID : object
                unique patient identifier  
            - PracticeType_mod : category
                practice setting (ACADEMIC, COMMUNITY, or BOTH)
       
        Notes
        -----
        - Duplicate PatientIDs are logged as warnings if found
        - Processed DataFrame is stored in self.practice_df
        """
        try:
            df = pd.read_csv(file_path)
            logging.info(f"Successfully read Practice.csv file with shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")

            # Filter for specific PatientIDs if provided
            if patient_ids is not None:
                logging.info(f"Filtering for {len(patient_ids)} specific PatientIDs")
                df = df[df['PatientID'].isin(patient_ids)]
                logging.info(f"Successfully filtered Practice.csv file with shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")

            df = df[['PatientID', 'PracticeType']]

            # Group by PatientID and get set of unique PracticeTypes
            grouped = df.groupby('PatientID')['PracticeType'].unique()
            new_df = pd.DataFrame(grouped).reset_index()

            # Function to determine the modified practice type
            def get_practice_type(practice_types):
                if len(practice_types) > 1:
                    return 'BOTH'
                return practice_types[0]
            
            # Apply the function to the column containing sets
            new_df['PracticeType_mod'] = new_df['PracticeType'].apply(get_practice_type).astype('category')

            new_df = new_df[['PatientID', 'PracticeType_mod']]

            # Check for duplicate PatientIDs
            if len(new_df) > new_df['PatientID'].nunique():
                logging.error(f"Duplicate PatientIDs found")
                return None
            
            logging.info(f"Successfully processed Practice.csv file with final shape: {new_df.shape} and unique PatientIDs: {(new_df['PatientID'].nunique())}")
            self.practice_df = new_df
            return new_df

        except Exception as e:
            logging.error(f"Error processing practice file: {e}")
            return None
        
    def process_mortality(self,
                          file_path: str,
                          index_date_df: pd.DataFrame,
                          index_date_column: str,
                          visit_path: str = None, 
                          telemedicine_path: str = None, 
                          biomarker_path: str = None, 
                          oral_path: str = None,
                          progression_path: str = None,
                          drop_dates: bool = True) -> pd.DataFrame:
        """
        Processes Enhanced_Mortality_V2.csv by cleaning data types, calculating time 
        from index date to death/censor, and determining mortality events. Handles
        incomplete death dates by imputing missing day/month values.

        Parameters
        ----------
        file_path : str
            Path to Enhanced_Mortality_V2.csv file
        index_date_df : pd.DataFrame
            DataFrame containing PatientID and index dates. Only mortality data for PatientIDs present in this DataFrame will be processed
        index_date_column : str
            Column name in index_date_df containing the index date
        df_merge_type : str, default='left'
            Merge type for pd.merge(index_date_df, mortality_data, on = 'PatientID', how = df_merge_type)
        visit_path : str
            Path to Visit.csv file
        telemedicine_path : str
            Path to Telemedicine.csv file
        biomarker_path : str
            Path to Enhanced_AdvUrothelialBiomarkers.csv file
        oral_path : str
            Path to Enhanced_AdvUrothelial_Orals.csv file
        progression_path : str
            Path to Enhanced_AdvUrothelial_Progression.csv file
        drop_dates : bool, default = True
            If True, drops date columns (index_date_column, DateOfDeath, last_ehr_date)   
        
        Returns
        -------
        pd.DataFrame
            - PatientID : object
                unique patient identifier
            - duration : float
                days from index date to death or censor 
            - event : Int64
                mortality status (1 = death, 0 = censored)

        Notes
        ------
        Death date imputation:
        - Missing day : Imputed to 15th of the month
        - Missing month and day : Imputed to July 1st
        Duplicate PatientIDs are logged as warnings if found
        Processed DataFrame is stored in self.mortality_df
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
            logging.info(f"Successfully read Enhanced_Mortality_V2.csv file with shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")

            # When only year is available: Impute to July 1st (mid-year)
            df['DateOfDeath'] = np.where(df['DateOfDeath'].str.len() == 4, df['DateOfDeath'] + '-07-01', df['DateOfDeath'])

            # When only month and year are available: Impute to the 15th day of the month
            df['DateOfDeath'] = np.where(df['DateOfDeath'].str.len() == 7, df['DateOfDeath'] + '-15', df['DateOfDeath'])

            df['DateOfDeath'] = pd.to_datetime(df['DateOfDeath'])

            # Process index dates and merge
            index_date_df[index_date_column] = pd.to_datetime(index_date_df[index_date_column])
            df_death = pd.merge(
                index_date_df[['PatientID', index_date_column]],
                df,
                on = 'PatientID',
                how = 'left'
            )
            
            logging.info(f"Successfully merged Enhanced_Mortality_V2.csv df with index_date_df resulting in shape: {df_death.shape} and unique PatientIDs: {(df_death.PatientID.nunique())}")
                
            # Create event column
            df_death['event'] = df_death['DateOfDeath'].notna().astype('Int64')

            # Initialize df_final
            df_final = df_death

            # Determine last EHR data
            if all(path is None for path in [visit_path, telemedicine_path, biomarker_path, oral_path, progression_path]):
                logging.info("WARNING: At least one of visit_path, telemedicine_path, biomarker_path, oral_path, or progression_path must be provided to calculate duration for those with a missing death date")

            else: 
                if visit_path is not None and telemedicine_path is not None:
                    try:
                        df_visit = pd.read_csv(visit_path)
                        df_tele = pd.read_csv(telemedicine_path)

                        df_visit_tele = (
                            pd.concat([
                                df_visit[['PatientID', 'VisitDate']],
                                df_tele[['PatientID', 'VisitDate']]
                                ]))
                        
                        df_visit_tele['VisitDate'] = pd.to_datetime(df_visit_tele['VisitDate'])

                        df_visit_tele_max = (
                            df_visit_tele
                            .query("PatientID in @index_date_df.PatientID")  
                            .groupby('PatientID', observed = True)['VisitDate']  
                            .max()
                            .to_frame(name = 'last_visit_date')          
                            .reset_index()
                            )
                    except Exception as e:
                        logging.error(f"Error reading Visit.csv and/or Telemedicine.csv files: {e}")
                        return None

                if visit_path is not None and telemedicine_path is None:
                    try: 
                        df_visit = pd.read_csv(visit_path)
                        df_visit['VisitDate'] = pd.to_datetime(df_visit['VisitDate'])

                        df_visit_max = (
                            df_visit
                            .query("PatientID in @index_date_df.PatientID")  
                            .groupby('PatientID', observed = True)['VisitDate']  
                            .max()
                            .to_frame(name = 'last_visit_date')          
                            .reset_index()
                        )
                    except Exception as e:
                        logging.error(f"Error reading Visit.csv file: {e}")
                        return None

                if telemedicine_path is not None and visit_path is None:
                    try: 
                        df_tele = pd.read_csv(telemedicine_path)
                        df_tele['VisitDate'] = pd.to_datetime(df_tele['VisitDate'])

                        df_tele_max = (
                            df_tele
                            .query("PatientID in @index_date_df.PatientID")  
                            .groupby('PatientID', observed = True)['VisitDate']  
                            .max()
                            .to_frame(name = 'last_visit_date')          
                            .reset_index()
                        )
                    except Exception as e:
                        logging.error(f"Error reading Telemedicine.csv file: {e}")
                        return None
                                            
                if biomarker_path is not None:
                    try: 
                        df_biomarker = pd.read_csv(biomarker_path)
                        df_biomarker['SpecimenCollectedDate'] = pd.to_datetime(df_biomarker['SpecimenCollectedDate'])

                        df_biomarker_max = (
                            df_biomarker
                            .query("PatientID in @index_date_df.PatientID")
                            .groupby('PatientID', observed = True)['SpecimenCollectedDate'].max()
                            .to_frame(name = 'last_biomarker_date')
                            .reset_index()
                        )
                    except Exception as e:
                        logging.error(f"Error reading Enhanced_AdvUrothelialBiomarkers.csv file: {e}")
                        return None

                if oral_path is not None:
                    try:
                        df_oral = pd.read_csv(oral_path)
                        df_oral['StartDate'] = pd.to_datetime(df_oral['StartDate'])
                        df_oral['EndDate'] = pd.to_datetime(df_oral['EndDate'])

                        df_oral_max = (
                            df_oral
                            .query("PatientID in @index_date_df.PatientID")
                            .assign(max_date = df_oral[['StartDate', 'EndDate']].max(axis = 1))
                            .groupby('PatientID', observed = True)['max_date'].max()
                            .to_frame(name = 'last_oral_date')
                            .reset_index()
                        )
                    except Exception as e:
                        logging.error(f"Error reading Enhanced_AdvUrothelial_Orals.csv file: {e}")
                        return None

                if progression_path is not None:
                    try: 
                        df_progression = pd.read_csv(progression_path)
                        df_progression['ProgressionDate'] = pd.to_datetime(df_progression['ProgressionDate'])
                        df_progression['LastClinicNoteDate'] = pd.to_datetime(df_progression['LastClinicNoteDate'])

                        df_progression_max = (
                            df_progression
                            .query("PatientID in @index_date_df.PatientID")
                            .assign(max_date = df_progression[['ProgressionDate', 'LastClinicNoteDate']].max(axis = 1))
                            .groupby('PatientID', observed = True)['max_date'].max()
                            .to_frame(name = 'last_progression_date')
                            .reset_index()
                        )
                    except Exception as e:
                        logging.error(f"Error reading Enhanced_AdvUrothelial_Progression.csv file: {e}")
                        return None

                # Create a dictionary to store all available dataframes
                dfs_to_merge = {}

                # Add dataframes to dictionary if they exist
                if visit_path is not None and telemedicine_path is not None:
                    dfs_to_merge['visit_tele'] = df_visit_tele_max
                elif visit_path is not None:
                    dfs_to_merge['visit'] = df_visit_max
                elif telemedicine_path is not None:
                    dfs_to_merge['tele'] = df_tele_max

                if biomarker_path is not None:
                    dfs_to_merge['biomarker'] = df_biomarker_max
                if oral_path is not None:
                    dfs_to_merge['oral'] = df_oral_max
                if progression_path is not None:
                    dfs_to_merge['progression'] = df_progression_max

                # Merge all available dataframes
                if dfs_to_merge:
                    df_last_ehr_activity = None
                    for name, df in dfs_to_merge.items():
                        if df_last_ehr_activity is None:
                            df_last_ehr_activity = df
                        else:
                            df_last_ehr_activity = pd.merge(df_last_ehr_activity, df, on = 'PatientID', how = 'outer')

                if df_last_ehr_activity is not None:
                    # Get the available date columns that exist in our merged dataframe
                    last_date_columns = [col for col in ['last_visit_date', 'last_oral_date', 'last_biomarker_date', 'last_progression_date']
                                        if col in df_last_ehr_activity.columns]
                    logging.info(f"The follwing columns {last_date_columns} are used to calculate the last EHR date")
                    
                    if last_date_columns:
                        single_date = (
                            df_last_ehr_activity
                            .assign(last_ehr_activity = df_last_ehr_activity[last_date_columns].max(axis = 1))
                            .filter(items = ['PatientID', 'last_ehr_activity'])
                        )

                        df_final = pd.merge(df_death, single_date, on = 'PatientID', how = 'left')

            # Calculate duration
            if 'last_ehr_activity' in df_final.columns:
                df_final['duration'] = np.where(df_final['event'] == 0, 
                                                (df_final['last_ehr_activity'] - df_final[index_date_column]).dt.days, 
                                                (df_final['DateOfDeath'] - df_final[index_date_column]).dt.days)
                
                # Drop date varibales if specified
                if drop_dates:               
                    df_final = df_final.drop(columns = [index_date_column, 'DateOfDeath', 'last_ehr_activity'])

                # Check for duplicate PatientIDs
                if len(df_final) > df_final.PatientID.nunique():
                    logging.error(f"Duplicate PatientIDs found")

                logging.info(f"Successfully processed Enhanced_Mortality_V2.csv file with final shape: {df_final.shape} and unique PatientIDs: {(df_final['PatientID'].nunique())}. There are {df_final['duration'].isna().sum()} out of {df_final['PatientID'].nunique()} patients with missing duration values")
                self.mortality_df = df_final
                return df_final
                       
            else: 
                df_final['duration'] = (df_final['DateOfDeath'] - df_final[index_date_column]).dt.days

                # Drop date varibales if specified
                if drop_dates:               
                    df_final = df_final.drop(columns = [index_date_column, 'DateOfDeath'])

                # Check for duplicate PatientIDs
                if len(df_final) > df_final.PatientID.nunique():
                    logging.error(f"Duplicate PatientIDs found")

                logging.info(f"Successfully processed Enhanced_Mortality_V2.csv file with final shape: {df_final.shape} and unique PatientIDs: {(df_final['PatientID'].nunique())}. There are {df_final['duration'].isna().sum()} out of {df_final['PatientID'].nunique()} patients with missing duration values")
                self.mortality_df = df_final
                return df_final

        except Exception as e:
            logging.error(f"Error processing Enhanced_Mortality_V2.csv file: {e}")
            return None
        
    def process_biomarkers(self,
                           file_path: str,
                           index_date_df: pd.DataFrame,
                           index_date_column: str, 
                           days_before: Optional[int] = None,
                           days_after: int = 0) -> pd.DataFrame:
        """
        Processes Enhanced_AdvUrothelialBiomarkers.csv by determining FGFR and PDL1 status for each patient within a specified time window relative to an index date
        For each biomarker:
        - FGFR status is classified as:
            - 'positive' if any test result is positive (ever-positive)
            - 'negative' if any test is negative without positives (only-negative) 
            - 'unknown' if all results are indeterminate
        - PDL1 status follows the same classification logic
        - PDL1 staining percentage is also captured

        Parameters
        ----------
        file_path : str
            Path to Enhanced_AdvUrothelialBiomarkers.csv file
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
            - fgfr_status : category
                positive if ever-positive, negative if only-negative, otherwise unknown
            - pdl1_status : cateogory
                positive if ever-positive, negative if only-negative, otherwise unknown
            - pdl1_staining : category, ordered 
                returns a patient's maximum percent staining for PDL1

        Notes
        ------
        Missing ResultDate is imputed with SpecimenReceivedDate.
        All PatientIDs from index_date_df are included in the output and values will be NaN for patients without any biomarker tests
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
            logging.info(f"Successfully read Enhanced_AdvUrothelialBiomarkers.csv file with shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")

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
            logging.info(f"Successfully merged Enhanced_AdvUrothelialBiomarkers.csv df with index_date_df resulting in shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")
            
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

            # Process FGFR status
            fgfr_df = (
                df_filtered
                .query('BiomarkerName == "FGFR"')
                .groupby('PatientID')['BiomarkerStatus']
                .agg(lambda x: 'positive' if any ('Positive' in val for val in x)
                    else ('negative' if any('Negative' in val for val in x)
                        else 'unknown'))
                .reset_index()
                .rename(columns={'BiomarkerStatus': 'fgfr_status'})
                )
            
            # Process PDL1 status
            pdl1_df = (
                df_filtered
                .query('BiomarkerName == "PDL1"')
                .groupby('PatientID')['BiomarkerStatus']
                .agg(lambda x: 'positive' if any ('PD-L1 positive' in val for val in x)
                    else ('negative' if any('PD-L1 negative/not detected' in val for val in x)
                        else 'unknown'))
                .reset_index()
                .rename(columns={'BiomarkerStatus': 'pdl1_status'})
                )

            # Process PDL1 staining 
            pdl1_staining_df = (
                df_filtered
                .query('BiomarkerName == "PDL1"')
                .query('BiomarkerStatus == "PD-L1 positive"')
                .groupby('PatientID')['PercentStaining']
                .apply(lambda x: x.map(self.PDL1_PERCENT_STAINING_MAPPING))
                .groupby('PatientID')
                .agg('max')
                .to_frame(name = 'pdl1_ordinal_value')
                .reset_index()
                )
            
            # Create reverse mapping to convert back to percentage strings
            reverse_pdl1_dict = {v: k for k, v in self.PDL1_PERCENT_STAINING_MAPPING.items()}
            pdl1_staining_df['pdl1_percent_staining'] = pdl1_staining_df['pdl1_ordinal_value'].map(reverse_pdl1_dict)
            pdl1_staining_df = pdl1_staining_df.drop(columns = ['pdl1_ordinal_value'])

            # Merge dataframes -- start with index_date_df to ensure all PatientIDs are included
            final_df = index_date_df[['PatientID']].copy()
            final_df = pd.merge(final_df, pdl1_df, on = 'PatientID', how = 'left')
            final_df = pd.merge(final_df, pdl1_staining_df, on = 'PatientID', how = 'left')
            final_df = pd.merge(final_df, fgfr_df, on = 'PatientID', how = 'left')

            final_df['pdl1_status'] = final_df['pdl1_status'].astype('category')
            final_df['fgfr_status'] = final_df['fgfr_status'].astype('category')

            staining_dtype = pd.CategoricalDtype(
                categories = ['0%', '< 1%', '1%', '2% - 4%', '5% - 9%', '10% - 19%',
                              '20% - 29%', '30% - 39%', '40% - 49%', '50% - 59%',
                              '60% - 69%', '70% - 79%', '80% - 89%', '90% - 99%', '100%'],
                              ordered = True
                              )
            
            final_df['pdl1_percent_staining'] = final_df['pdl1_percent_staining'].astype(staining_dtype)

            # Check for duplicate PatientIDs
            if len(final_df) > final_df['PatientID'].nunique():
                logging.error(f"Duplicate PatientIDs found")
                return None

            logging.info(f"Successfully processed Enhanced_AdvUrothelialBiomarkers.csv file with final shape: {final_df.shape} and unique PatientIDs: {(final_df['PatientID'].nunique())}")
            self.biomarkers_df = final_df
            return final_df

        except Exception as e:
            logging.error(f"Error processing Enhanced_AdvUrothelialBiomarkers.csv file: {e}")
            return None
        
    def process_ecog(self,
                     file_path: str,
                     index_date_df: pd.DataFrame,
                     index_date_column: str, 
                     days_before: int = 90,
                     days_after: int = 0, 
                     days_before_further: int = 180) -> pd.DataFrame:
        """
        Processes ECOG.csv to determine patient ECOG scores and progression patterns relative 
        to a reference index date. Uses two different time windows for distinct clinical purposes:
        
        1. A smaller window near the index date to find the most clinically relevant ECOG score
           that represents the patient's status at that time point
        2. A larger lookback window to detect clinically significant ECOG progression,
           specifically looking for patients whose condition worsened from ECOG 0-1 to ≥2
        
        This dual-window approach allows for both accurate point-in-time assessment and
        detection of deteriorating performance status over a clinically meaningful period.

        For each patient, finds:
        1. The ECOG score closest to index date (selecting higher score in case of ties)
        2. Whether ECOG newly increased to ≥2 from 0-1 in the lookback period

        Parameters
        ----------
        file_path : str
            Path to ECOG.csv file
        index_date_df : pd.DataFrame
            DataFrame containing PatientID and index dates. Only ECOGs for PatientIDs present in this DataFrame will be processed
        index_date_column : str
            Column name in index_date_df containing the index date
        days_before : int, optional
            Number of days before the index date to include. Must be >= 0. Default: 90
        days_after : int, optional
            Number of days after the index date to include. Must be >= 0. Default: 0
        days_before_futher : int, optional
            Number of days before index date to look for ECOG progression (0-1 to ≥2). Must be >= 0. Consdier
            selecting a larger integer than days_before to capture meaningful clinical deterioration over time.
            Default: 180
            
        Returns
        -------
        pd.DataFrame
            - PatientID : object
                unique patient identifier
            - ecog_index : category, ordered 
                ECOG score (0-5) closest to index date
            - ecog_newly_gte2 : Int64
                binary indicator (0/1) for ECOG increased from 0-1 to ≥2 in 6 months before index

        Notes
        ------
        When multiple ECOG scores are equidistant to index date, the higher score is selected
        All PatientIDs from index_date_df are included in the output and values will be NaN for patients without ECOG values
        Duplicate PatientIDs are logged as warnings if found
        Processed DataFrame is stored in self.ecog_df
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
            logging.info(f"Successfully read ECOG.csv file with shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")

            df['EcogDate'] = pd.to_datetime(df['EcogDate'])
            df['EcogValue'] = pd.to_numeric(df['EcogValue'], errors = 'coerce').astype('Int64')

            index_date_df[index_date_column] = pd.to_datetime(index_date_df[index_date_column])

            # Select PatientIDs that are included in the index_date_df the merge on 'left'
            df = df[df.PatientID.isin(index_date_df.PatientID)]
            df = pd.merge(
                df,
                index_date_df[['PatientID', index_date_column]],
                on = 'PatientID',
                how = 'left'
                )
            logging.info(f"Successfully merged ECOG.csv df with index_date_df resulting in shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")
                        
            # Create new variable 'index_to_ecog' that notes difference in days between ECOG date and index date
            df['index_to_ecog'] = (df['EcogDate'] - df[index_date_column]).dt.days
            
            # Select ECOG that fall within desired before and after index date
            df_closest_window = df[
                (df['index_to_ecog'] <= days_after) & 
                (df['index_to_ecog'] >= -days_before)].copy()

            # Find EcogValue closest to index date within specified window periods
            ecog_index_df = (
                df_closest_window
                .assign(abs_days_to_index = lambda x: abs(x['index_to_ecog']))
                .sort_values(
                    by=['PatientID', 'abs_days_to_index', 'EcogValue'], 
                    ascending=[True, True, False])
                .groupby('PatientID')
                .first()
                .reset_index()
                [['PatientID', 'EcogValue']]
                .rename(columns = {'EcogValue': 'ecog_index'})
                .assign(
                    ecog_index = lambda x: x['ecog_index'].astype(pd.CategoricalDtype(categories = [0, 1, 2, 3, 4, 5], ordered = True))
                    )
                )
            
            # # Process 2: Check for ECOG progression in wider window
            df_progression_window = df[
                    (df['index_to_ecog'] <= days_after) & 
                    (df['index_to_ecog'] >= -days_before_further)].copy()
            
            # Create flag for ECOG newly greater than or equal to 2
            ecog_newly_gte2_df = (
                df_progression_window
                .sort_values(['PatientID', 'EcogDate']) 
                .groupby('PatientID')
                .agg({
                    'EcogValue': lambda x: (
                        # 1. Last ECOG is ≥2
                        (x.iloc[-1] >= 2) and 
                        # 2. Any previous ECOG was 0 or 1
                        any(x.iloc[:-1].isin([0, 1]))
                    )
                })
                .reset_index()
                .rename(columns={'EcogValue': 'ecog_newly_gte2'})
            )

            # Merge dataframes - start with index_date_df to ensure all PatientIDs are included
            final_df = index_date_df[['PatientID']].copy()
            final_df = pd.merge(final_df, ecog_index_df, on = 'PatientID', how = 'left')
            final_df = pd.merge(final_df, ecog_newly_gte2_df, on = 'PatientID', how = 'left')
            
            # Assign datatypes 
            final_df['ecog_index'] = final_df['ecog_index'].astype(pd.CategoricalDtype(categories=[0, 1, 2, 3, 4, 5], ordered=True))
            final_df['ecog_newly_gte2'] = final_df['ecog_newly_gte2'].astype('Int64')

            # Check for duplicate PatientIDs
            if len(final_df) > final_df['PatientID'].nunique():
                logging.error(f"Duplicate PatientIDs found")
                return None
                
            logging.info(f"Successfully processed ECOG.csv file with final shape: {final_df.shape} and unique PatientIDs: {(final_df['PatientID'].nunique())}")
            self.ecog_df = final_df
            return final_df

        except Exception as e:
            logging.error(f"Error processing ECOG.csv file: {e}")
            return None
        

    def process_vitals(self,
                       file_path: str,
                       index_date_df: pd.DataFrame,
                       index_date_column: str, 
                       weight_days_before: int = 90,
                       weight_days_after: int = 0,
                       vital_summary_lookback: int = 180) -> pd.DataFrame:
        """
        Processes Vitals.csv to determine patient BMI, weight, change in weight, and vital sign abnormalities
        within a specified time window relative to an index date. Uses two different time windows for distinct 
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
        weight_days_after : int, optional
            Number of days after the index date to include for weight and BMI calculations. Also used as the end point for 
            vital sign abnormalities and weight change calculations. Must be >= 0. Default: 0
        vital_summary_lookback : int, optional
            Number of days before index date to assess for weight change, hypotension, tachycardia, and fever. Must be >= 0. Default: 180
        
        Returns
        -------
        pd.DataFrame
            - PatientID : object 
                unique patient identifier
            - weight : float
                weight in kg closest to index date within specified window (index_date - weight_days_before) to (index_date + weight_days_after)
            - bmi : float, 
                BMI closest to index date within specified window (index_date - weight_days_before) to (index_date + weight_days_after)
            - percent_change_weight : float
                percentage change in weight over period from (index_date - vital_summary_lookback) to (index_date + weight_days_after), calculated as ((end_weight - start_weight) / start_weight) * 100
            - hypotension : Int64
                binary indicator (0/1) for systolic blood pressure <90 mmHg on ≥2 separate readings between (index_date - vital_summary_lookback) and (index_date + weight_days_after)
            - tachycardia : Int64
                binary indicator (0/1) for heart rate >100 bpm on ≥2 separate readings between (index_date - vital_summary_lookback) and (index_date + weight_days_after)
            - fevers : Int64
                binary indicator (0/1) for temperature >38°C on ≥2 separate readings between (index_date - vital_summary_lookback) and (index_date + weight_days_after)

        Notes
        -----
        BMI is calculated using weight closest to index date within specified window while height outside the specified window may be used. The equation used: weight (kg)/height (m)^2
        BMI <13 are considered implausible and removed
        Vital sign thresholds:
            * Hypotension: systolic BP <90 mmHg
            * Tachycardia: HR >100 bpm
            * Fever: temperature >38°C
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
        if not isinstance(weight_days_after, int) or weight_days_after < 0:
            raise ValueError("weight_days_after must be a non-negative integer")
        if not isinstance(vital_summary_lookback, int) or vital_summary_lookback < 0:
            raise ValueError("vital_summary_lookback must be a non-negative integer")

        try:
            df = pd.read_csv(file_path, low_memory = False)
            logging.info(f"Successfully read Vitals.csv file with shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")

            df['TestDate'] = pd.to_datetime(df['TestDate'])
            df['TestResult'] = pd.to_numeric(df['TestResult'], errors = 'coerce').astype('float')

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
            weight_df = df.query('Test == "body weight"')
            mask_needs_imputation = weight_df['TestResultCleaned'].isna() & weight_df['TestResult'].notna()
            
            imputed_weights = weight_df.loc[mask_needs_imputation, 'TestResult'].apply(
                lambda x: x/2.2046 if x > 100  # Convert to kg if likely lbs (>100)
                else x if x < 60  # Keep as is if likely kg (<60)
                else None  # Leave as null if ambiguous
                )
            
            weight_df.loc[mask_needs_imputation, 'TestResultCleaned'] = imputed_weights
            weight_df = weight_df.query('TestResultCleaned > 0')
            
            df_weight_filtered = weight_df[
                (weight_df['index_to_vital'] <= weight_days_after) & 
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
                (weight_df['index_to_vital'] <= weight_days_after) & 
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

            # Calculate hypotension 
            df_summary_filtered = df[
                (df['index_to_vital'] <= weight_days_after) & 
                (df['index_to_vital'] >= -vital_summary_lookback)].copy()
            
            hypotension_df = (
                df_summary_filtered
                .query("Test == 'systolic blood pressure'")
                .sort_values(['PatientID', 'TestDate'])
                .groupby('PatientID')
                .agg({
                    'TestResult': lambda x: (
                        sum(x < 90) >= 2) # True if 2 readings of systolic <90
                })
                .reset_index()
                .rename(columns = {'TestResult': 'hypotension'})
                )

            # Calculate tachycardia
            tachycardia_df = (
                df_summary_filtered 
                .query("Test == 'heart rate'")
                .sort_values(['PatientID', 'TestDate'])
                .groupby('PatientID')
                .agg({
                    'TestResult': lambda x: (
                        sum(x > 100) >= 2) # True if 2 readings of heart rate >100 
                })
                .reset_index()
                .rename(columns = {'TestResult': 'tachycardia'})
                )

            # Calculate fevers 
            fevers_df = (
                df_summary_filtered 
                .query("Test == 'body temperature'")
            )

            # Any temperature >45 is presumed F, otherwise C
            fevers_df.loc[:, 'TestResult'] = np.where(fevers_df['TestResult'] > 45,
                                                      (fevers_df['TestResult'] - 32) * 5/9,
                                                      fevers_df['TestResult'])

            fevers_df = (
                fevers_df
                .sort_values(['PatientID', 'TestDate'])
                .groupby('PatientID')
                .agg({
                    'TestResult': lambda x: sum(x >= 38) >= 2 # True if 2 readings of temperature >38C
                })
                .reset_index()
                .rename(columns={'TestResult': 'fevers'})
            )

            # Merge dataframes - start with index_date_df to ensure all PatientIDs are included
            final_df = index_date_df[['PatientID']].copy()
            final_df = pd.merge(final_df, weight_index_df, on = 'PatientID', how = 'left')
            final_df = pd.merge(final_df, change_weight_df, on = 'PatientID', how = 'left')
            final_df = pd.merge(final_df, hypotension_df, on = 'PatientID', how = 'left')
            final_df = pd.merge(final_df, tachycardia_df, on = 'PatientID', how = 'left')
            final_df = pd.merge(final_df, fevers_df, on = 'PatientID', how = 'left')

            boolean_columns = ['hypotension', 'tachycardia', 'fevers']
            for col in boolean_columns:
                final_df[col] = final_df[col].fillna(0).astype('Int64')
            
            # Check for duplicate PatientIDs
            if len(final_df) > final_df['PatientID'].nunique():
                logging.error(f"Duplicate PatientIDs found")
                return None

            logging.info(f"Successfully processed Vitals.csv file with final shape: {final_df.shape} and unique PatientIDs: {(final_df['PatientID'].nunique())}")
            self.vitals_df = final_df
            return final_df

        except Exception as e:
            logging.error(f"Error processing Vitals.csv file: {e}")
            return None
        
    def process_labs(self, 
                     file_path: str,
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
        All PatientIDs from index_date_df are included in the output and values will be NaN for patients without lab values 
        Duplicate PatientIDs are logged as warnings but retained in output
        Results are stored in self.labs_df attribute
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
        if not isinstance(summary_lookback, int) or summary_lookback < 0:
            raise ValueError("summary_lookback must be a non-negative integer")
        
        # Add user-provided mappings if they exist
        if additional_loinc_mappings is not None:
            if not isinstance(additional_loinc_mappings, dict):
                raise ValueError("Additional LOINC mappings must be provided as a dictionary")
            if not all(isinstance(v, list) for v in additional_loinc_mappings.values()):
                raise ValueError("LOINC codes must be provided as lists of strings")
                
            # Update the default mappings with additional ones
            self.LOINC_MAPPINGS.update(additional_loinc_mappings)

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
            all_loinc_codes = sum(self.LOINC_MAPPINGS.values(), [])

            # Filter for LOINC codes 
            df = df[df['LOINC'].isin(all_loinc_codes)].copy()

            # Map LOINC codes to lab names
            for lab_name, loinc_codes in self.LOINC_MAPPINGS.items():
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
            
            # Merge dataframes - start with index_date_df to ensure all PatientIDs are included
            final_df = index_date_df[['PatientID']].copy()
            final_df = pd.merge(final_df, lab_df, on = 'PatientID', how = 'left')
            final_df = pd.merge(final_df, max_df, on = 'PatientID', how = 'left')
            final_df = pd.merge(final_df, min_df, on = 'PatientID', how = 'left')
            final_df = pd.merge(final_df, std_df, on = 'PatientID', how = 'left')
            final_df = pd.merge(final_df, slope_df, on = 'PatientID', how = 'left')

            # Check for duplicate PatientIDs
            if len(final_df) > final_df['PatientID'].nunique():
                logging.error(f"Duplicate PatientIDs found")
                return None

            logging.info(f"Successfully processed Lab.csv file with final shape: {final_df.shape} and unique PatientIDs: {(final_df['PatientID'].nunique())}")
            self.labs_df = final_df
            return final_df

        except Exception as e:
            logging.error(f"Error processing Lab.csv file: {e}")
            return None
        
    def process_medications(self,
                            file_path: str,
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
            - anticoagulated : Int64
                binary indicator (0/1) for therapeutic anticoagulation (heparin IV with specific units, enoxaparin >40mg, dalteparin >5000u, fondaparinux >2.5mg, or any DOAC/warfarin) 
            - opioid : Int64
                binary indicator (0/1) for oral, transdermal, or sublingual opioids
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

            # Create dictionary of medication categories and their respective IDs
            med_categories = {
                'anticoagulated': anticoagulated_IDs,
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
                logging.error(f"Duplicate PatientIDs found")
                return None

            logging.info(f"Successfully processed MedicationAdministration.csv file with final shape: {final_df.shape} and unique PatientIDs: {(final_df['PatientID'].nunique())}")
            self.medications_df = final_df
            return final_df

        except Exception as e:
            logging.error(f"Error processing MedicationAdministration.csv file: {e}")
            return None
        
    def process_diagnosis(self, 
                          file_path: str,
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
            - PatientID : object unique patient identifier
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
        All PatientIDs from index_date_df are included in the output and values will be set to 0 for patients with misisng Elixhauser comorbidities or metastasis sites, but NaN for van_walraven_score
        Duplicate PatientIDs are logged as warnings but retained in output
        Results are stored in self.diagnoses_df attribute
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
                    lambda code: next((comorb for pattern, comorb in self.ICD_9_EXLIXHAUSER_MAPPING.items() 
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

            # Elixhauser comorbidities based on ICD-10 codes
            df10_elix = (
                df_filtered
                .query('DiagnosisCodeSystem == "ICD-10-CM"')
                .assign(diagnosis_code = lambda x: x['DiagnosisCode'].replace(r'\.', '', regex=True)) # Remove decimal points from ICD-10 codes to make mapping easier 
                .drop_duplicates(subset = ['PatientID', 'diagnosis_code'], keep = 'first')
                .assign(comorbidity=lambda x: x['diagnosis_code'].map(
                    lambda code: next((comorb for pattern, comorb in self.ICD_10_ELIXHAUSER_MAPPING.items() 
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

            all_columns_elix = ['PatientID'] + list(set(self.ICD_9_EXLIXHAUSER_MAPPING.values()) - {'Metastatic', 'Tumor'})
            
            # Reindex both dataframes to have all columns, filling missing ones with 0
            df9_elix_aligned = df9_elix.reindex(columns = all_columns_elix, fill_value = 0)
            df10_elix_aligned = df10_elix.reindex(columns = all_columns_elix, fill_value = 0)

            # Combine Elixhauser comorbidity dataframes for ICD-9 and ICD-10
            df_elix_combined = pd.concat([df9_elix_aligned, df10_elix_aligned]).groupby('PatientID').max().reset_index()

            # Calculate van Walraven score
            van_walraven_score = df_elix_combined.drop('PatientID', axis=1).mul(self.VAN_WALRAVEN_WEIGHTS).sum(axis=1)
            df_elix_combined['van_walraven_score'] = van_walraven_score

            # Metastatic sites based on ICD-9 codes 
            df9_mets = (
                df_filtered
                .query('DiagnosisCodeSystem == "ICD-9-CM"')
                .assign(diagnosis_code = lambda x: x['DiagnosisCode'].replace(r'\.', '', regex=True)) # Remove decimal points from ICD-9 codes to make mapping easier 
                .drop_duplicates(subset = ['PatientID', 'diagnosis_code'], keep = 'first')
                .assign(met_site=lambda x: x['diagnosis_code'].map(
                    lambda code: next((site for pattern, site in self.ICD_9_METS_MAPPING.items()
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

            # Metastatic sites based on ICD-10 codes 
            df10_mets = (
                df_filtered
                .query('DiagnosisCodeSystem == "ICD-10-CM"')
                .assign(diagnosis_code = lambda x: x['DiagnosisCode'].replace(r'\.', '', regex=True)) # Remove decimal points from ICD-9 codes to make mapping easier 
                .drop_duplicates(subset = ['PatientID', 'diagnosis_code'], keep = 'first')
                .assign(met_site=lambda x: x['diagnosis_code'].map(
                    lambda code: next((site for pattern, site in self.ICD_10_METS_MAPPING.items()
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

            all_columns_mets = ['PatientID'] + list(set(self.ICD_9_METS_MAPPING.values())) 
            
            # Reindex both dataframes to have all columns, filling missing ones with 0
            df9_mets_aligned = df9_mets.reindex(columns = all_columns_mets, fill_value = 0)
            df10_mets_aligned = df10_mets.reindex(columns = all_columns_mets, fill_value = 0)

            df_mets_combined = pd.concat([df9_mets_aligned, df10_mets_aligned]).groupby('PatientID').max().reset_index()

            # Start with index_date_df to ensure all PatientIDs are included
            final_df = index_date_df[['PatientID']].copy()
            final_df = pd.merge(final_df, df_elix_combined, on = 'PatientID', how = 'left')
            final_df = pd.merge(final_df, df_mets_combined, on = 'PatientID', how = 'left')

            binary_columns = [col for col in final_df.columns 
                    if col not in ['PatientID', 'van_walraven_score']]
            final_df[binary_columns] = final_df[binary_columns].fillna(0).astype('Int64')

            # Check for duplicate PatientIDs
            if len(final_df) > final_df['PatientID'].nunique():
                logging.error(f"Duplicate PatientIDs found")
                return None

            logging.info(f"Successfully processed Diagnosis.csv file with final shape: {final_df.shape} and unique PatientIDs: {(final_df['PatientID'].nunique())}")
            self.diagnosis_df = final_df
            return final_df

        except Exception as e:
            logging.error(f"Error processing Diagnosis.csv file: {e}")
            return None
        
    def process_insurance(self, 
                          file_path: str,
                          index_date_df: pd.DataFrame,
                          index_date_column: str,
                          days_before: Optional[int] = None,
                          days_after: int = 0) -> pd.DataFrame:
        """
        Processes insurance data to identify insurance coverage relative to a specified index date.
        Insurance types are grouped into four categories: Medicare, Medicaid, Commercial, and Other. 
        
        Parameters
        ----------
        file_path : str
            Path to Insurance.csv file
        index_date_df : pd.DataFrame
            DataFrame containing PatientID and index dates. Only insurances for PatientIDs present in this DataFrame will be processed
        index_date_column : str
            Column name in index_date_df containing the index date
        days_before : int | None, optional
            Number of days before the index date to include for window period. Must be >= 0 or None. If None, includes all prior results. Default: None
        days_after : int, optional
            Number of days after the index date to include for window period. Must be >= 0. Default: 0
        
        Returns
        -------
        pd.DataFrame
            - PatientID : object
                unique patient identifier
            - medicare : Int64
                binary indicator (0/1) for Medicare coverage
            - medicaid : Int64
                binary indicator (0/1) for Medicaid coverage
            - commercial : Int64
                binary indicator (0/1) for commercial insuarnce coverage
            - other : Int64
                binaroy indicator (0/1) for other insurance types (eg., other payer, other government program, patient assistance program, self pay, and workers compensation)

        Notes
        -----
        Insurance is considered active if:
        1. StartDate falls before or during the specified time window AND
        2. Either:
            - EndDate is missing (considered still active) OR
            - EndDate falls on or after the start of the time window 
        EndDate is missing for most patients
        Missing StartDate values are conservatively imputed with EndDate values
        All PatientIDs from index_date_df are included in the output
        Duplicate PatientIDs are logged as warnings but retained in output
        Results are stored in self.insurance_df attribute
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
            logging.info(f"Successfully read Insurance.csv file with shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")

            df['StartDate'] = pd.to_datetime(df['StartDate'])
            df['EndDate'] = pd.to_datetime(df['EndDate'])

            # Impute missing StartDate with EndDate
            df['StartDate'] = np.where(df['StartDate'].isna(), df['EndDate'], df['StartDate'])

            index_date_df[index_date_column] = pd.to_datetime(index_date_df[index_date_column])

            # Select PatientIDs that are included in the index_date_df the merge on 'left'
            df = df[df.PatientID.isin(index_date_df.PatientID)]
            df = pd.merge(
                df,
                index_date_df[['PatientID', index_date_column]],
                on = 'PatientID',
                how = 'left'
                )
            logging.info(f"Successfully merged Insurance.csv df with index_date_df resulting in shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")

            # Calculate days relative to index date for start 
            df['days_to_start'] = (df['StartDate'] - df[index_date_column]).dt.days

            # Define window boundaries
            window_start = -days_before if days_before is not None else float('-inf')
            window_end = days_after

            # Insurance is active if it:
            # 1. Starts before or during the window AND
            # 2. Either has no end date OR ends after window starts
            df_filtered = df[
                (df['days_to_start'] <= window_end) &  # Starts before window ends
                (
                    df['EndDate'].isna() |  # Either has no end date (presumed to be still active)
                    ((df['EndDate'] - df[index_date_column]).dt.days >= window_start)  # Or ends after window starts
                )
            ].copy()

            df_filtered['PayerCategory'] = df_filtered['PayerCategory'].replace(self.INSURANCE_MAPPING)

            final_df = (
                df_filtered
                .drop_duplicates(subset = ['PatientID', 'PayerCategory'], keep = 'first')
                .assign(value=1)
                .pivot(index = 'PatientID', columns = 'PayerCategory', values = 'value')
                .fillna(0) 
                .astype(int)  
                .rename_axis(columns = None)
                .reset_index()
            )

            # Merger index_date_df to ensure all PatientIDs are included
            final_df = pd.merge(index_date_df[['PatientID']], final_df, on = 'PatientID', how = 'left')
            
            insurance_columns = list(set(self.INSURANCE_MAPPING.values()))
            for col in insurance_columns:
                final_df[col] = final_df[col].fillna(0).astype('Int64')

            # Check for duplicate PatientIDs
            if len(final_df) > final_df['PatientID'].nunique():
                logging.error(f"Duplicate PatientIDs found")
                return None

            logging.info(f"Successfully processed Insurance.csv file with final shape: {final_df.shape} and unique PatientIDs: {(final_df['PatientID'].nunique())}")
            self.insurance_df = final_df
            return final_df

        except Exception as e:
            logging.error(f"Error processing Insurance.csv file: {e}")
            return None

            

        