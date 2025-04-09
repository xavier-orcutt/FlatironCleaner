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
        'Stage IV': 'IV',
        'Stage IVA': 'IV',
        'Stage IVB': 'IV',
        'Stage III': 'III',
        'Stage IIIA': 'III',
        'Stage IIIB': 'III',
        'Stage II': 'II',
        'Stage I': 'I',
        'Stage 0is': '0',
        'Stage 0a': '0',
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
        'Ta': 'nmibc',
        'Tis': 'nmibc',
        'T0': 'other',
        'TX': 'other',
        'Unknown/not documented': 'other'
    }

    N_STAGE_MAPPING = {
        'N3': 'N3',
        'N2': 'N2',
        'N1': 'N1',
        'N0': 'N0',
        'NX': 'unknown',
        'Unknown/not documented': 'unknown'
    }

    M_STAGE_MAPPING = {
        'M1': 'M1',
        'M1a': 'M1',
        'M1b': 'M1',
        'M0': 'M0',
        'MX': 'unknown',
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
        'Unknown/not documented': 'unknown'
    }

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
        'platelet': ['26515-7', '777-3', '778-1'],
        'creatinine': ['2160-0', '38483-4'],
        'bun': ['3094-0'],
        'sodium': ['2947-0', '2951-2'],
        'bicarbonate': ['1963-8', '1959-6', '14627-4', '1960-4', '2028-9'],
        'chloride': ['2075-0'],
        'potassium': ['6298-4', '2823-3'],
        'albumin': ['1751-7', '35706-1', '13980-8'],
        'calcium': ['17861-6', '49765-1'],
        'total_bilirubin': ['42719-5', '1975-2'],
        'ast': ['1920-8', '30239-8'],
        'alt': ['1742-6', '1743-4', '1744-2'],
        'alp': ['6768-6']
    }

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
        r'^042|^043|^044': 'aids_hiv',
        
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
        r'^2652|^2911|^2912|^2913|^2915|^2916|^2917|^2918|^2919|^3030|^3039|^3050|^3575|^4255|^5353|^5710|^5711|^5712|^5713|^980|^V113': 'alcohol_abuse',
        
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
        r'^B20|^B21|^B22|^B24': 'aids_hiv',
        
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
        
        # Fluid and electrolyte disorders
        r'^E222|^E86|^E87': 'fluid',
        
        # Blood loss anemia
        r'^D500': 'blood_loss_anemia',
        
        # Deficiency anemia
        r'^D508|^D509|^D51|^D52|^D53': 'deficiency_anemia',
        
        # Alcohol abuse
        r'^F10|^E52|^G621|^I426|^K292|^K700|^K703|^K709|^T51|^Z502|^Z714|^Z721': 'alcohol_abuse',
        
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
        'aids_hiv': 0,
        'lymphoma': 9,
        'rheumatic': 0,
        'coagulopathy': 3,
        'obesity': -4,
        'weight_loss': 6,
        'fluid': 5,
        'blood_loss_anemia': -2,
        'deficiency_anemia': -2,
        'alcohol_abuse': 0,
        'drug_abuse': -7,
        'psychoses': 0,
        'depression': -3
    }

    ICD_9_METS_MAPPING = {
        # Lymph nodes
        r'^196': 'lymph_met',
        
        # Respiratory
        r'^1970|^1971|^1972|^1973': 'thoracic_met',

        # Liver
        r'^1977': 'liver_met',
        
        # Bone
        r'^1985': 'bone_met',
        
        # Brain/CNS
        r'^1983|^1984': 'brain_met',
        
        # Adrenal
        r'^1987': 'adrenal_met',
        
        # Peritoneum/retroperitoneum
        r'^1976': 'peritoneum_met',
        
        # Other GI tract
        r'^1974|^1975|^1978': 'gi_met',
        
        # Other sites
        r'^1980|^1981|^1982|^1986|^1988|^199': 'other_met'
    }

    ICD_10_METS_MAPPING = {
        # Lymph nodes
        r'^C77': 'lymph_met',

        # Thoracic
        r'^C780|^C781|^C782|^C783': 'thoracic_met',

        # Liver
        r'^C787': 'liver_met',

        # Bone
        r'^C795': 'bone_met',

        # Brain/CNS
        r'^C793|^C794': 'brain_met',

        # Adrenal
        r'^C797': 'adrenal_met',

        # Peritoneum/retroperitoneum
        r'^C786': 'peritoneum_met',

        # Other GI tract
        r'^C784|^C785|^C788': 'gi_met',

        # Other sites
        r'^C790|^C791|^C792|^C796|^C798|^C799|^C80': 'other_met'
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

    def process_enhanced(self,
                         file_path: str,
                         patient_ids: list = None,
                         drop_stages: bool = True, 
                         drop_surgery_type: bool = True,
                         drop_dates: bool = True) -> Optional[pd.DataFrame]: 
        """
        Processes Enhanced_AdvUrothelial.csv to standardize categories, consolidate 
        staging information, and calculate time-based metrics between key clinical events.

        Parameters
        ----------
        file_path : str
            Path to Enhanced_AdvUrothelial.csv file
        patient_ids : list, optional
            List of PatientIDs to process. If None, processes all patients
        drop_stages : bool, default=True
            If True, drops original staging columns (GroupStage, TStage, NStage, and MStage) after creating modified versions
        drop_surgery_type : bool, default=True
            If True, drops original surgery type after creating modified version
        drop_dates : bool, default=True
            If True, drops date columns (DiagnosisDate, AdvancedDiagnosisDate, and SurgeryDate) after calculating durations

        Returns
        -------
        pd.DataFrame or None
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
            - DiseaseGrade : category
                tumor grade (high, low, and unknown) at time of first diagnosis
            - GroupStage_mod : category
                consolidated overall staging (0-IV and unknown) at time of first diagnosis
            - TStage_mod : category
                consolidated tumor staging (T1-T4, non-muscle invasive bladder cancer [nmibc], other) at time of first diagnosis
            - NStage_mod : category
                consolidated lymph node staging (N0, N1, N2, N3, and unknown) at time of first diagnosis
            - MStage_mod : category
                consolidated metastasis staging (M0, M1, and unknown) at time of first diagnosis
            - days_diagnosis_to_adv : float
                days from first diagnosis to advanced disease 
            - adv_diagnosis_year : category
                year of advanced diagnosis 
            - days_diagnosis_to_surgery : float
                days from first diagnosis to surgery 
            
            Original staging and date columns retained if respective drop_* = False

        Notes
        -----
        Notable T-Stage consolidation decisions:
            - Ta and Tis : 'nmibc' (Non-muscle invasive disease)
            - T0, TX, Unknown/not documented : 'other' 

        If days_diagnosis_to_surgery is < 0, the value is set to 0; otherwise, the value is left as is. 
        Cases where days_diagnosis_to_surgery is < 0 occur when the date of surgery is set to the first of the year 
        and likely reflects uknown exact date of surgery. By setting these cases to 0, we'll presume that the 
        surgery date occurred on the diagnosis date. 

        Output handling:
        - Duplicate PatientIDs are logged as warnings if found but reatained in output
        - Processed DataFrame is stored in self.enhanced_df
        """
        # Input validation
        if patient_ids is not None:
            if not isinstance(patient_ids, list):
                raise TypeError("patient_ids must be a list or None")
            
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
            df['NStage_mod'] = df['NStage'].map(self.N_STAGE_MAPPING).astype('category')
            df['MStage_mod'] = df['MStage'].map(self.M_STAGE_MAPPING).astype('category')

            # Drop original stage variables if specified
            if drop_stages:
                df = df.drop(columns=['GroupStage', 'TStage', 'NStage', 'MStage'])

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
            df['Surgery'] = df['Surgery'].astype('Int64')

            # Generate new variables 
            df['days_diagnosis_to_adv'] = (df['AdvancedDiagnosisDate'] - df['DiagnosisDate']).dt.days
            df['adv_diagnosis_year'] = pd.Categorical(df['AdvancedDiagnosisDate'].dt.year)
            df['days_diagnosis_to_surgery'] = (df['SurgeryDate'] - df['DiagnosisDate']).dt.days
            
            # For those with value <0, set to 0; otherwise, leave as is
            df['days_diagnosis_to_surgery'] = np.where(df['days_diagnosis_to_surgery'] < 0, 0, df['days_diagnosis_to_surgery'])
    
            if drop_dates:
                df = df.drop(columns = ['AdvancedDiagnosisDate', 'DiagnosisDate', 'SurgeryDate'])

            # Check for duplicate PatientIDs
            if len(df) > df['PatientID'].nunique():
                duplicate_ids = df[df.duplicated(subset = ['PatientID'], keep = False)]['PatientID'].unique()
                logging.warning(f"Duplicate PatientIDs found: {duplicate_ids}")

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
                             drop_state: bool = True) -> Optional[pd.DataFrame]:
        """
        Processes Demographics.csv by standardizing categorical variables, mapping states to census regions, and calculating age at index date.

        Parameters
        ----------
        file_path : str
            Path to Demographics.csv file
        index_date_df : pd.DataFrame, optional
            DataFrame containing unique PatientIDs and their corresponding index dates. Only demographics for PatientIDs present in this DataFrame will be processed
        index_date_column : str, optional
            Column name in index_date_df containing index date
        drop_state : bool, default = True
            If True, drops State column after mapping to regions

        Returns
        -------
        pd.DataFrame or None
            - PatientID : object
                unique patient identifier
            - Gender : category
                gender
            - Race_mod : category
                race (White, Black or African America, Asian, Other Race)
            - Ethnicity_mod : category
                ethnicity (Hispanic or Latino, Not Hispanic or Latino)
            - age : Int64
                age at index date (index year - birth year)
            - region : category
                Maps all 50 states, plus DC and Puerto Rico (PR), to a US Census Bureau region
            - State : category
                US state (if drop_state=False)
            
        Notes
        -----
        Data cleaning and processing: 
        - Imputation for Race and Ethnicity:
            - If Race='Hispanic or Latino', Race value is replaced with NaN
            - If Race='Hispanic or Latino' and Ethnicity is missing, Ethnicity is set to 'Hispanic or Latino'
            - Otherwise, missing Race and Ethnicity values remain unchanged
        - Ages calculated as <18 or >120 are logged as warning if found, but not removed
        - Missing States and Puerto Rico are imputed as unknown during the mapping to regions
        
        Output handling: 
        - Duplicate PatientIDs are logged as warnings if found but retained in output
        - Processed DataFrame is stored in self.demographics_df
        """
        # Input validation
        if not isinstance(index_date_df, pd.DataFrame):
            raise ValueError("index_date_df must be a pandas DataFrame")
        if 'PatientID' not in index_date_df.columns:
            raise ValueError("index_date_df must contain a 'PatientID' column")
        if not index_date_column or index_date_column not in index_date_df.columns:
            raise ValueError('index_date_column not found in index_date_df')
        if index_date_df['PatientID'].duplicated().any():
            raise ValueError("index_date_df contains duplicate PatientID values, which is not allowed")
        
        index_date_df = index_date_df.copy()
        # Rename all columns from index_date_df except PatientID to avoid conflicts with merging and processing 
        for col in index_date_df.columns:
            if col != 'PatientID':  # Keep PatientID unchanged for merging
                index_date_df.rename(columns={col: f'imported_{col}'}, inplace=True)

        # Update index_date_column name
        index_date_column = f'imported_{index_date_column}'
        
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
            # If Race == 'Hispanic or Latino' and Ethnicity is empty, fill 'Hispanic or Latino' for Ethnicity
            df['Ethnicity_mod'] = np.where((df['Race'] == 'Hispanic or Latino') & (df['Ethnicity'].isna()), 
                                            'Hispanic or Latino', 
                                            df['Ethnicity'])

            # If Race == 'Hispanic or Latino' replace with Nan
            df['Race_mod'] = np.where(df['Race'] == 'Hispanic or Latino', 
                                      np.nan, 
                                      df['Race'])

            df[['Race_mod', 'Ethnicity_mod']] = df[['Race_mod', 'Ethnicity_mod']].astype('category')
            df = df.drop(columns = ['Race', 'Ethnicity'])

            # Region processing
            # Group states into Census-Bureau regions  
            df['region'] = (df['State']
                            .map(self.STATE_REGIONS_MAPPING)
                            .fillna('unknown')
                            .astype('category'))

            # Drop State varibale if specified
            if drop_state:               
                df = df.drop(columns = ['State'])

            # Check for duplicate PatientIDs
            if len(df) > df['PatientID'].nunique():
                duplicate_ids = df[df.duplicated(subset = ['PatientID'], keep = False)]['PatientID'].unique()
                logging.warning(f"Duplicate PatientIDs found: {duplicate_ids}")
            
            logging.info(f"Successfully processed Demographics.csv file with final shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")
            self.demographics_df = df
            return df

        except Exception as e:
            logging.error(f"Error processing Demographics.csv file: {e}")
            return None
        

    def process_practice(self,
                         file_path: str,
                         patient_ids: list = None) -> Optional[pd.DataFrame]:
        """
        Processes Practice.csv to consolidate practice types per patient into a single categorical value indicating academic, community, or both settings.

        Parameters
        ----------
        file_path : str
            Path to Practice.csv file
        patient_ids : list, optional
            List of PatientIDs to process. If None, processes all patients

        Returns
        -------
        pd.DataFrame or None
            - PatientID : object
                unique patient identifier  
            - PracticeType_mod : category
                practice setting (ACADEMIC, COMMUNITY, or BOTH)
       
        Notes
        -----
        Output handling: 
        - PracticeID and PrimaryPhysicianID are removed
        - Duplicate PatientIDs are logged as warnings if found but retained in output
        - Processed DataFrame is stored in self.practice_df
        """
        # Input validation
        if patient_ids is not None:
            if not isinstance(patient_ids, list):
                raise TypeError("patient_ids must be a list or None")
            
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
            grouped_df = pd.DataFrame(grouped).reset_index()

            # Function to determine the modified practice type
            def get_practice_type(practice_types):
                if len(practice_types) == 0:
                    return 'UNKNOWN'
                if len(practice_types) > 1:
                    return 'BOTH'
                return practice_types[0]
            
            # Apply the function to the column containing sets
            grouped_df['PracticeType_mod'] = grouped_df['PracticeType'].apply(get_practice_type).astype('category')

            final_df = grouped_df[['PatientID', 'PracticeType_mod']]

            # Check for duplicate PatientIDs
            if len(final_df) > final_df['PatientID'].nunique():
                duplicate_ids = final_df[final_df.duplicated(subset = ['PatientID'], keep = False)]['PatientID'].unique()
                logging.warning(f"Duplicate PatientIDs found: {duplicate_ids}")
            
            logging.info(f"Successfully processed Practice.csv file with final shape: {final_df.shape} and unique PatientIDs: {(final_df['PatientID'].nunique())}")
            self.practice_df = final_df
            return final_df

        except Exception as e:
            logging.error(f"Error processing Practice.csv file: {e}")
            return None
        
    def process_mortality(self,
                          file_path: str,
                          index_date_df: pd.DataFrame,
                          index_date_column: str,
                          visit_path: str = None, 
                          telemedicine_path: str = None, 
                          biomarkers_path: str = None, 
                          oral_path: str = None,
                          progression_path: str = None,
                          drop_dates: bool = True) -> Optional[pd.DataFrame]:
        """
        Processes Enhanced_Mortality_V2.csv by cleaning data types, calculating time from index date to death/censor, and determining mortality events. 

        Parameters
        ----------
        file_path : str
            Path to Enhanced_Mortality_V2.csv file
        index_date_df : pd.DataFrame
            DataFrame containing unique PatientIDs and their corresponding index dates. Only mortality data for PatientIDs present in this DataFrame will be processed
        index_date_column : str
            Column name in index_date_df containing the index date
        visit_path : str
            Path to Visit.csv file, using VisitDate to determine last EHR activity date for censored patients
        telemedicine_path : str
            Path to Telemedicine.csv file, using VisitDate to determine last EHR activity date for censored patients
        biomarkers_path : str
            Path to Enhanced_AdvUrothelialBiomarkers.csv file, using SpecimenCollectedDate to determine last EHR activity date for censored patients
        oral_path : str
            Path to Enhanced_AdvUrothelial_Orals.csv file, using StartDate and EndDate to determine last EHR activity date for censored patients
        progression_path : str
            Path to Enhanced_AdvUrothelial_Progression.csv file, using ProgresionDate and LastClinicNoteDate to determine last EHR activity date for censored patients
        drop_dates : bool, default = True
            If True, drops date columns (index_date_column, DateOfDeath, last_ehr_date)   
        
        Returns
        -------
        pd.DataFrame or None
            - PatientID : object
                unique patient identifier
            - duration : float
                days from index date to death or censor 
            - event : Int64
                mortality status (1 = death, 0 = censored)

        If drop_dates=False, the DataFrame will also include:
            - {index_date_column} : datetime64
                The index date for each patient
            - DateOfDeath : datetime64
                Date of death (if available)
            - last_ehr_activity : datetime64
                Most recent EHR activity date (if available from supplementary files)
                
        Notes
        ------
        Death date handling:
        - Known death date: 'event' = 1, 'duration' = days from index to death
        - No death date: 'event' = 0, 'duration' = days from index to last EHR activity
        
        Death date imputation for incomplete dates:
        - Missing day: Imputed to 15th of the month
        - Missing month and day: Imputed to July 1st of the year
    
        Censoring logic:
        - Patients without death dates are censored at their last EHR activity
        - Last EHR activity is determined as the maximum date across all provided supplementary files (visit, telemedicine, biomarkers, oral, progression)
        - If no supplementary files are provided or a patient has no activity in supplementary files, duration may be null for censored patients
        
        Output handling: 
        - Duplicate PatientIDs are logged as warnings if found but retained in output
        - Processed DataFrame is stored in self.mortality_df
        """
        # Input validation
        if not isinstance(index_date_df, pd.DataFrame):
            raise ValueError("index_date_df must be a pandas DataFrame")
        if 'PatientID' not in index_date_df.columns:
            raise ValueError("index_date_df must contain a 'PatientID' column")
        if not index_date_column or index_date_column not in index_date_df.columns:
            raise ValueError('index_date_column not found in index_date_df')
        if index_date_df['PatientID'].duplicated().any():
            raise ValueError("index_date_df contains duplicate PatientID values, which is not allowed")
        
        index_date_df = index_date_df.copy()
        # Rename all columns from index_date_df except PatientID to avoid conflicts with merging and processing 
        for col in index_date_df.columns:
            if col != 'PatientID':  # Keep PatientID unchanged for merging
                index_date_df.rename(columns={col: f'imported_{col}'}, inplace=True)

        # Update index_date_column name
        index_date_column = f'imported_{index_date_column}'

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
            df = pd.merge(
                index_date_df[['PatientID', index_date_column]],
                df,
                on = 'PatientID',
                how = 'left'
            )
            logging.info(f"Successfully merged Enhanced_Mortality_V2.csv df with index_date_df resulting in shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")
                
            # Create event column
            df['event'] = df['DateOfDeath'].notna().astype('Int64')

            # Initialize final dataframe
            final_df = df.copy()

            # Create a list to store all last activity date dataframes
            patient_last_dates = []

            # Determine last EHR data
            if all(path is None for path in [visit_path, telemedicine_path, biomarkers_path, oral_path, progression_path]):
                logging.info("WARNING: At least one of visit_path, telemedicine_path, biomarkers_path, oral_path, or progression_path must be provided to calculate duration for those with a missing death date")
            else: 
                # Process visit and telemedicine data
                if visit_path is not None or telemedicine_path is not None:
                    visit_dates = []
                    try:
                        if visit_path is not None:
                            df_visit = pd.read_csv(visit_path)
                            df_visit['VisitDate'] = pd.to_datetime(df_visit['VisitDate'])
                            visit_dates.append(df_visit[['PatientID', 'VisitDate']])
                            
                        if telemedicine_path is not None:
                            df_tele = pd.read_csv(telemedicine_path)
                            df_tele['VisitDate'] = pd.to_datetime(df_tele['VisitDate'])
                            visit_dates.append(df_tele[['PatientID', 'VisitDate']])
                        
                        if visit_dates:
                            df_visit_combined = pd.concat(visit_dates)
                            df_visit_max = (
                                df_visit_combined
                                .query("PatientID in @index_date_df.PatientID")
                                .groupby('PatientID')['VisitDate']
                                .max()
                                .to_frame(name='last_visit_date')
                                .reset_index()
                            )
                            patient_last_dates.append(df_visit_max)
                    except Exception as e:
                        logging.error(f"Error processing Visit.csv or Telemedicine.csv: {e}")
                                            
                # Process biomarkers data
                if biomarkers_path is not None:
                    try: 
                        df_biomarkers = pd.read_csv(biomarkers_path)
                        df_biomarkers['SpecimenCollectedDate'] = pd.to_datetime(df_biomarkers['SpecimenCollectedDate'])

                        df_biomarkers_max = (
                            df_biomarkers
                            .query("PatientID in @index_date_df.PatientID")
                            .groupby('PatientID')['SpecimenCollectedDate']
                            .max()
                            .to_frame(name='last_biomarker_date')
                            .reset_index()
                        )
                        patient_last_dates.append(df_biomarkers_max)
                    except Exception as e:
                        logging.error(f"Error reading Enhanced_AdvUrothelialBiomarkers.csv file: {e}")

                # Process oral medication data
                if oral_path is not None:
                    try:
                        df_oral = pd.read_csv(oral_path)
                        df_oral['StartDate'] = pd.to_datetime(df_oral['StartDate'])
                        df_oral['EndDate'] = pd.to_datetime(df_oral['EndDate'])

                        df_oral_max = (
                            df_oral
                            .query("PatientID in @index_date_df.PatientID")
                            .assign(max_date=lambda x: x[['StartDate', 'EndDate']].max(axis=1))
                            .groupby('PatientID')['max_date']
                            .max()
                            .to_frame(name='last_oral_date')
                            .reset_index()
                        )
                        patient_last_dates.append(df_oral_max)
                    except Exception as e:
                        logging.error(f"Error reading Enhanced_AdvUrothelial_Orals.csv file: {e}")

                # Process progression data
                if progression_path is not None:
                    try: 
                        df_progression = pd.read_csv(progression_path)
                        df_progression['ProgressionDate'] = pd.to_datetime(df_progression['ProgressionDate'])
                        df_progression['LastClinicNoteDate'] = pd.to_datetime(df_progression['LastClinicNoteDate'])

                        df_progression_max = (
                            df_progression
                            .query("PatientID in @index_date_df.PatientID")
                            .assign(max_date=lambda x: x[['ProgressionDate', 'LastClinicNoteDate']].max(axis=1))
                            .groupby('PatientID')['max_date']
                            .max()
                            .to_frame(name='last_progression_date')
                            .reset_index()
                        )
                        patient_last_dates.append(df_progression_max)
                    except Exception as e:
                        logging.error(f"Error reading Enhanced_AdvUrothelial_Progression.csv file: {e}")

                # Combine all last activity dates
                if patient_last_dates:
                    # Start with the first dataframe
                    combined_dates = patient_last_dates[0]
                    
                    # Merge with any additional dataframes
                    for date_df in patient_last_dates[1:]:
                        combined_dates = pd.merge(combined_dates, date_df, on = 'PatientID', how = 'outer')
                    
                    # Calculate the last activity date across all columns
                    date_columns = [col for col in combined_dates.columns if col != 'PatientID']
                    if date_columns:
                        logging.info(f"The following columns {date_columns} are used to calculate the last EHR date")
                        combined_dates['last_ehr_activity'] = combined_dates[date_columns].max(axis=1)
                        single_date = combined_dates[['PatientID', 'last_ehr_activity']]
                        
                        # Merge with the main dataframe
                        final_df = pd.merge(final_df, single_date, on='PatientID', how='left')
     
            # Calculate duration
            if 'last_ehr_activity' in final_df.columns:
                final_df['duration'] = np.where(
                    final_df['event'] == 0, 
                    (final_df['last_ehr_activity'] - final_df[index_date_column]).dt.days, 
                    (final_df['DateOfDeath'] - final_df[index_date_column]).dt.days
                )
                
                # Drop date variables if specified
                if drop_dates:               
                    final_df = final_df.drop(columns=[index_date_column, 'DateOfDeath', 'last_ehr_activity'])
                       
            else: 
                final_df['duration'] = (final_df['DateOfDeath'] - final_df[index_date_column]).dt.days
            
                # Drop date variables if specified
                if drop_dates:               
                    final_df = final_df.drop(columns=[index_date_column, 'DateOfDeath'])
                
            # Check for duplicate PatientIDs
            if len(final_df) > final_df['PatientID'].nunique():
                duplicate_ids = final_df[final_df.duplicated(subset=['PatientID'], keep=False)]['PatientID'].unique()
                logging.warning(f"Duplicate PatientIDs found: {duplicate_ids}")

            logging.info(f"Successfully processed Enhanced_Mortality_V2.csv file with final shape: {final_df.shape} and unique PatientIDs: {final_df['PatientID'].nunique()}. There are {final_df['duration'].isna().sum()} out of {final_df['PatientID'].nunique()} patients with missing duration values")
            self.mortality_df = final_df
            return final_df

        except Exception as e:
            logging.error(f"Error processing Enhanced_Mortality_V2.csv file: {e}")
            return None
        
    def process_biomarkers(self,
                           file_path: str,
                           index_date_df: pd.DataFrame,
                           index_date_column: str, 
                           days_before: Optional[int] = None,
                           days_after: int = 0) -> Optional[pd.DataFrame]:
        """
        Processes Enhanced_AdvUrothelialBiomarkers.csv by determining FGFR and PDL1 status for each patient within a specified time window relative to an index date

        Parameters
        ----------
        file_path : str
            Path to Enhanced_AdvUrothelialBiomarkers.csv file
        index_date_df : pd.DataFrame
            DataFrame containing unique PatientIDs and their corresponding index dates. Only biomarker data for PatientIDs present in this DataFrame will be processed
        index_date_column : str
            Column name in index_date_df containing the index date
        days_before : int | None, optional
            Number of days before the index date to include. Must be >= 0 or None. If None, includes all prior results. Default: None
        days_after : int, optional
            Number of days after the index date to include. Must be >= 0. Default: 0
        
        Returns
        -------
        pd.DataFrame or None
            - PatientID : object
                unique patient identifier
            - FGFR_status : category
                positive if ever-positive, negative if only-negative, otherwise unknown
            - PDL1_status : category
                positive if ever-positive, negative if only-negative, otherwise unknown
            - PDL1_percent_staining : category, ordered 
                returns a patient's maximum percent staining for PDL1 (NaN if no positive PDL1 results)

        Notes
        ------
        Biomarker cleaning and processing: 
        - For each biomarker, status is classified as:
            - 'positive' if any test result is positive (ever-positive)
            - 'negative' if any test is negative without positives (only-negative) 
            - 'unknown' if all results are indeterminate

        - Missing biomarker data handling:
            - All PatientIDs from index_date_df are included in the output
            - Patients without any biomarker tests will have NaN values for all biomarker columns
            - Missing ResultDate is imputed with SpecimenReceivedDate

        Output handling: 
        - Duplicate PatientIDs are logged as warnings if found but retained in output
        - Processed DataFrame is stored in self.biomarkers_df
        """
        # Input validation
        if not isinstance(index_date_df, pd.DataFrame):
            raise ValueError("index_date_df must be a pandas DataFrame")
        if 'PatientID' not in index_date_df.columns:
            raise ValueError("index_date_df must contain a 'PatientID' column")
        if not index_date_column or index_date_column not in index_date_df.columns:
            raise ValueError('index_date_column not found in index_date_df')
        if index_date_df['PatientID'].duplicated().any():
            raise ValueError("index_date_df contains duplicate PatientID values, which is not allowed")
        
        if days_before is not None:
            if not isinstance(days_before, int) or days_before < 0:
                raise ValueError("days_before must be a non-negative integer or None")
        if not isinstance(days_after, int) or days_after < 0:
            raise ValueError("days_after must be a non-negative integer")
        
        index_date_df = index_date_df.copy()
        # Rename all columns from index_date_df except PatientID to avoid conflicts with merging and processing 
        for col in index_date_df.columns:
            if col != 'PatientID':  # Keep PatientID unchanged for merging
                index_date_df.rename(columns={col: f'imported_{col}'}, inplace=True)

        # Update index_date_column name
        index_date_column = f'imported_{index_date_column}'

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
            FGFR_df = (
                df_filtered
                .query('BiomarkerName == "FGFR"')
                .groupby('PatientID')['BiomarkerStatus']
                .agg(lambda x: 'positive' if any ('Positive' in val for val in x)
                    else ('negative' if any('Negative' in val for val in x)
                        else 'unknown'))
                .reset_index()
                .rename(columns={'BiomarkerStatus': 'FGFR_status'})
            )
            
            # Process PDL1 status
            PDL1_df = (
                df_filtered
                .query('BiomarkerName == "PDL1"')
                .groupby('PatientID')['BiomarkerStatus']
                .agg(lambda x: 'positive' if any ('PD-L1 positive' in val for val in x)
                    else ('negative' if any('PD-L1 negative/not detected' in val for val in x)
                        else 'unknown'))
                .reset_index()
                .rename(columns={'BiomarkerStatus': 'PDL1_status'})
            )

            # Process PDL1 staining 
            PDL1_staining_df = (
                df_filtered
                .query('BiomarkerName == "PDL1"')
                .query('BiomarkerStatus == "PD-L1 positive"')
                .groupby('PatientID')['PercentStaining']
                .apply(lambda x: x.map(self.PDL1_PERCENT_STAINING_MAPPING))
                .groupby('PatientID')
                .agg('max')
                .to_frame(name = 'PDL1_ordinal_value')
                .reset_index()
            )
            
            # Create reverse mapping to convert back to percentage strings
            reverse_pdl1_dict = {v: k for k, v in self.PDL1_PERCENT_STAINING_MAPPING.items()}
            PDL1_staining_df['PDL1_percent_staining'] = PDL1_staining_df['PDL1_ordinal_value'].map(reverse_pdl1_dict)
            PDL1_staining_df = PDL1_staining_df.drop(columns = ['PDL1_ordinal_value'])

            # Merge dataframes -- start with index_date_df to ensure all PatientIDs are included
            final_df = index_date_df[['PatientID']].copy()
            final_df = pd.merge(final_df, PDL1_df, on = 'PatientID', how = 'left')
            final_df = pd.merge(final_df, PDL1_staining_df, on = 'PatientID', how = 'left')
            final_df = pd.merge(final_df, FGFR_df, on = 'PatientID', how = 'left')

            final_df['PDL1_status'] = final_df['PDL1_status'].astype('category')
            final_df['FGFR_status'] = final_df['FGFR_status'].astype('category')

            staining_dtype = pd.CategoricalDtype(
                categories = ['0%', '< 1%', '1%', '2% - 4%', '5% - 9%', '10% - 19%',
                              '20% - 29%', '30% - 39%', '40% - 49%', '50% - 59%',
                              '60% - 69%', '70% - 79%', '80% - 89%', '90% - 99%', '100%'],
                              ordered = True
            )
            
            final_df['PDL1_percent_staining'] = final_df['PDL1_percent_staining'].astype(staining_dtype)

            # Check for duplicate PatientIDs
            if len(final_df) > final_df['PatientID'].nunique():
                duplicate_ids = final_df[final_df.duplicated(subset = ['PatientID'], keep = False)]['PatientID'].unique()
                logging.warning(f"Duplicate PatientIDs found: {duplicate_ids}")

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
                     days_before_further: int = 180) -> Optional[pd.DataFrame]:
        """
        Processes ECOG.csv to determine patient ECOG scores and progression patterns relative 
        to a reference index date. Uses two different time windows for distinct clinical purposes:
        
        1. A smaller window near the index date to find the most clinically relevant ECOG score
           that represents the patient's status at that time point
        2. A larger lookback window to detect clinically significant ECOG progression,
           specifically looking for patients whose condition worsened from ECOG 0-1 to ≥2

        Parameters
        ----------
        file_path : str
            Path to ECOG.csv file
        index_date_df : pd.DataFrame
            DataFrame containing unique PatientIDs and their corresponding index dates. Only ECOGs for PatientIDs present in this DataFrame will be processed
        index_date_column : str
            Column name in index_date_df containing the index date
        days_before : int, optional
            Number of days before the index date to include. Must be >= 0. Default: 90
        days_after : int, optional
            Number of days after the index date to include. Must be >= 0. Default: 0
        days_before_further : int, optional
            Number of days before index date to look for ECOG progression (0-1 to ≥2). Must be >= 0. Consider
            selecting a larger integer than days_before to capture meaningful clinical deterioration over time.
            Default: 180
            
        Returns
        -------
        pd.DataFrame or None
            - PatientID : object
                unique patient identifier
            - ecog_index : category, ordered 
                ECOG score (0-4) closest to index date
            - ecog_newly_gte2 : Int64
                binary indicator (0/1) for ECOG increased from 0-1 to ≥2 in larger lookback window 

        Notes
        ------
        Data cleaning and processing: 
        - The function selects the most clinically relevant ECOG score using the following priority rules:
            1. ECOG closest to index date is selected by minimum absolute day difference
            2. For equidistant measurements, higher ECOG score is selected
        
        Output handling: 
        - All PatientIDs from index_date_df are included in the output and values will be NaN for patients without ECOG values
        - Duplicate PatientIDs are logged as warnings if found but retained in output
        - Processed DataFrame is stored in self.ecog_df
        """
        # Input validation
        if not isinstance(index_date_df, pd.DataFrame):
            raise ValueError("index_date_df must be a pandas DataFrame")
        if 'PatientID' not in index_date_df.columns:
            raise ValueError("index_date_df must contain a 'PatientID' column")
        if not index_date_column or index_date_column not in index_date_df.columns:
            raise ValueError('index_date_column not found in index_date_df')
        if index_date_df['PatientID'].duplicated().any():
            raise ValueError("index_date_df contains duplicate PatientID values, which is not allowed")
        
        if not isinstance(days_before, int) or days_before < 0:
            raise ValueError("days_before must be a non-negative integer")
        if not isinstance(days_after, int) or days_after < 0:
            raise ValueError("days_after must be a non-negative integer")
        
        index_date_df = index_date_df.copy()
        # Rename all columns from index_date_df except PatientID to avoid conflicts with merging and processing 
        for col in index_date_df.columns:
            if col != 'PatientID':  # Keep PatientID unchanged for merging
                index_date_df.rename(columns={col: f'imported_{col}'}, inplace=True)

        # Update index_date_column name
        index_date_column = f'imported_{index_date_column}'

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
                    ascending=[True, True, False]) # Last False means highest ECOG is selected in ties
                .groupby('PatientID')
                .first()
                .reset_index()
                [['PatientID', 'EcogValue']]
                .rename(columns = {'EcogValue': 'ecog_index'})
                .assign(
                    ecog_index = lambda x: x['ecog_index'].astype(pd.CategoricalDtype(categories = [0, 1, 2, 3, 4], ordered = True))
                    )
            )
            
            # Filter dataframe using farther back window
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
            final_df['ecog_index'] = final_df['ecog_index'].astype(pd.CategoricalDtype(categories=[0, 1, 2, 3, 4], ordered=True))
            final_df['ecog_newly_gte2'] = final_df['ecog_newly_gte2'].astype('Int64')

            # Check for duplicate PatientIDs
            if len(final_df) > final_df['PatientID'].nunique():
                duplicate_ids = final_df[final_df.duplicated(subset = ['PatientID'], keep = False)]['PatientID'].unique()
                logging.warning(f"Duplicate PatientIDs found: {duplicate_ids}")
                
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
                       days_after: int = 0,
                       vital_summary_lookback: int = 180, 
                       abnormal_reading_threshold: int = 2) -> Optional[pd.DataFrame]:
        """
        Processes Vitals.csv to determine patient BMI, weight, change in weight, and vital sign abnormalities
        within a specified time window relative to an index date. Two different time windows are used:
        
        1. A smaller window near the index date to find weight and BMI at that time point
        2. A larger lookback window to detect clinically significant vital sign abnormalities 
        suggesting possible deterioration

        Parameters
        ----------
        file_path : str
            Path to Vitals.csv file
        index_date_df : pd.DataFrame
            DataFrame containing unique PatientIDs and their corresponding index dates. Only vitals for PatientIDs present in this DataFrame will be processed
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
        pd.DataFrame or None
            - PatientID : object 
                unique patient identifier
            - weight_index : float
                weight in kg closest to index date within specified window (index_date - weight_days_before) to (index_date + weight_days_after)
            - bmi_index : float
                BMI closest to index date within specified window (index_date - weight_days_before) to (index_date + days_after)
            - percent_change_weight : float
                percentage change in weight over period from (index_date - vital_summary_lookback) to (index_date + days_after)
            - hypotension : Int64
                binary indicator (0/1) for systolic blood pressure <90 mmHg on ≥{abnormal_reading_threshold} separate readings 
                between (index_date - vital_summary_lookback) and (index_date + days_after)
            - tachycardia : Int64
                binary indicator (0/1) for heart rate >100 bpm on ≥{abnormal_reading_threshold} separate readings 
                between (index_date - vital_summary_lookback) and (index_date + days_after)
            - fevers : Int64
                binary indicator (0/1) for temperature >=38°C on ≥{abnormal_reading_threshold} separate readings 
                between (index_date - vital_summary_lookback) and (index_date + days_after)
            - hypoxemia : Int64
                binary indicator (0/1) for SpO2 <90% on ≥{abnormal_reading_threshold} separate readings 
                between (index_date - vital_summary_lookback) and (index_date + days_after)

        Notes
        -----
        Data cleaning and processing: 
        - Missing TestResultCleaned values are imputed using TestResult. For those where units are ambiguous, unit conversion is based on thresholds:
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
        - Weight closest to index date is selected by minimum absolute day difference
        - BMI is calculated using closest weight to index within specified window and mean height over patient's entire data range (weight(kg)/height(m)²)
        - BMI calucalted as <13 are considered implausible and removed
        - Percent change in weight is calculated as ((end_weight - start_weight) / start_weight) * 100
        - TestDate rather than ResultDate is used since TestDate is always populated and, for vital signs, the measurement date (TestDate) and result date (ResultDate) should be identical since vitals are recorded in real-time
        
        Output handling: 
        - All PatientIDs from index_date_df are included in the output and values will be NaN for patients without weight, BMI, or percent_change_weight, but set to 0 for hypotension, tachycardia, fevers, and hypoxemia 
        - Duplicate PatientIDs are logged as warnings but retained in output
        - Results are stored in self.vitals_df attribute
        """
        # Input validation
        if not isinstance(index_date_df, pd.DataFrame):
            raise ValueError("index_date_df must be a pandas DataFrame")
        if 'PatientID' not in index_date_df.columns:
            raise ValueError("index_date_df must contain a 'PatientID' column")
        if not index_date_column or index_date_column not in index_date_df.columns:
            raise ValueError('index_date_column not found in index_date_df')
        if index_date_df['PatientID'].duplicated().any():
            raise ValueError("index_date_df contains duplicate PatientID values, which is not allowed")
        
        if not isinstance(weight_days_before, int) or weight_days_before < 0:
            raise ValueError("weight_days_before must be a non-negative integer")
        if not isinstance(days_after, int) or days_after < 0:
            raise ValueError("days_after must be a non-negative integer")
        if not isinstance(vital_summary_lookback, int) or vital_summary_lookback < 0:
            raise ValueError("vital_summary_lookback must be a non-negative integer")
        if not isinstance(abnormal_reading_threshold, int) or abnormal_reading_threshold < 1:
            raise ValueError("abnormal_reading_threshold must be an integer ≥1")
        
        index_date_df = index_date_df.copy()
        # Rename all columns from index_date_df except PatientID to avoid conflicts with merging and processing 
        for col in index_date_df.columns:
            if col != 'PatientID':  # Keep PatientID unchanged for merging
                index_date_df.rename(columns={col: f'imported_{col}'}, inplace=True)

        # Update index_date_column name
        index_date_column = f'imported_{index_date_column}'

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
                .rename(columns = {'TestResultCleaned': 'weight_index'})
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
            has_both_measures = weight_index_df['weight_index'].notna() & weight_index_df['height'].notna()
            
            # Only calculate BMI where both measurements exist
            weight_index_df.loc[has_both_measures, 'bmi_index'] = (
                weight_index_df.loc[has_both_measures, 'weight_index'] / 
                weight_index_df.loc[has_both_measures, 'height']**2
            )

            # Replace implausible BMI values with NaN
            implausible_bmi = weight_index_df['bmi_index'] < 13
            weight_index_df.loc[implausible_bmi, 'bmi_index'] = np.nan
                    
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
                    'TestResultCleaned': lambda x: sum(x < 90) >= abnormal_reading_threshold 
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
                     summary_lookback: int = 180) -> Optional[pd.DataFrame]:
        """
        Processes Lab.csv to determine patient lab values within a specified time window relative to an index date. Returns CBC and CMP values 
        nearest to index date, along with summary statistics (max, min, standard deviation, and slope) calculated over the summary period. 
        Additional lab tests can be included by providing corresponding LOINC code mappings.

        Parameters
        ----------
        file_path : str
            Path to Labs.csv file
        index_date_df : pd.DataFrame
            DataFrame containing unique PatientIDs and their corresponding index dates. Only labs for PatientIDs present in this DataFrame will be processed
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
        pd.DataFrame or None
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
            - {lab}_slope : float, rate of change over time (days)

        Notes
        -----
        Data cleaning and processing: 
        - Imputation strategy for lab dates: missing ResultDate is imputed with TestDate
        - Imputation strategy for lab values:
            - For each lab, missing TestResultCleaned values are imputed from TestResult after removing flags (L, H, <, >)
            - Values outside physiological ranges for each lab are filtered out
        - Unit conversion corrections:
            - Hemoglobin: Values in g/uL are divided by 100,000 to convert to g/dL
            - WBC/Platelet: Values in 10*3/L are multiplied by 1,000,000; values in /mm3 or 10*3/mL are multiplied by 1,000
            - Creatinine/BUN/Calcium: Values in mg/L are multiplied by 10 to convert to mg/dL
            - Albumin: Values in mg/dL are multiplied by 1,000 to convert to g/L; values 1-6 are assumed to be g/dL and multiplied by 10
        - Lab value selection:
            - Baseline lab value closest to index date is selected by minimum absolute day difference within window period of 
            (index_date - days_before) to (index_date + days_after)
            - Summary lab values are calculated within window period of (index_date - summary_lookback) to (index_date + days_after)
        For slope calculation:
            - Patient needs at least 2 valid measurements, at 2 valid time points, and time points must not be identical
        
        Output handling: 
        - All PatientIDs from index_date_df are included in the output and values are NaN for patients without lab values 
        - Duplicate PatientIDs are logged as warnings but retained in output 
        - Results are stored in self.labs_df attribute
        """
        # Input validation
        if not isinstance(index_date_df, pd.DataFrame):
            raise ValueError("index_date_df must be a pandas DataFrame")
        if 'PatientID' not in index_date_df.columns:
            raise ValueError("index_date_df must contain a 'PatientID' column")
        if not index_date_column or index_date_column not in index_date_df.columns:
            raise ValueError('index_date_column not found in index_date_df')
        if index_date_df['PatientID'].duplicated().any():
            raise ValueError("index_date_df contains duplicate PatientID values, which is not allowed")
        
        if not isinstance(days_before, int) or days_before < 0:
            raise ValueError("days_before must be a non-negative integer")
        if not isinstance(days_after, int) or days_after < 0:
            raise ValueError("days_after must be a non-negative integer")
        if not isinstance(summary_lookback, int) or summary_lookback < 0:
            raise ValueError("summary_lookback must be a non-negative integer")
        
        index_date_df = index_date_df.copy()
        # Rename all columns from index_date_df except PatientID to avoid conflicts with merging and processing 
        for col in index_date_df.columns:
            if col != 'PatientID':  # Keep PatientID unchanged for merging
                index_date_df.rename(columns={col: f'imported_{col}'}, inplace=True)

        # Update index_date_column name
        index_date_column = f'imported_{index_date_column}'
        
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
            all_loinc_codes = sum(self.LOINC_MAPPINGS.values(), [])

            # Filter for LOINC codes 
            df = df[df['LOINC'].isin(all_loinc_codes)]

            # Map LOINC codes to lab names
            for lab_name, loinc_codes in self.LOINC_MAPPINGS.items():
                mask = df['LOINC'].isin(loinc_codes)
                df.loc[mask, 'lab_name'] = lab_name

            ## CBC PROCESSING ##
            
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
            
            # Platelet: Convert to TestResult to numeric after removing L, H, <, and >; filter for ranges from 0-1000; and impute to TestResultCleaned
            mask = df.query('lab_name == "platelet" and TestResultCleaned.isna() and TestResult.notna()').index
            df.loc[mask, 'TestResultCleaned'] = (
                pd.to_numeric(df.loc[mask, 'TestResult'].str.replace(r'[LH<>]', '', regex=True).str.strip(), errors='coerce')
                .where(lambda x: (x >= 0) & (x <= 1000))
            )

            ## CMP PROCESSING ##
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
            # Need to multiply 1000 to restore correct value
            mask = (
                (df['lab_name'] == 'albumin') & 
                (df['TestUnits'] == 'mg/dL')
            )
            df.loc[mask, 'TestResultCleaned'] = df.loc[mask, 'TestResultCleaned'] * 1000         

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
                df_lab_summary_filtered
                .groupby(['PatientID', 'lab_name'])[['index_to_lab', 'TestResultCleaned']]
                .apply(lambda x: np.polyfit(x['index_to_lab'],
                                            x['TestResultCleaned'],
                                            1)[0]                       # Extract slope coefficient with [0]
                    if (x['TestResultCleaned'].notna().sum() > 1 and    # Need at least 2 valid measurements
                        x['index_to_lab'].notna().sum() > 1 and         # Need at least 2 valid time points
                        len(x['index_to_lab'].unique()) > 1)            # Time points must not be identical
                    else np.nan)                                        # Return NaN if conditions for valid slope calculation aren't met
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
            self.labs_df = None
            return final_df
        
        except Exception as e:
            logging.error(f"Error processing Lab.csv file: {e}")
            return None
        
    def process_medications(self,
                            file_path: str,
                            index_date_df: pd.DataFrame,
                            index_date_column: str,
                            days_before: int = 90,
                            days_after: int = 0) -> Optional[pd.DataFrame]:
        """
        Processes MedicationAdministration.csv to determine clinically relevant medicines received by patients within a specified time window 
        relative to an index date. 
        
        Parameters
        ----------
        file_path : str
            Path to MedicationAdministration.csv file
        index_date_df : pd.DataFrame
            DataFrame containing unique PatientIDs and their corresponding index dates. Only medicines for PatientIDs present in this DataFrame will be processed
        index_date_column : str
            Column name in index_date_df containing the index date
        days_before : int, optional
            Number of days before the index date to include for window period. Must be >= 0. Default: 90
        days_after : int, optional
            Number of days after the index date to include for window period. Must be >= 0. Default: 0
        
        Returns
        -------
        pd.DataFrame or None
            - PatientID : ojbect
                unique patient identifier
            - anticoagulant : Int64
                binary indicator (0/1) for therapeutic anticoagulation (heparin with specific units [e.g., "unit/kg/hr", "U/hr", "U/kg"], 
                enoxaparin >40mg, dalteparin >5000u, fondaparinux >2.5mg, or any DOAC/warfarin) 
            - opioid : Int64
                binary indicator (0/1) for oral, transdermal, sublingual, or enteral opioids
            - steroid : Int64
                binary indicator (0/1) for oral steroids
            - antibiotic : Int64
                binary indicator (0/1) for oral/IV antibiotics (excluding antifungals/antivirals)
            - diabetic_med : Int64
                binary indicator (0/1) for antihyperglycemic medication 
            - antidepressant : Int64
                binary indicator (0/1) for antidepressant
            - bone_therapy_agent : Int64
                binary indicator (0/1) for bone-targeted therapy (e.g., bisphosphonates, denosumab)
            - immunosuppressant : Int64
                binary indicator (0/1) for immunosuppressive medications

        Notes
        -----
        Output handling: 
        - All PatientIDs from index_date_df are included in the output
        - Duplicate PatientIDs are logged as warnings but retained in output 
        - Results are stored in self.medicines_df attribute
        """
        # Input validation
        if not isinstance(index_date_df, pd.DataFrame):
            raise ValueError("index_date_df must be a pandas DataFrame")
        if 'PatientID' not in index_date_df.columns:
            raise ValueError("index_date_df must contain a 'PatientID' column")
        if not index_date_column or index_date_column not in index_date_df.columns:
            raise ValueError('index_date_column not found in index_date_df')
        if index_date_df['PatientID'].duplicated().any():
            raise ValueError("index_date_df contains duplicate PatientID values, which is not allowed")
        
        if not isinstance(days_before, int) or days_before < 0:
            raise ValueError("days_before must be a non-negative integer")
        if not isinstance(days_after, int) or days_after < 0:
            raise ValueError("days_after must be a non-negative integer")
        
        index_date_df = index_date_df.copy()
        # Rename all columns from index_date_df except PatientID to avoid conflicts with merging and processing 
        for col in index_date_df.columns:
            if col != 'PatientID':  # Keep PatientID unchanged for merging
                index_date_df.rename(columns={col: f'imported_{col}'}, inplace=True)

        # Update index_date_column name
        index_date_column = f'imported_{index_date_column}'

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
            
            anticoagulant_IDs = pd.concat([
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
                )]).unique()
            
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
                "fosfomycin" 
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

            immunosuppressant_IDs = (
                df_filtered
                .query('DrugCategory == "immunosuppressive"')
                .PatientID
            ).unique()

            # Create dictionary of medication categories and their respective IDs
            med_categories = {
                'anticoagulant': anticoagulant_IDs,
                'opioid': opioid_IDs,
                'steroid': steroid_IDs,
                'antibiotic': antibiotic_IDs,
                'diabetic_med': diabetic_IDs,
                'antidepressant': antidepressant_IDs,
                'bone_therapy_agent': bta_IDs,
                'immunosuppressant': immunosuppressant_IDs
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
                          days_after: int = 0) -> Optional[pd.DataFrame]:
        """
        Processes Diagnosis.csv by mapping ICD 9 and 10 codes to Elixhauser comorbidity index and calculates a van Walraven score. 
        It also determines site of metastases based on ICD 9 and 10 codes. 
        
        Parameters
        ----------
        file_path : str
            Path to Diagnosis.csv file
        index_date_df : pd.DataFrame
            DataFrame containing unique PatientIDs and their corresponding index dates. Only diagnoses for PatientIDs present in this DataFrame will be processed
        index_date_column : str
            Column name in index_date_df containing the index date
        days_before : int | None, optional
            Number of days before the index date to include for window period. Must be >= 0 or None. If None, includes all prior results. Default: None
        days_after : int, optional
            Number of days after the index date to include for window period. Must be >= 0. Default: 0
        
        Returns
        -------
        pd.DataFrame or None
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
            - renal_failure : binary indicator for renal failure
            - liver_disease : binary indicator for liver disease
            - PUD : binary indicator for peptic ulcer disease
            - aids_hiv : binary indicator for AIDS/HIV
            - lymphoma : binary indicator for lymphoma
            - rheumatic : binary indicator for rheumatoid arthritis/collagen vascular diseases
            - coagulopathy : binary indicator for coagulopathy
            - obesity : binary indicator for obesity
            - weight_loss : binary indicator for weight loss
            - fluid : binary indicator for fluid and electrolyte disorders
            - blood_loss_anemia : binary indicator for blood loss anemia
            - deficiency_anemia : binary indicator for deficiency anemia
            - alcohol_abuse : binary indicator for alcohol abuse
            - drug_abuse : binary indicator for drug abuse
            - psychoses : binary indicator for psychoses
            - depression : binary indicator for depression
            - van_walraven_score : weighted composite of the binary Elixhauser comorbidities 
            - lymph_met : binary indicator for lymph node metastasis
            - thoracic_met : binary indicator for thoracic metastasis (eg., lung, pleura, mediastinum, or other respiratory)
            - liver_met : binary indicator for liver metastasis
            - bone_met : binary indicator for bone metastasis
            - brain_met : binary indicator for brain/CNS metastasis
            - adrenal_met : binary indicator for adrenal metastasis
            - peritoneum_met : binary indicator for peritoneal/retroperitoneal metastasis
            - gi_met : binary indicator for gastrointestinal tract metastasis excluding liver, adrenal, and peritoneum
            - other_met : binary indicator for other sites of metastasis

        Notes
        -----
        Mapping information: 
        - See "Coding algorithms for defining comorbidities in ICD-9-CM and ICD-10 administrative data" by Quan et al for details on ICD mapping to comorbidities. 
        For ICD-9 codes, the Enhanced ICD-9-CM by Quan was used for mapping.
        - See "A modification of the Elixhauser comorbidity measures into a point system for hospital death using administrative data" by van Walraven et al for 
        details on van Walraven score.
        - Metastatic cancer and tumor categories are excluded in the Elxihauser comorbidities and van Walraven score as this is intended for an advanced cancer population
        
        Output handling: 
        - All PatientIDs from index_date_df are included in the output and values will be set to 0 for patients with misisng Elixhauser comorbidities or metastasis sites, but NaN for missing van_walraven_score
        - Duplicate PatientIDs are logged as warnings but retained in output
        - Results are stored in self.diagnoses_df attribute
        """
        # Input validation
        if not isinstance(index_date_df, pd.DataFrame):
            raise ValueError("index_date_df must be a pandas DataFrame")
        if 'PatientID' not in index_date_df.columns:
            raise ValueError("index_date_df must contain a 'PatientID' column")
        if not index_date_column or index_date_column not in index_date_df.columns:
            raise ValueError('index_date_column not found in index_date_df')
        if index_date_df['PatientID'].duplicated().any():
            raise ValueError("index_date_df contains duplicate PatientID values, which is not allowed")
        
        if days_before is not None:
            if not isinstance(days_before, int) or days_before < 0:
                raise ValueError("days_before must be a non-negative integer or None")
        if not isinstance(days_after, int) or days_after < 0:
            raise ValueError("days_after must be a non-negative integer")
        
        index_date_df = index_date_df.copy()
        # Rename all columns from index_date_df except PatientID to avoid conflicts with merging and processing 
        for col in index_date_df.columns:
            if col != 'PatientID':  # Keep PatientID unchanged for merging
                index_date_df.rename(columns={col: f'imported_{col}'}, inplace=True)

        # Update index_date_column name
        index_date_column = f'imported_{index_date_column}'

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

            df['index_to_diagnosis'] = (df['DiagnosisDate'] - df[index_date_column]).dt.days
            
            # Select ICD codes that fall within desired before and after index date
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
                    lambda code: next((comorb for pattern, comorb in self.ICD_10_ELIXHAUSER_MAPPING.items() 
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

            all_columns_elix = ['PatientID'] + list(self.ICD_9_EXLIXHAUSER_MAPPING.values())
            
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
                .astype('Int64')  
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
                .astype('Int64')  
                .rename_axis(columns = None)
                .reset_index()
            )

            all_columns_mets = ['PatientID'] + list(self.ICD_9_METS_MAPPING.values())
            
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
                duplicate_ids = final_df[final_df.duplicated(subset = ['PatientID'], keep = False)]['PatientID'].unique()
                logging.warning(f"Duplicate PatientIDs found: {duplicate_ids}")

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
                          days_after: int = 0,
                          missing_date_strategy: str = 'conservative') -> Optional[pd.DataFrame]:
        """
        Processes insurance data to identify insurance coverage relative to a specified index date.
        Insurance types are grouped into four categories: Medicare, Medicaid, Commercial, and Other Insurance. 
        
        Parameters
        ----------
        file_path : str
            Path to Insurance.csv file
        index_date_df : pd.DataFrame
            DataFrame containing unique PatientIDs and their corresponding index dates. Only insurances for PatientIDs present in this DataFrame will be processed
        index_date_column : str
            Column name in index_date_df containing the index date
        days_before : int | None, optional
            Number of days before the index date to include for window period. Must be >= 0 or None. If None, includes all prior results. Default: None
        days_after : int, optional
            Number of days after the index date to include for window period. Must be >= 0. Default: 0
        missing_date_strategy : str
            Strategy for handling missing StartDate:
            - 'conservative': Excludes records with both StartDate and EndDate missing and imputes EndDate for missing StartDate (may underestimate coverage)
            - 'liberal': Assumes records with missing StartDates are always active and imputes default date of 2000-01-01 (may overestimate coverage)
        
        Returns
        -------
        pd.DataFrame or None
            - PatientID : object
                unique patient identifier
            - medicare : Int64
                binary indicator (0/1) for Medicare coverage
            - medicaid : Int64
                binary indicator (0/1) for Medicaid coverage
            - commercial : Int64
                binary indicator (0/1) for commercial insurance coverage
            - other_insurance : Int64
                binary indicator (0/1) for other insurance types (eg., other payer, other government program, patient assistance program, self pay, and workers compensation)

        Notes
        -----
        Insurance is considered active if:
        1. StartDate falls before or during the specified time window AND
        2. Either:
            - EndDate is missing (considered still active) OR
            - EndDate falls on or after the start of the time window 
        
        Date filtering:
        - Records with StartDate or EndDate before 1900-01-01 are excluded to prevent integer overflow issues
        when calculating date differences. This is a data quality measure as extremely old dates are likely
        erroneous and can cause numerical problems in pandas datetime calculations.
        - About 5% of the full dataset has misisng StartDate and EndDate.

        Insurance categorization logic:
        1. Original payer categories are preserved but enhanced with hybrid categories:
        - Commercial_Medicare: Commercial plans with Medicare Advantage or Supplement
        - Commercial_Medicaid: Commercial plans with Managed Medicaid
        - Commercial_Medicare_Medicaid: Commercial plans with both Medicare and Medicaid indicators
        - Other_Medicare: Other government program or other payer plans with Medicare Advantage or Supplement
        - Other_Medicaid: Other government program or other payer plans with Managed Medicaid
        - Other_Medicare_Medicaid: Other government program or other payer plans with both Medicare and Medicaid indicators
            
        2. Final insurance indicators are set as follows:
        - medicare: Set to 1 for PayerCategory = Medicare, Commercial_Medicare, Commercial_Medicare_Medicaid, Other_Medicare, or Other_Medicare_Medicaid
        - medicaid: Set to 1 for PayerCategory = Medicaid, Commercial_Medicaid, Commercial_Medicare_Medicaid, Other_Medicaid, or Other_Medicare_Medicaid
        - commercial: Set to 1 for PayerCategory = Commercial Health Plan, Commercial_Medicare, Commercial_Medicaid, or Commercial_Medicare_Medicaid
        - other_insurance: Set to 1 for PayerCategory = Other Payer - Type Unknown, Other Government Program, Patient Assistance Program, Self Pay, 
            Workers Compensation, Other_Medicare, Other_Medicaid, Other_Medicare_Medicaid

        Output handling:     
        - All PatientIDs from index_date_df are included in the output and value is set to 0 for those without insurance type
        - Duplicate PatientIDs are logged as warnings but retained in output
        - Results are stored in self.insurance_df attribute
        """
        # Input validation
        if not isinstance(index_date_df, pd.DataFrame):
            raise ValueError("index_date_df must be a pandas DataFrame")
        if 'PatientID' not in index_date_df.columns:
            raise ValueError("index_date_df must contain a 'PatientID' column")
        if not index_date_column or index_date_column not in index_date_df.columns:
            raise ValueError('index_date_column not found in index_date_df')
        if index_date_df['PatientID'].duplicated().any():
            raise ValueError("index_date_df contains duplicate PatientID values, which is not allowed")
        
        if days_before is not None:
            if not isinstance(days_before, int) or days_before < 0:
                raise ValueError("days_before must be a non-negative integer or None")
        if not isinstance(days_after, int) or days_after < 0:
            raise ValueError("days_after must be a non-negative integer")
        
        if not isinstance(missing_date_strategy, str):
            raise ValueError("missing_date_strategy must be a string")    
        valid_strategies = ['conservative', 'liberal']
        if missing_date_strategy not in valid_strategies:
            raise ValueError("missing_date_strategy must be 'conservative' or 'liberal'")
        
        index_date_df = index_date_df.copy()
        # Rename all columns from index_date_df except PatientID to avoid conflicts with merging and processing 
        for col in index_date_df.columns:
            if col != 'PatientID':  # Keep PatientID unchanged for merging
                index_date_df.rename(columns={col: f'imported_{col}'}, inplace=True)

        # Update index_date_column name
        index_date_column = f'imported_{index_date_column}'

        try:
            df = pd.read_csv(file_path)
            logging.info(f"Successfully read Insurance.csv file with shape: {df.shape} and unique PatientIDs: {(df['PatientID'].nunique())}")

            df['StartDate'] = pd.to_datetime(df['StartDate'])
            df['EndDate'] = pd.to_datetime(df['EndDate'])

            both_dates_missing = df['StartDate'].isna() & df['EndDate'].isna()
            start_date_missing = df['StartDate'].isna()

            if missing_date_strategy == 'conservative':
                # Exclude records with both dates missing, and impute EndDate for missing StartDate
                df = df[~both_dates_missing]
                df['StartDate'] = np.where(df['StartDate'].isna(), df['EndDate'], df['StartDate'])
            elif missing_date_strategy == 'liberal':
                # Assume always active by setting StartDate to default date of 2000-01-01
                df.loc[start_date_missing, 'StartDate'] = pd.Timestamp('2000-01-01')

            # Filter for StartDate after 1900-01-01
            df = df[df['StartDate'] > pd.Timestamp('1900-01-01')]
            # Filter for Enddate missing or after 1900-01-01
            df = df[(df['EndDate'].isna()) | (df['EndDate'] > pd.Timestamp('1900-01-01'))]

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

            # Reclassify Commerical Health Plans that have elements of Medicare, Medicaid, or Both
            # Identify Commerical plus Medicare Advantage or Supplement plans
            df['PayerCategory'] = np.where((df['PayerCategory'] == 'Commercial Health Plan') & ((df['IsMedicareAdv'] == 'Yes') | (df['IsMedicareSupp'] == 'Yes')) & (df['IsMedicareMedicaid'] != 'Yes') & (df['IsManagedMedicaid'] != 'Yes'),
                                            'Commercial_Medicare',
                                            df['PayerCategory'])

            # Identify Commerical plus Managed Medicaid plans
            df['PayerCategory'] = np.where((df['PayerCategory'] == 'Commercial Health Plan') & (df['IsManagedMedicaid'] == 'Yes') & (df['IsMedicareMedicaid'] != 'Yes') & (df['IsMedicareAdv'] != 'Yes') & (df['IsMedicareSupp'] != 'Yes'),
                                            'Commercial_Medicaid',
                                            df['PayerCategory'])
            
            # Identify Commercial plus MedicareMedicaid plan
            df['PayerCategory'] = np.where((df['PayerCategory'] == 'Commercial Health Plan') & (df['IsMedicareMedicaid'] == 'Yes'),
                                            'Commercial_Medicare_Medicaid',
                                            df['PayerCategory'])

            # Identify Commercial plus Managed Medicaid and Medicare Advantage or Supplement plans
            df['PayerCategory'] = np.where((df['PayerCategory'] == 'Commercial Health Plan') & (df['IsManagedMedicaid'] == 'Yes') & ((df['IsMedicareAdv'] == 'Yes') | (df['IsMedicareSupp'] == 'Yes')),
                                            'Commercial_Medicare_Medicaid',
                                            df['PayerCategory'])
            

            # Reclassify Other Health Plans that have elements of Medicare, Medicaid, or Both
            # Identify Other plus Medicare Advantage or Supplement plans
            df['PayerCategory'] = np.where(((df['PayerCategory'] == 'Other Payer - Type Unknown') | (df['PayerCategory'] == 'Other Government Program')) & ((df['IsMedicareAdv'] == 'Yes') | (df['IsMedicareSupp'] == 'Yes')) & (df['IsMedicareMedicaid'] != 'Yes') & (df['IsManagedMedicaid'] != 'Yes'),
                                            'Other_Medicare',
                                            df['PayerCategory'])

            # Identify Other plus Managed Medicaid plans
            df['PayerCategory'] = np.where(((df['PayerCategory'] == 'Other Payer - Type Unknown') | (df['PayerCategory'] == 'Other Government Program')) & (df['IsManagedMedicaid'] == 'Yes') & (df['IsMedicareMedicaid'] != 'Yes') & (df['IsMedicareAdv'] != 'Yes') & (df['IsMedicareSupp'] != 'Yes'),
                                            'Other_Medicaid',
                                            df['PayerCategory'])
            
            # Identify Other plus MedicareMedicaid plan
            df['PayerCategory'] = np.where(((df['PayerCategory'] == 'Other Payer - Type Unknown') | (df['PayerCategory'] == 'Other Government Program')) & (df['IsMedicareMedicaid'] == 'Yes'),
                                            'Other_Medicare_Medicaid',
                                            df['PayerCategory'])

            # Identify Other plus Managed Medicaid and Medicare Advantage or Supplement plans
            df['PayerCategory'] = np.where(((df['PayerCategory'] == 'Other Payer - Type Unknown') | (df['PayerCategory'] == 'Other Government Program')) & (df['IsManagedMedicaid'] == 'Yes') & ((df['IsMedicareAdv'] == 'Yes') | (df['IsMedicareSupp'] == 'Yes')),
                                            'Other_Medicare_Medicaid',
                                            df['PayerCategory'])
            
            # Add hybrid insurance schems to mapping
            self.INSURANCE_MAPPING['Commercial_Medicare'] = 'commercial_medicare'
            self.INSURANCE_MAPPING['Commercial_Medicaid'] = 'commercial_medicaid'
            self.INSURANCE_MAPPING['Commercial_Medicare_Medicaid'] = 'commercial_medicare_medicaid'
            self.INSURANCE_MAPPING['Other_Medicare'] = 'other_medicare'
            self.INSURANCE_MAPPING['Other_Medicaid'] = 'other_medicaid'
            self.INSURANCE_MAPPING['Other_Medicare_Medicaid'] = 'other_medicare_medicaid'

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
                .astype('Int64')  
                .rename_axis(columns = None)
                .reset_index()
            )

            # Adjust column indicators for commercial and other with medicare and medicaid plans
            if 'commercial_medicare' in final_df.columns:
                final_df.loc[final_df['commercial_medicare'] == 1, 'commercial'] = 1
                final_df.loc[final_df['commercial_medicare'] == 1, 'medicare'] = 1
            
            if 'commercial_medicaid' in final_df.columns:
                final_df.loc[final_df['commercial_medicaid'] == 1, 'commercial'] = 1
                final_df.loc[final_df['commercial_medicaid'] == 1, 'medicaid'] = 1

            if 'commercial_medicare_medicaid' in final_df.columns:
                final_df.loc[final_df['commercial_medicare_medicaid'] == 1, 'commercial'] = 1
                final_df.loc[final_df['commercial_medicare_medicaid'] == 1, 'medicare'] = 1
                final_df.loc[final_df['commercial_medicare_medicaid'] == 1, 'medicaid'] = 1
            
            if 'other_medicare' in final_df.columns:
                final_df.loc[final_df['other_medicare'] == 1, 'other_insurance'] = 1
                final_df.loc[final_df['other_medicare'] == 1, 'medicare'] = 1
            
            if 'other_medicaid' in final_df.columns:
                final_df.loc[final_df['other_medicaid'] == 1, 'other_insurance'] = 1
                final_df.loc[final_df['other_medicaid'] == 1, 'medicaid'] = 1

            if 'other_medicare_medicaid' in final_df.columns:
                final_df.loc[final_df['other_medicare_medicaid'] == 1, 'other_insurance'] = 1
                final_df.loc[final_df['other_medicare_medicaid'] == 1, 'medicare'] = 1
                final_df.loc[final_df['other_medicare_medicaid'] == 1, 'medicaid'] = 1

            # Merger index_date_df to ensure all PatientIDs are included
            final_df = pd.merge(index_date_df[['PatientID']], final_df, on = 'PatientID', how = 'left')
            
            # Ensure all core insurance columns exist 
            core_insurance_columns = ['medicare', 'medicaid', 'commercial', 'other_insurance']
            for col in core_insurance_columns:
                if col not in final_df.columns:
                    final_df[col] = 0
                final_df[col] = final_df[col].fillna(0).astype('Int64')

            # Safely drop hybrid columns if they exist
            hybrid_columns = ['commercial_medicare', 
                              'commercial_medicaid', 
                              'commercial_medicare_medicaid',
                              'other_medicare', 
                              'other_medicaid', 
                              'other_medicare_medicaid']
            
            # Drop hybrid columns; errors = 'ignore' prevents error in the setting when column doesn't exist 
            final_df = final_df.drop(columns=hybrid_columns, errors='ignore')

            # Check for duplicate PatientIDs
            if len(final_df) > final_df['PatientID'].nunique():
                duplicate_ids = final_df[final_df.duplicated(subset = ['PatientID'], keep = False)]['PatientID'].unique()
                logging.warning(f"Duplicate PatientIDs found: {duplicate_ids}")

            logging.info(f"Successfully processed Insurance.csv file with final shape: {final_df.shape} and unique PatientIDs: {(final_df['PatientID'].nunique())}")
            self.insurance_df = final_df
            return final_df

        except Exception as e:
            logging.error(f"Error processing Insurance.csv file: {e}")
            return None