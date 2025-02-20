# Flatiron Cancer ETL

`flatiron-cancer-etl` is a Python package designed to clean and harmonize Flatiron Health cancer datasets. The pipeline standardizes the process of transforming raw Flatiron CSV files into analysis-ready datasets for different cancer types.

## Overview

Each cancer type has its own dedicated data processor class (e.g., `DataProcessorUrothelial`). These classes contain specialized functions for cleaning specific Flatiron CSV files, with cleaning rules tailored to each cancer type and aligned with an index date of interest: 

- `process_demographics()`: Cleans Demographics.csv
- `process_ecog()`: Cleans ECOG.csv
- `process_medications()`: Cleans MedicationAdministration.csv
- `process_diagnosis()`: Cleans Diagnosis.csv
- `process_labs()`: Cleans Lab.csv
- `process_vitals()`: Clean Vitals.csv 
- `process_biomarkers()`: Cleans Biomarkers.csv
- `process_insurance()`: Cleans Insurance.csv
- `process_mortality()`: Cleans Enhanced_Mortality_V2.csv
- `process_practice()`: Cleans Practice.csv

Currently available processors:
- Advanced Urothelial Cancer (`DataProcessorUrothelial`)

In development:
- Advanced NSCLC (`DataProcessorNSCLC`)

Coming soon: 
- Metastatic Breast Cancer
- Metastatic Colorectal Cancer 
- Metastatic Prostate Cancer

## Usage 

See the example notebooks in the example/ directory for usage demonstrations.

```python
from urothelial_processor import DataProcessorUrothelial
from merge_utils import merge_dataframes

# Initialize the ETL pipeline
processor = DataProcessorUrothelial()

# Load and clean data
cleaned_ecog_df = processor.process_ecog('path/to/your/ECOG.csv',
                                         index_date_df=df,
                                         index_date_column='AdvancedDiagnosisDate',
                                         days_before=30,
                                         days_after=0)                  

cleaned_medication_df = processor.process_medications('path/to/your/MedicationAdmninistration.csv',
                                                      index_date_df = df,
                                                      index_date_column='AdvancedDiagnosisDate',
                                                      days_before=180,
                                                      days_after=0)

merged_data = merge_dataframes(cleaned_ecog_df, cleaned_medication_df)
```

## Contact

I welcome contributions and feedback. Email me at: xavierorcutt@gmail.com