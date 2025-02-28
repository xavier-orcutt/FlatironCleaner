# FlatironCleaner

`FlatironCleaner` is a Python package designed to clean and harmonize Flatiron Health cancer data structured around a specified index date of interest (e.g., metastatic diagnosis datw or treatment initiation). The package standardizes the process of transforming raw Flatiron CSV files into analysis-ready datasets for different cancer types. The goal is to help researchers maximize insights from the data, reduce time spent on data cleaning, and improve reproducibility. 

## Overview

Each cancer type has its own dedicated data processor class (e.g., `DataProcessorUrothelial`). These classes contain specialized functions for cleaning specific Flatiron CSV files, with cleaning rules tailored to each cancer type and aligned with an index date of interest. Available functions include: 

- `process_enhanced()`: Cleans Enhanced_{cancer}.csv
- `process_demographics()`: Cleans Demographics.csv
- `process_mortality()`: Cleans Enhanced_Mortality_V2.csv
- `process_ecog()`: Cleans ECOG.csv
- `process_medications()`: Cleans MedicationAdministration.csv
- `process_diagnosis()`: Cleans Diagnosis.csv
- `process_labs()`: Cleans Lab.csv
- `process_vitals()`: Clean Vitals.csv 
- `process_biomarkers()`: Cleans {cancer}_Biomarkers.csv
- `process_insurance()`: Cleans Insurance.csv
- `process_practice()`: Cleans Practice.csv

Currently available processors:
- Advanced Urothelial Cancer (`DataProcessorUrothelial`)
- Advanced NSCLC (`DataProcessorNSCLC`)

In development:
- Metastatic Colorectal Cancer (`DataProcessorColorectal`)

Coming soon: 
- Metastatic Breast Cancer (`DataProcessorBreast`)
- Metastatic Prostate Cancer (`DataProcessorProstate`)

## Usage 

```python
from urothelial import DataProcessorUrothelial
from merge_utils import merge_dataframes

# Initialize class
processor = DataProcessorUrothelial()

# Import dataframe with index date of interest
df = pd.read_csv('path/to/your/data')

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

# Merge dataframes 
merged_data = merge_dataframes(cleaned_ecog_df, cleaned_medication_df)
```

For a more detailed usage demonstration, see the notebook titled "tutorial" in the example/ directory.

## Contact

I welcome contributions and feedback. Email me at: xavierorcutt@gmail.com