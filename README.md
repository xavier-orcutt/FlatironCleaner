# FlatironCleaner

`FlatironCleaner` is a Python package designed to clean and harmonize Flatiron Health cancer data structured around a specified index date of interest (e.g., metastatic diagnosis datw or treatment initiation). The package standardizes the process of transforming raw Flatiron CSV files into analysis-ready datasets for different cancer types. The goal is to help researchers maximize insights from the data, reduce time spent on data cleaning, and improve reproducibility. 

## Features

- **Cancer-specific processors** for different cancer types
- **Flexible date windowing** to analyze data relative to an index date
- **Consistent data harmonization** across Flatiron file types
- **Reproducible data preparation** workflows

## Installation

Coming soon. 

## Available Processors

### Cancer-Specific Processors

The following cancers have their own dedicated data processor class:

| Cancer Type | Processor Name | Status |
|-------------|-----------------|--------|
| Advanced Urothelial Cancer | `DataProcessorUrothelial` | Available |
| Advanced NSCLC | `DataProcessorNSCLC` | Available |
| Metastatic Colorectal Cancer | `DataProcessorColorectal` | Available |
| Metastatic Breast Cancer | `DataProcessorBreast` | Available |
| Metastatic Prostate Cancer | `DataProcessorProstate` | In Development |

### General Processor (Coming Soon)

For cancer types without a dedicated processor, I've provide a general processor:

```python
from flatiron_cleaner.general import DataProcessorGeneral

```

The general processor includes only standard methods.

## Processing Methods

### Standard Methods

The following methods are available across all processor classes, including the general processor:

| Method | Description | File Processed |
|--------|-------------|----------------|
| `process_demographics()` | Processes patient demographic information | Demographics.csv |
| `process_mortality()` | Handles mortality data | Enhanced_Mortality_V2.csv |
| `process_ecog()` | Processes performance status data | ECOG.csv |
| `process_medications()` | Cleans medication administration records | MedicationAdministration.csv |
| `process_diagnosis()` | Processes diagnosis information | Diagnosis.csv |
| `process_labs()` | Handles laboratory test results | Lab.csv |
| `process_vitals()` | Processes vital signs data | Vitals.csv |
| `process_insurance()` | Processes insurance information | Insurance.csv |
| `process_practice()` | Handles practice/facility data | Practice.csv |

### Cancer-Specific Methods

Cancer-specific classes contain additional methods (e.g., `process_enhanced()` and `process_biomarkers()`). For a complete list of available methods for each cancer type, refer to the class documentation in the source code or use Python's built-in help functionality:

```python
from flatiron_cleaner.urothelial import DataProcessorUrothelial

# View all methods and documentation
help(DataProcessorUrothelial)

```

## Usage Example

```python
from flatiron_cleaner.urothelial import DataProcessorUrothelial
from flatiron_cleaner.merge_utils import merge_dataframes

# Initialize class
processor = DataProcessorUrothelial()

# Import dataframe with PatientIDs and index date of interest
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

For a more detailed usage demonstration, see the notebook titled "tutorial" in the `example/` directory.

## Contact

I welcome contributions and feedback. Email me at: xavierorcutt@gmail.com