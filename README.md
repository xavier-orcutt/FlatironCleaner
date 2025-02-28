# FlatironCleaner

`FlatironCleaner` is a Python package designed to clean and harmonize Flatiron Health cancer data structured around a specified index date of interest (e.g., metastatic diagnosis datw or treatment initiation). The package standardizes the process of transforming raw Flatiron CSV files into analysis-ready datasets for different cancer types. The goal is to help researchers maximize insights from the data, reduce time spent on data cleaning, and improve reproducibility. 

## Features

- **Cancer-specific processors** for different cancer types
- **Standardized cleaning methods** aligned with clinical research needs
- **Flexible date windowing** to analyze data relative to an index date
- **Consistent data harmonization** across Flatiron file types
- **Reproducible data preparation** workflows

## Installation

Coming soon. 

## Available Processors

Each cancer type has its own dedicated data processor class:

| Cancer Type | Processor Class | Status |
|-------------|-----------------|--------|
| Advanced Urothelial Cancer | `DataProcessorUrothelial` | Available |
| Advanced NSCLC | `DataProcessorNSCLC` | Available |
| Metastatic Colorectal Cancer | `DataProcessorColorectal` | Available |
| Metastatic Breast Cancer | `DataProcessorBreast` | In Development |
| Metastatic Prostate Cancer | `DataProcessorProstate` | Coming Soon |

## Processing Methods

### Standard Methods

The following methods are available across all processor classes:

| Method | Description | File Processed |
|--------|-------------|----------------|
| `process_enhanced()` | Cleans cancer-specific enhanced data | Enhanced_{cancer}.csv |
| `process_demographics()` | Processes patient demographic information | Demographics.csv |
| `process_mortality()` | Handles mortality data | Enhanced_Mortality_V2.csv |
| `process_ecog()` | Processes performance status data | ECOG.csv |
| `process_medications()` | Cleans medication administration records | MedicationAdministration.csv |
| `process_diagnosis()` | Processes diagnosis information | Diagnosis.csv |
| `process_labs()` | Handles laboratory test results | Lab.csv |
| `process_vitals()` | Processes vital signs data | Vitals.csv |
| `process_biomarkers()` | Cleans cancer-specific biomarker data | {cancer}_Biomarkers.csv |
| `process_insurance()` | Processes insurance information | Insurance.csv |
| `process_practice()` | Handles practice/facility data | Practice.csv |

### Cancer-Specific Methods

Some processor classes contain additional methods specific to certain cancer types. For a complete list of available methods for each cancer type, refer to the class documentation or use Python's built-in help functionality:

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

For a more detailed usage demonstration, see the notebook titled "tutorial" in the `example/` directory.

## Contact

I welcome contributions and feedback. Email me at: xavierorcutt@gmail.com