# FlatironCleaner

`FlatironCleaner` is a Python package that cleans Flatiron Health cancer datasets into analysis-ready formats, specifically designed with predictive modeling and survival analysis in mind. By automating complex and tedious data processing workflows, it helps researchers extract meaningful insights and ensure reproducible results while reducing preparation time.

Key features of the package include:
- Providing a modular architecture that allows researchers to select which Flatiron files to process
- Converting long-format dataframes (with multiple rows per patient) into wide-format dataframes with unique PatientIDs per row
- Ensuring appropriate data types for predictive modeling and statistical analysis
- Standardizing data cleaning around a user-specified index date, such as metastatic diagnosis or treatment initiation
- Engineering clinically relevant variables for analysis

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

For cancer types without a dedicated processor, `DataProcessorGeneral` is available with standard methods. 

## Processing Methods

### Standard Methods

The following methods are available across all processor classes, including the general processor:

| Method | Description | File Processed |
|--------|-------------|----------------|
| `process_demographics()` | Processes patient demographic information | Demographics.csv |
| `process_mortality()` | Processes mortality data | Enhanced_Mortality_V2.csv |
| `process_ecog()` | Processes performance status data | ECOG.csv |
| `process_medications()` | Processes medication administration records | MedicationAdministration.csv |
| `process_diagnosis()` | Processes ICD coding information | Diagnosis.csv |
| `process_labs()` | Processes laboratory test results | Lab.csv |
| `process_vitals()` | Processes vital signs data | Vitals.csv |
| `process_insurance()` | Processes insurance information | Insurance.csv |
| `process_practice()` | Processes practice type data | Practice.csv |

### Cancer-Specific Methods

Cancer-specific classes contain additional methods (e.g., `process_enhanced()` and `process_biomarkers()`). For a complete list of available methods for each cancer type, refer to the source code or use Python's built-in help functionality:

```python
from flatiron_cleaner import DataProcessorUrothelial

# View all methods and documentation
help(DataProcessorUrothelial)

```

## Usage Example

```python
from flatiron_cleaner import DataProcessorUrothelial
from flatiron_cleaner import merge_dataframes

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