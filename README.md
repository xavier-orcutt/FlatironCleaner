# Flatiron Cancer ETL

`flatiron-cancer-etl` is a Python package designed to clean and harmonize Flatiron Health cancer datasets. The pipeline standardizes the process of transforming raw Flatiron CSV files (eg., demographics, vitals, labs, biomarkers, etc.) into analysis-ready datasets for different cancer types.

## Overview

The pipeline automates several key data processing steps:
- Standardizes variable names and data types across cancers
- Ensures unique PatientIDs per row
- Applies cancer-specific cleaning rules
- Provides detailed logging of transformations

Each cancer type can have its own specific cleaning rules defined through configuration files, while sharing common data processing logic. This modular design allows for consistent data cleaning across different cancer research projects while maintaining flexibility for cancer-specific requirements.
