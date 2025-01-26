import pandas as pd
import logging

logging.basicConfig(
    level = logging.INFO,                                 
    format = '%(asctime)s - %(levelname)s - %(message)s'  
)

class DataProcessorUrothelial:
    def __init__(self):
        self.enhanced_df = None
        self.demographics_df = None

    def process_enhanced_adv(self, file_path: str, drop_dates: bool = True) -> pd.DataFrame: 
        try:
            df = pd.read_csv(file_path)
            logging.info(f"Successfully read Enhanced_AdvUrothelial.csv file with shape: {df.shape} and unique PatientIDs: {(df.PatientID.nunique())}")
        
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
        
            # Convert date columns
            date_cols = ['DiagnosisDate', 'AdvancedDiagnosisDate', 'SurgeryDate']
            for col in date_cols:
                df[col] = pd.to_datetime(df[col])
            
            # Convert boolean column
            df['Surgery'] = df['Surgery'].astype(bool)

            # Generate new variables 
            df['days_diagnosis_to_advanced'] = (df['AdvancedDiagnosisDate'] - df['DiagnosisDate']).dt.days
            df['advanced_diagnosis_year'] = pd.Categorical(df['AdvancedDiagnosisDate'].dt.year)
            df['days_diagnosis_to_surgery'] = (df['SurgeryDate'] - df['DiagnosisDate']).dt.days
    
            # Drop date-based variables if specified
            if drop_dates:
                df = df.drop(columns=['AdvancedDiagnosisDate', 'DiagnosisDate', 'SurgeryDate'])

            # Check for duplicate PatientIDs
            if len(df) > df.PatientID.nunique():
                logging.error(f"Duplicate PatientIDs found")
                return None

            logging.info(f"Successfully processed Enhanced_AdvUrothelial.csv file with final shape: {df.shape} and unique PatientIDs: {(df.PatientID.nunique())}")
            self.enhanced_df = df
            return df

        except Exception as e:
            logging.error(f"Error processing Enhanced_AdvUrothelial.csv file: {e}")
            return None
    
    def process_demographics(self, file_path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path)
            logging.info(f"Successfully read Demographics.csv file with shape: {df.shape} and unique PatientIDs: {(df.PatientID.nunique())}")
        
            logging.info(f"Successfully processed Demographics.csv file with final shape: {df.shape} and unique PatientIDs: {(df.PatientID.nunique())}")
            self.demographics_df = df
            return df
    
        except Exception as e:
            logging.error(f"Error processing Demographics.csv file: {e}")
            return None
    
    def merge_datasets(self, *dataframes: pd.DataFrame) -> pd.DataFrame:
        """
        Outer merge of multiple datasets based on PatientID
    
        Parameters:
        *dataframes: Variable number of pandas DataFrames to merge
    
        Returns:
        pd.DataFrame: Merged dataset
        """
        # Check if any dataframes are provided
        if not dataframes:
            logging.error("No dataframes provided for merging")
            return None
        
        try:
            # Log pre-merge information for each dataset
            for i, df in enumerate(dataframes):
                logging.info(f"Pre-merge unique PatientIDs in dataset {i+1}: {df.PatientID.nunique()}")
        
            # Start with first dataframe and merge others iteratively
            merged_df = dataframes[0]
        
            for i, df in enumerate(dataframes[1:], 2):
                merged_df = pd.merge(
                    merged_df,
                    df,
                    on = 'PatientID',
                    how = 'outer'
                )
                logging.info(f"After merging dataset {i}, shape: {merged_df.shape}, unique PatientIDs: {merged_df.PatientID.nunique()}")
            
            # Log final merge information
            logging.info(f"Final merged dataset shape: {merged_df.shape} with {merged_df.PatientID.nunique()} unique PatientIDs")
        
            return merged_df
            
        except Exception as e:
            logging.error(f"Error merging datasets: {e}")
            return None

if __name__ == "__main__":
    processor = DataProcessorUrothelial()
    
    enhanced_file_path = "data/Enhanced_AdvUrothelial.csv"
    demographics_file_path = "data/Demographics.csv"
    
    # Process both datasets
    enhanced_df = processor.process_enhanced_adv(enhanced_file_path)
    demographics_df = processor.process_demographics(demographics_file_path)
    
    # Merge datasets
    merged_data = processor.merge_datasets(enhanced_df, demographics_df)
    
    if merged_data is not None:
        print("\nFirst few rows of merged data:")
        print(merged_data.head())