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
        
            # Log success and number of rows
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
                logging.info("Date columns dropped from dataset")

            logging.info(f"Successfully processed Enhanced_AdvUrothelial.csv file with final shape: {df.shape} and unique PatientIDs: {(df.PatientID.nunique())}")
            self.enhanced_df = df
            return df

        except Exception as e:
            # If any error occurs, log it and return None
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
    
    def merge_datasets(self) -> pd.DataFrame:
        """Merge enhanced and demographics datasets if both are loaded"""
        if self.enhanced_df is None or self.demographics_df is None:
            logging.error("Both datasets must be processed before merging")
            return None
            
        try:
            merged_df = pd.merge(
                self.enhanced_df, 
                self.demographics_df,
                on = 'PatientID',
                how = 'inner'
            )
            logging.info(f"Successfully merged datasets. Final shape: {merged_df.shape}. There are {merged_df.PatientID.nunique()} uniuqe PatientID's")
            return merged_df
            
        except Exception as e:
            logging.error(f"Error merging datasets: {e}")
            return None

if __name__ == "__main__":
    processor = DataProcessorUrothelial()
    
    enhanced_file_path = "data/Enhanced_AdvUrothelial.csv"
    demographics_file_path = "data/Demographics.csv"
    
    # Process both datasets
    processor.process_enhanced_adv(enhanced_file_path)
    processor.process_demographics(demographics_file_path)
    
    # Merge datasets
    merged_data = processor.merge_datasets()
    
    if merged_data is not None:
        print("\nFirst few rows of merged data:")
        print(merged_data.head())