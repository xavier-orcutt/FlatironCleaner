import pandas as pd
import logging

logging.basicConfig(
    level = logging.INFO,                                 
    format = '%(asctime)s - %(levelname)s - %(message)s'  
)

def process_enhanced_adv(file_path: str) -> pd.DataFrame:
    # Log that we're starting to read the file
    logging.info(f"Reading file: {file_path}")
    
    try:
        # Attempt to read the CSV file
        df = pd.read_csv(file_path)
        
        # Log success and number of rows
        logging.info(f"Successfully read file with {len(df)} rows and {(df.PatientID.nunique())} unique PatientIDs")
        
        # Return the dataframe
        return df
    
    except Exception as e:
        # If any error occurs, log it and return None
        logging.error(f"Error processing file: {e}")
        return None
    
if __name__ == "__main__":
    file_path = "data/Enhanced_AdvUrothelial.csv"
    processed_df = process_enhanced_adv(file_path)
    
    if processed_df is not None:
        print("\nFirst few rows of the data:")
        print(processed_df.head())