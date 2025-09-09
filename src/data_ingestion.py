# src/data_ingestion.py
import os
import zipfile
from pathlib import Path

# --- Configuration (we will import this properly later, but define it here for now) ---
ROOT_DIR = Path(__file__).parent.parent
KAGGLE_JSON_PATH = ROOT_DIR / "kaggle.json"
DATA_DIR = ROOT_DIR / "data"
KAGGLE_DATASET_ID = "shubhamgoel27/dermnet"
KAGGLE_DOWNLOAD_PATH = DATA_DIR / "dermnet-download"
EXTRACTED_DATA_PATH = DATA_DIR / "dermnet"



# Before importing the kaggle library, we set the environment variable
# to point to the directory containing our kaggle.json file.
os.environ['KAGGLE_CONFIG_DIR'] = str(ROOT_DIR)

# Now, we can safely import the Kaggle API
from kaggle.api.kaggle_api_extended import KaggleApi

def download_and_extract_data(dataset_id: str, download_path: Path, extract_path: Path):
    """
    Downloads a dataset from Kaggle using the API and extracts it.

    This function handles authentication, downloading, and unzipping, ensuring
    the data is ready for the preprocessing pipeline.

    Args:
        dataset_id (str): The 'username/dataset-name' string from Kaggle.
        download_path (Path): The directory to save the downloaded zip file.
        extract_path (Path): The directory where the contents will be extracted.
    """
    # --- CRITICAL: Set up Kaggle API credentials ---
    # This is the new, important part.
    if not KAGGLE_JSON_PATH.exists():
        raise FileNotFoundError(
            f"Kaggle API credentials not found at {KAGGLE_JSON_PATH}. "
            "Please download your kaggle.json from your Kaggle account and place it in the project's root directory."
        )
    
    print("--- Starting Data Ingestion ---")

    # Ensure the download and extraction directories exist
    download_path.mkdir(parents=True, exist_ok=True)
    extract_path.mkdir(parents=True, exist_ok=True)

    # Check if data is already extracted
    if any(extract_path.iterdir()):
        print(f"Data already extracted in '{extract_path}'. Skipping download and extraction.")
        return

    print("Authenticating with Kaggle API...")
    api = KaggleApi()
    api.authenticate()

    print(f"Downloading dataset '{dataset_id}' to '{download_path}'...")
    api.dataset_download_files(
        dataset=dataset_id,
        path=download_path,
        unzip=False
    )
    print("Download complete.")

    zip_files = list(download_path.glob('*.zip'))
    if not zip_files:
        raise FileNotFoundError("Error: No zip file found after download.")
    
    zip_path = zip_files[0]

    print(f"Extracting '{zip_path.name}' to '{extract_path}'...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extraction complete.")

    os.remove(zip_path)
    print(f"Cleaned up zip file: '{zip_path.name}'.")

    print("--- Data Ingestion Finished ---")

if __name__ == '__main__':
    # This block allows you to run this script directly for testing
    download_and_extract_data(
        dataset_id=KAGGLE_DATASET_ID,
        download_path=KAGGLE_DOWNLOAD_PATH,
        extract_path=EXTRACTED_DATA_PATH
    )