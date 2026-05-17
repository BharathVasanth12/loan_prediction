import os
import shutil
from src.config import DATASET_NAME, RAW_PATH
from src.logger import logging, log_section
import kagglehub

# Download latest version
class DataIngestion:
    def __init__(self, DATASET_NAME):
        self.dataset_name = DATASET_NAME
    
    def download_dataset(self):
        log_section("Step: download_dataset (Kaggle)")
        path = kagglehub.dataset_download(self.dataset_name)

        logging.info("Path to dataset files: %s", path)

        for dataset in os.listdir(path):
            src = os.path.join(path, dataset)
            dest = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', RAW_PATH )
            os.makedirs(dest, exist_ok=True)
            shutil.copy2(src, dest)
            logging.info("Copied dataset file: %s", dataset)

    def run(self):
        logging.info("Starting data ingestion process...")
        self.download_dataset()
        logging.info("Data ingestion process completed successfully.")

if __name__ == "__main__":
    data_ingestion = DataIngestion(DATASET_NAME)
    data_ingestion.run()