import os
import sys
import shutil
import zipfile
import urllib.request

from book_recommender_system.logger.log import logging
from book_recommender_system.exception.exception_handler import AppException
from book_recommender_system.config.configuration import AppConfiguration


class DataIngestion:
    def __init__(self, app_config=AppConfiguration()):
        """
        DataIngestion Initialization
        """
        try:
            logging.info(f"{'='*20}Data Ingestion Log started.{'='*20} ")
            self.data_ingestion_config = app_config.get_data_ingestion_config()
        except Exception as e:
            raise AppException(e, sys) from e

    def download_data(self):
        """Download the zip file from the configured URL."""
        try:
            os.makedirs(self.data_ingestion_config.raw_data_dir, exist_ok=True)

            download_url = self.data_ingestion_config.dataset_download_url
            zip_file_path = os.path.join(
                self.data_ingestion_config.raw_data_dir, "books_data.zip"
            )

            logging.info(f"Downloading data from {download_url} ...")
            urllib.request.urlretrieve(download_url, zip_file_path)
            logging.info(f"Data downloaded successfully at {zip_file_path}")
            return zip_file_path
        except Exception as e:
            logging.info("Download failed. Will try local fallback.")
            raise AppException(e, sys) from e

    def extract_zip_file(self, zip_file_path: str):
        """Extract the downloaded zip."""
        try:
            os.makedirs(self.data_ingestion_config.ingested_dir, exist_ok=True)

            with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                zip_ref.extractall(self.data_ingestion_config.ingested_dir)

            logging.info(f"Extracted data into {self.data_ingestion_config.ingested_dir}")
        except Exception as e:
            raise AppException(e, sys) from e

    def _use_local_notebook_data(self):
        """Fallback: copy CSVs from notebook/ if download fails or zip not present."""
        try:
            notebook_dir = os.path.join(os.getcwd(), "notebook")
            os.makedirs(self.data_ingestion_config.ingested_dir, exist_ok=True)

            for fname in ["BX-Books.csv", "BX-Book-Ratings.csv", "BX-Users.csv"]:
                src = os.path.join(notebook_dir, fname)
                dst = os.path.join(self.data_ingestion_config.ingested_dir, fname)
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                    logging.info(f"Using local notebook data: {fname}")
                else:
                    logging.info(f"Local file not found: {src}")

            logging.info("Fell back to local notebook CSVs.")
        except Exception as e:
            raise AppException(e, sys) from e

    def initiate_data_ingestion(self):
        """
        Main method to start data ingestion.
        Tries download + extract, falls back to local notebook data.
        """
        try:
            # Try download
            try:
                zip_path = self.download_data()
                self.extract_zip_file(zip_path)
            except Exception as ex:
                logging.info(f"Download/extract failed: {ex}")
                self._use_local_notebook_data()

            # Verify we have the required files
            ingested_dir = self.data_ingestion_config.ingested_dir
            required_files = ["BX-Books.csv", "BX-Book-Ratings.csv", "BX-Users.csv"]
            for f in required_files:
                full_path = os.path.join(ingested_dir, f)
                if not os.path.exists(full_path):
                    raise FileNotFoundError(f"Missing required file after ingestion: {full_path}")

            logging.info("Data ingestion completed successfully.")
            return self.data_ingestion_config.ingested_dir

        except Exception as e:
            raise AppException(e, sys) from e
