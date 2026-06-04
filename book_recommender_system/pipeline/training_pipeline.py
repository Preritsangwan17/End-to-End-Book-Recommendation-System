import sys

from book_recommender_system.logger.log import logging
from book_recommender_system.exception.exception_handler import AppException

from book_recommender_system.components.stage_00_data_ingestion import DataIngestion


class TrainingPipeline:
    """
    Corrected Training Pipeline class.
    This orchestrates the different stages of the book recommendation system.
    """

    def __init__(self):
        self.data_ingestion = DataIngestion()
        logging.info("TrainingPipeline initialized.")

    def start_data_ingestion(self):
        """
        Run the data ingestion stage.
        (Corrected - we let initiate_data_ingestion handle download + fallback internally
         to avoid duplicate download attempts.)
        """
        try:
            logging.info("Starting data ingestion stage...")
            ingested_dir = self.data_ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion stage finished. Data available at: {ingested_dir}")
            return ingested_dir
        except Exception as e:
            raise AppException(e, sys) from e

    def start_training_pipeline(self):
        """
        Main entry point for the full training pipeline.
        Currently only runs data ingestion (other stages are still empty placeholders).
        """
        try:
            logging.info(f"{'=' * 30} Starting Training Pipeline {'=' * 30}")

            self.start_data_ingestion()

            # TODO: Add calls to other stages when they are implemented:
            # self.start_data_validation()
            # self.start_data_transformation()
            # self.start_model_trainer()

            logging.info(f"{'=' * 30} Training Pipeline Completed (data ingestion only for now) {'=' * 30}")
            return True

        except Exception as e:
            raise AppException(e, sys) from e

    # Support the original typo in main.py for backward compatibility
    def start_traning_pipeline(self):
        """Alias to support the misspelled call in main.py"""
        logging.info("Note: 'start_traning_pipeline' is a typo. Using 'start_training_pipeline' instead.")
        return self.start_training_pipeline()
