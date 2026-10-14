"""Use local Book-Crossing data first, with an explicit download fallback."""

from pathlib import Path
import shutil
import urllib.request
import zipfile
from book_recommender_system.config.configuration import AppConfiguration
from book_recommender_system.constant import ROOT_DIR

REQUIRED_FILES = ("BX-Books.csv", "BX-Book-Ratings.csv", "BX-Users.csv")


class DataIngestion:
    def __init__(self, app_config=None):
        self.data_ingestion_config = (
            app_config or AppConfiguration()
        ).get_data_ingestion_config()

    def extract_zip_file(self, archive):
        destination = Path(self.data_ingestion_config.ingested_dir).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as handle:
            for member in handle.infolist():
                if (
                    not (destination / member.filename)
                    .resolve()
                    .is_relative_to(destination)
                ):
                    raise ValueError("Unsafe path in dataset archive")
            handle.extractall(destination)

    def initiate_data_ingestion(self):
        destination = Path(self.data_ingestion_config.ingested_dir)
        destination.mkdir(parents=True, exist_ok=True)
        source = ROOT_DIR / "notebook"
        if all((source / name).is_file() for name in REQUIRED_FILES):
            for name in REQUIRED_FILES:
                shutil.copy2(source / name, destination / name)
        elif (ROOT_DIR / "books_data.zip").is_file():
            self.extract_zip_file(ROOT_DIR / "books_data.zip")
        elif not all((destination / name).is_file() for name in REQUIRED_FILES):
            archive = Path(self.data_ingestion_config.raw_data_dir) / "books_data.zip"
            archive.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(
                self.data_ingestion_config.dataset_download_url, archive
            )
            self.extract_zip_file(archive)
        for name in REQUIRED_FILES:
            if not (destination / name).is_file():
                raise FileNotFoundError(
                    f"Missing Book-Crossing CSV: {destination / name}"
                )
        return str(destination)
