from collections import namedtuple

DataIngestionConfig = namedtuple(
    "DataIngestionConfig", ["dataset_download_url", "raw_data_dir", "ingested_dir"]
)

DataValidationConfig = namedtuple(
    "DataValidationConfig",
    ["cleaned_data_dir", "serialized_data_dir", "Book_csv_file", "rating_csv_file"],
)

DataTransformationConfig = namedtuple(
    "DataTransformationConfig", ["transformed_data_dir", "clean_data_dir"]
)
ModelTrainingConfig = namedtuple(
    "ModelTrainingConfig",
    ["trained_model_dir", "trained_model_name", "transformed_data_file_path"],
)
