from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_FOLDER_NAME = "config"
CONFIG_FILE_NAME = "config.yaml"
CONFIG_FILE_PATH = str(ROOT_DIR / CONFIG_FOLDER_NAME / CONFIG_FILE_NAME)
