SUPPORTED_ENTITIES: list[str] = ["PERSON", "EMAIL", "PHONE", "ADDRESS", "ID"]
SUPPORTED_OPERATORS: list[str] = ["replace", "mask", "pseudonymize"]
PIPELINE_NAME: str = "hybrid"
MODEL_PATH: str = "models/final_model"

MAX_TEXT_LENGTH: int = 200_000
MAX_FILE_SIZE_BYTES: int = 2 * 1024 * 1024       
MAX_ARCHIVE_SIZE_BYTES: int = 10 * 1024 * 1024
MAX_FILES: int = 10
