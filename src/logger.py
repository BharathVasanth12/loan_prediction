import logging
import os
from datetime import datetime

LOG_DIR = "logs"

os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE_NAME = f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

LOG_FILE = os.path.join(LOG_DIR, LOG_FILE_NAME)

logging.basicConfig(
    level=logging.INFO,
    # format="%(asctime)s - %(levelname)s - %(message)s - %(filename)s:%(lineno)d",
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

logging = logging.getLogger('loan_prediction')


def log_section(title: str, char: str = "-", width: int = 60) -> None:
    """Visual banner around a single logical step.

    Use to mark the start of any non-trivial method so the log reads like
    a guided tour rather than a wall of text.
    """
    logging.info(char * width)
    logging.info(f"  >> {title}")
    logging.info(char * width)
