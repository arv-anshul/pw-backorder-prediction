""" Basic logging definition for this project. """

import logging
from datetime import date
from pathlib import Path

LOG_DIR = Path('logs')
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE_NAME = LOG_DIR / (date.today().strftime('%d-%m-%Y') + '.log')

logging.basicConfig(
    filename=LOG_FILE_NAME,
    format="[ %(asctime)s ] %(filename)s:[%(lineno)d] %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
