#Creating a custome logger
import logging
import os
from datetime import datetime
from pathlib import Path
# Use /tmp for Lambda

raw_main_path = r"C:\Users\Thimira\Desktop\Data\ML & AI\Deep Learning\LangChain\Medical-Assistant"
main_path = Path(raw_main_path)

# Logs folder inside project
log_dir = main_path / "logs"
os.makedirs(log_dir, exist_ok=True)

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(log_dir, LOG_FILE)

logging.basicConfig(
    filename=logs_path,
    format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG
)