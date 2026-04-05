import logging
import sys
import os
from datetime import datetime

def get_logger(name='NanoGPT', log_dir='logs', master_process: bool = False):
    level = logging.INFO if master_process else logging.ERROR
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s (%(name)s): %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        if master_process:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_filename = f'train_{timestamp}.log'
            log_path = os.path.join(log_dir, log_filename)
            
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
    return logger