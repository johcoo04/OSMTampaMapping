import logging
import os

# Ensure the logs directory exists
if not os.path.exists('logs'):
    os.makedirs('logs')

def setup_logging(log_file, logger_name='root'):
    handler = logging.FileHandler(os.path.join('logs', log_file), mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return logger