import logging
import pytz
from datetime import datetime
import os

# Ensure the logs directory exists
if not os.path.exists('logs'):
    os.makedirs('logs')

# Configure logging with Eastern Standard Time
class ESTFormatter(logging.Formatter):
    def converter(self, timestamp):
        dt = datetime.fromtimestamp(timestamp, tz=pytz.timezone('US/Eastern'))
        return dt.timetuple()

    def formatTime(self, record, datefmt=None):
        record.created = self.converter(record.created)
        return super().formatTime(record, datefmt)

formatter = ESTFormatter('%(asctime)s - %(levelname)s - %(message)s')

def setup_logging(log_file, logger_name='root'):
    handler = logging.FileHandler(os.path.join('logs', log_file), mode='w')
    handler.setFormatter(formatter)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return logger