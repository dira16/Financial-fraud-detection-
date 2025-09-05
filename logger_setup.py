import logging

def setup_logger(name='fraud_logger', log_file='fraud_detection.log', level=logging.INFO):
    """Setup shared logger for all modules."""
    formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s', '%H:%M:%S')

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger