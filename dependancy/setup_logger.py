import logging
import os
from datetime import datetime

def create_logger(args, log_dir:str, specific_experiment_name:str=None):
    if args.logger:
        if specific_experiment_name is None:
            return setup_logger(name=f"{args.experiment_name}", log_dir=log_dir)
        else:
            return setup_logger(name=f"{specific_experiment_name}", log_dir=log_dir)
    else:
        return None

def log_and_print(args, logger:logging.Logger, message:str, type_of_message:str="info"):
    if args.verbose:
        if args.logger:
            if type_of_message == "info":
                logger.info(message)
            elif type_of_message == "error":
                logger.error(message)
            elif type_of_message == "debug":
                logger.debug(message)
            elif type_of_message == "warning":
                logger.warning(message)
            elif type_of_message == "critical" or type_of_message == "fatal":
                logger.critical(message)
            elif type_of_message == "exception":
                logger.exception(message)
            else:
                raise ValueError(f"Invalid type of message: {type_of_message}")
        else:
            if type_of_message == "info":
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO] {message}")
            elif type_of_message == "error":
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ERROR] {message}")
            elif type_of_message == "debug":
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - DEBUG] {message}")
            elif type_of_message == "warning":
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - WARNING] {message}")
            elif type_of_message == "critical" or type_of_message == "fatal":
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - CRITICAL] {message}")
            elif type_of_message == "exception":
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - EXCEPTION] {message}")
            else:
                raise ValueError(f"Invalid type of message: {type_of_message}")

def setup_logger(name:str=None, stream:bool=True, log_dir:str=None):
    """Set up logger configuration"""
    # Create logs directory in the parent directory of utils
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)
    else:
        pass
    
    
    # Create timestamp for unique log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if name is None:
        log_file = os.path.join(log_dir, f'log_{timestamp}.log')
    else:
        log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
    
    # Configure logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    if stream:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    if stream:
        logger.addHandler(console_handler)
    
    return logger
