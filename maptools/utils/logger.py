import os
import time
from loguru import logger

def make_logger(folder, filename=None, level='DEBUG', mode='w', console=True, rotation="10 MB", compression="zip"):
    """
    Creates a logger with specified settings.

    Args:
    folder (str): Folder where the log file will be stored.
    filename (str, optional): Name of the log file. Defaults to a date-based name.
    level (str, optional): Logging level. Defaults to 'DEBUG'.
    mode (str, optional): File mode. Defaults to 'w'.
    console (bool, optional): If True, also log to console. Defaults to False.
    rotation (str/int, optional): Rotate the log file at a certain interval or file size. Defaults to "10 MB".
    compression (str, optional): Compression for rotated logs. Defaults to "zip".

    Returns:
    loguru.Logger: Configured logger object.
    """
    
    if not console:
        logger.remove()

    if filename is None:
        log_filename = "log"
    log_filename = f"{filename}_{time.strftime('%Y-%m-%d', time.localtime())}.log"
    log_path = os.path.join(folder, log_filename)

    try:
        logger.add(log_path, enqueue=True, backtrace=True, diagnose=True, 
                   level=level, mode=mode, 
                   rotation=rotation, compression=compression)
    except Exception as e:
        print(f"Error configuring logger: {e}")
        return None
    
    return logger
