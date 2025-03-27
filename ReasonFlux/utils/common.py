import uuid
import logging
import colorama
from colorama import Fore, Style

# Initialize colorama
colorama.init(autoreset=True)

class ColorFormatter(logging.Formatter):
    """Custom formatter to set colors and bold text for different log levels"""
    COLOR_CODES = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN + Style.BRIGHT,
        logging.WARNING: Fore.YELLOW + Style.BRIGHT,
        logging.ERROR: Fore.RED + Style.BRIGHT,
        logging.CRITICAL: Fore.MAGENTA + Style.BRIGHT
    }

    def format(self, record):
        color_code = self.COLOR_CODES.get(record.levelno, "")
        return color_code + super().format(record) + Style.RESET_ALL
def get_uuid():
    return str(uuid.uuid4())

def get_logger(name):
    """Get a logger with colored and bold text output"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)  # Set the log level

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    formatter = ColorFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    return logger

logger = get_logger("ReasonFlux")

if __name__ == "__main__":
    logger.info("Hello, World!")
    logger.debug("This is a debug message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")
    logger.info("Hello, World!")
    logger.debug("This is a debug message.")
    logger.warning("This is a warning message.")