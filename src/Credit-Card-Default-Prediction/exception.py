import sys
import logging
from src.logger import logging as custom_logging  # Renaming to avoid conflicts with built-in logging

def error_msg_details(error, error_detail: sys):
    """
    Generate a detailed error message with file name, line number, and error message.

    Args:
        error (Exception): The caught exception.
        error_detail (sys): The sys module to extract error details.

    Returns:
        str: A formatted string with detailed error information.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_msg = (f"Error occurred in script [{file_name}] "
                 f"at line number [{exc_tb.tb_lineno}] with error message [{str(error)}]")
    return error_msg

class CustomException(Exception):
    def __init__(self, error_msg, error_detail: sys):
        super().__init__(error_msg)
        self.error_msg = error_msg_details(error_msg, error_detail)

    def __str__(self):
        return self.error_msg

#if __name__ == "__main__":
 #   try:
#        a = 1 / 0
#    except Exception as e:
#        custom_logging.error("An exception occurred", exc_info=True)  # Log the exception with traceback
#        raise CustomException(e, sys)
