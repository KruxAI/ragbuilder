import os
import logging
from datetime import datetime
import sys
import re

class CustomFormatter(logging.Formatter):
    """
    Custom logging formatter to include filename and function name in log messages.
    """
    def format(self, record):
        format_str = '[%(levelname)s] %(asctime)s - %(filename)s - %(funcName)s - %(message)s'
        self._style._fmt = format_str
        return super().format(record)

class ExcludeFilter(logging.Filter):
    """
    Custom logging filter to exclude log messages that match certain patterns.
    """
    def filter(self, record):
        exclude_patterns = ["GET /get_log_updates", "common.py - flush"]
        return not any(re.search(pattern, record.getMessage()) for pattern in exclude_patterns)

def set_params_helper_by_src(src, **kwargs):
    try:
        kwargs['source'] = kwargs['loader_kwargs'][src]['source']
        kwargs['input_path'] = kwargs['loader_kwargs'][src]['input_path']
        kwargs['chunk_strategy'] = kwargs['chunking_kwargs'][src].get('chunk_strategy', None)
        kwargs['chunk_size'] = kwargs['chunking_kwargs'][src].get('chunk_size', None)
        kwargs['chunk_overlap'] = kwargs['chunking_kwargs'][src].get('chunk_overlap', None)
        kwargs['breakpoint_threshold_type'] = kwargs['chunking_kwargs'][src].get('breakpoint_threshold_type', 'percentile')
        kwargs['embedding_model'] = kwargs['embedding_kwargs'][src][0]['embedding_model']
        kwargs['vectorDB'] = kwargs['vectorDB_kwargs'][src]['vectorDB']
    except KeyError as e:
        logging.error(f"Key error: {e}")
    return kwargs

def setup_logging():
    logger = logging.getLogger('ragbuilder')
    if not os.path.exists('logs'):
        os.makedirs('logs')

    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # Create a custom formatter to define the log format
        formatter = CustomFormatter()

        # Create a file handler to write logs to a file
        log_filename = f"logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        # Create a stream handler to print logs to the console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # You can set the desired log level for console output
        console_handler.setFormatter(formatter)

        # Create and add the exclude filter to both handlers
        exclude_filter = ExcludeFilter()
        file_handler.addFilter(exclude_filter)
        console_handler.addFilter(exclude_filter)

        # Add the handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        # Redirect stdout and stderr to the logger
        sys.stdout = LoggerWriter(logger, logging.INFO)
        sys.stderr = LoggerWriter(logger, logging.ERROR)

        print(log_filename)
        return log_filename

class LoggerWriter:
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self._buffer = ''
        self._is_logging = False  # Flag to avoid recursion

    def write(self, message):
        if self._is_logging:
            return

        self._buffer += message
        while '\n' in self._buffer:
            line, self._buffer = self._buffer.split('\n', 1)
            if line.rstrip():  # Avoid extra blank lines
                # Ignore lines with "GET /get_log_updates" or "common.py - flush -"
                if re.search(r"GET /get_log_updates|common.py - flush -", line):
                    continue

                self._is_logging = True
                try:
                    # Log HTTP requests at INFO level
                    if re.search(r'GET|POST', line):
                        self.logger.log(logging.INFO, line.rstrip())
                    else:
                        self.logger.log(self.level, line.rstrip())
                finally:
                    self._is_logging = False

    def flush(self):
        if self._is_logging:
            return

        if self._buffer:
            # Ignore lines with "GET /get_log_updates" or "common.py - flush -"
            if not re.search(r"GET /get_log_updates|common.py - flush -", self._buffer):
                self._is_logging = True
                try:
                    self.logger.log(self.level, self._buffer.rstrip())
                finally:
                    self._is_logging = False
            self._buffer = ''
    
    def isatty(self):
        return False



def codeGen(code_string,return_code,output_var):
    globals_dict = {}
    locals_dict = {}
    try:
        if not return_code:
            exec(code_string,globals_dict,locals_dict)
            return locals_dict[output_var]
        else:
            return code_string
    except Exception as e:
        return e