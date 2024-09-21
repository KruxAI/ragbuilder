import os
import logging
from datetime import datetime
import sys
import re

#TODO: Fetch level from command arguments to ragbuilder. For now, using this global variable
LOG_LEVEL = logging.INFO

class ProgressState:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.progress_info = {
                'current_run': 0,
                'total_runs': 1,
                'first_eval_complete': False,
                'synth_data_gen_in_progress': 0
            }
        return cls._instance

    def get_progress(self):
        return self.progress_info

    def toggle_synth_data_gen_progress(self, n):
        self.progress_info['synth_data_gen_in_progress'] = n

    def set_total_runs(self, n):
        self.progress_info['total_runs'] = n

    def increment_progress(self):
        self.progress_info['current_run'] += 1

    def set_first_eval_complete(self):
        self.progress_info['first_eval_complete'] = True

    def reset(self):
        self.progress_info['current_run'] = 0
        self.progress_info['total_runs'] = 1
        self.progress_info['first_eval_complete'] = False

progress_state = ProgressState()

class CustomFormatter(logging.Formatter):
    """
    Custom logging formatter to include filename and function name in log messages.
    """
    def format(self, record):
        # format_str = '[%(levelname)s] %(asctime)s - %(filename)s - %(funcName)s - %(message)s'
        format_str = '[%(levelname)s] %(asctime)s - %(filename)s - %(message)s'
        self._style._fmt = format_str
        self.datefmt='%Y-%m-%d %H:%M:%S'
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
        logger.setLevel(LOG_LEVEL)

        # Create a custom formatter to define the log format
        formatter = CustomFormatter()

        # Create a file handler to write logs to a file
        log_filename = f"logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(LOG_LEVEL)
        file_handler.setFormatter(formatter)

        # Create a stream handler to print logs to the console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(LOG_LEVEL)  # You can set the desired log level for console output
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
        #TODO: The stdout that we want to capture is currently being captured under stderr for some reason
        # This is because in the below line if we make it logging.ERROR, all the stdout stuff that we want
        # comes with the tag of [ERROR] instead of [INFO]. Or maybe there's a different issue? 
        # Anyhow, keeping even stderr as INFO logging in the below line for sanity. 
        sys.stderr = LoggerWriter(logger, logging.INFO) 

        return log_filename

class LoggerWriter:
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self._buffer = ''
        self._is_logging = False  # Flag to avoid recursion
        dt_time=datetime.now().strftime("%Y-%m-%d")
        self.skip_log_pattern=f"GET /get_log_updates|GET /progress|{dt_time}.*INFO|{dt_time}.*ERROR|{dt_time}.*DEBUG|^\\s*$"

    def write(self, message):
        if self._is_logging:
            return

        self._buffer += message
        while '\n' in self._buffer:
            line, self._buffer = self._buffer.split('\n', 1)
            if line.rstrip():  # Avoid extra blank lines
                # Ignore lines with "GET /get_log_updates" or "common.py - flush -"
                if re.search(self.skip_log_pattern, line):
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
            if not re.search(self.skip_log_pattern, self._buffer):
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