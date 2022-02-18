import os
import logging
import multiprocessing
import threading
import socket
import queue
import sys
import traceback
import time

original_print = print


logging_color_set = {
    "underline_grey":  "\033[4m",
    "grey": "\x1b[38;21m",
    "yellow":  "\x1b[33;1m",
    "red": "\x1b[31;1m",
    "green": "\033[32m",
    "reset":  "\x1b[0m",
}


class MultiProcessingHandler(logging.Handler):
    def __init__(self, name, sub_handler=None):
        """Multiprocessing logger handler. Code from:
        https://github.com/jruere/multiprocessing-logging/blob/master/multiprocessing_logging.py

        Args:
            name (str): Logger name. The same logger name refers to the same logger instance.
            sub_handler (logging.Handler, optional): The additional sub_handler to emit records. Defaults to None.
        """
        super(MultiProcessingHandler, self).__init__()

        if sub_handler is None:
            sub_handler = logging.StreamHandler()
        self.sub_handler = sub_handler

        self.setLevel(self.sub_handler.level)
        self.setFormatter(self.sub_handler.formatter)
        self.filters = self.sub_handler.filters

        self.queue = multiprocessing.Queue(-1)
        self._is_closed = False
        # The thread handles receiving records asynchronously.
        self._receive_thread = threading.Thread(target=self._receive, name=name)
        self._receive_thread.daemon = True
        self._receive_thread.start()


    def setFormatter(self, fmt):
        super(MultiProcessingHandler, self).setFormatter(fmt)
        self.sub_handler.setFormatter(fmt)


    def _receive(self):
        try:
            broken_pipe_error = BrokenPipeError
        except NameError:
            broken_pipe_error = socket.error
        
        while True:
            try:
                if self._is_closed and self.queue.empty():
                    break

                record = self.queue.get(timeout=0.2)
                self.sub_handler.emit(record)
            except (KeyboardInterrupt, SystemExit):
                raise
            except (broken_pipe_error, EOFError):
                break
            except queue.Empty:
                pass  # This periodically checks if the logger is closed.
            except:
                traceback.print_exc(file=sys.stderr)

        self.queue.close()
        self.queue.join_thread()


    def _send(self, s):
        self.queue.put_nowait(s)


    def _format_record(self, record):
        # ensure that exc_info and args
        # have been stringified. Removes any chance of
        # unpickleable things inside and possibly reduces
        # message size sent over the pipe.
        if record.args:
            record.msg = record.msg % record.args
            record.args = None
        if record.exc_info:
            self.format(record)
            record.exc_info = None

        return record


    def emit(self, record):
        try:
            s = self._format_record(record)
            self._send(s)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


    def close(self):
        if not self._is_closed:
            self._is_closed = True
            self._receive_thread.join(5.0)  # Waits for receive queue to empty.

            self.sub_handler.close()
            super(MultiProcessingHandler, self).close()


class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors
    """
    # format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    format_ = "%(asctime)s | %(message)s"

    FORMATS = {
        logging.DEBUG: logging_color_set["underline_grey"] + format_ + logging_color_set["reset"],
        logging.INFO: logging_color_set["grey"] + format_ + logging_color_set["reset"],
        logging.WARNING: logging_color_set["yellow"] + format_ + logging_color_set["reset"],
        logging.ERROR: logging_color_set["red"] + format_ + logging_color_set["reset"],
        logging.CRITICAL: logging_color_set["red"] + format_ + logging_color_set["reset"]
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)


def get_logger(logger_name='', console=True, log_path=None, level="DEBUG", 
              multi_processing=False, propagate=False, file_mode='w'):
    """A logger wrapper function.

    Args:
        logger_name (str, optional): 
            The name for the logger. loggers with the same name all point to the same logger instance. Defaults to ''.
        console (bool, optional): 
            If True, redirect the logging message to the stdout. Defaults to True.
        log_path (str, optional): 
            If set, redirect the logging message to the log file. Defaults to None.
        level (str, optional): 
            Logging level. Defaults to "DEBUG".
        multi_processing (bool, optional):
            Whether to use multi-processing logger. Defaults to False.
        propagate (bool, optional): 
            Setting False to prevent duplicate logs in third-party framesworks that hack root loggers. 
            Defaults to False.
        file_mode (str, optional): 
            Logger file mode.. Defaults to 'w'.
    """
    if console is False and log_path is None:
        raise ValueError(f"You cannot set `console` to False and `log_path` to None at the same time!")

    assert level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

    logger = logging.getLogger(logger_name)
    logger.propagate = propagate
    formatter = CustomFormatter()

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        if multi_processing:
            logger.addHandler(MultiProcessingHandler(logger_name, console_handler))
        else:
            logger.addHandler(console_handler)

    if log_path:
        filer_handler = logging.FileHandler(log_path, file_mode)
        # escape symbols for highlight may not display properly in log files, so fall back to basic formatter
        # here I keep the levelname keyword, so that IDEs like vscode can highlight them automatically
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] | %(message)s', datefmt='%a, %Y-%m-%d %H:%M:%S')
        filer_handler.setFormatter(formatter)
        if multi_processing:
            logger.addHandler(MultiProcessingHandler(logger_name, filer_handler))
        else:
            logger.addHandler(filer_handler)
    
    logger.setLevel(level=getattr(logging, level))

    return logger
