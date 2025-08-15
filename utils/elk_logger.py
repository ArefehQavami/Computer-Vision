

import logging
import logstash

py_logger = logging.getLogger()
host = "10.187.160.104"
port = 5011

class Log():

    def __init__(self):

        logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(levelno)s -- %(message)s-- %(asctime)s--%(pathname)s')
        # py_logger.addHandler(logstash.TCPLogstashHandler(host, port, version=1))


    def warning_level_log(self, warn_msg):
        py_logger.warning(warn_msg)

    def info_level_log(self, info_msg):
        py_logger.info(info_msg)

    def error_level_log(self, err_msg):
        py_logger.error(err_msg)

    def critical_level_log(self, crit_msg):
        py_logger.critical(crit_msg)
