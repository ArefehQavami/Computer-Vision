import logging
import logstash


class ElasticsearchLogger:
    """
        A logger that sends log entries to Elasticsearch.

        Args:
            host (str): The Elasticsearch host.
            port (int): The Elasticsearch port.
            index_name (str): The name of the Elasticsearch index to log to.
            log_level (int, optional): The minimum log level to log. Defaults to logging.DEBUG.
    """

    def __init__(self, config, index_name, log_level=logging.DEBUG, name='python-logstash-logger'):

        self.logger_elk = logging.getLogger(name)
        self.logger_elk.setLevel(log_level)

        self.config = config

        # self.logger_elk.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s - %(asctime)s -')
        self.logger_elk.addHandler(logstash.TCPLogstashHandler(self.config['elastic_host'], self.config['elastic_port'], version=1))

        self.index_name = index_name

    def log(self, level, message):
        """
            Logs a message with the specified log level.

            Args:
                level (str): The log level, e.g. 'debug', 'info', 'warning', 'error', 'critical'.
                message (str): The log message.
        """

        if level == 'debug':
            self.logger_elk.debug(message)
        elif level == 'info':
            self.logger_elk.info(message)
        elif level == 'warning':
            self.logger_elk.warning(message)
        elif level == 'error':
            self.logger_elk.error(message)
        elif level == 'critical':
            self.logger_elk.critical(message)


