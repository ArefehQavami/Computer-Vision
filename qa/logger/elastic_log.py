import logging
from elasticsearch import Elasticsearch


class ElasticsearchLogger:
    """
        A logger that sends log entries to Elasticsearch.

        Args:
            host (str): The Elasticsearch host.
            port (int): The Elasticsearch port.
            index_name (str): The name of the Elasticsearch index to log to.
            log_level (int, optional): The minimum log level to log. Defaults to logging.DEBUG.
    """

    def __init__(self, config, index_name, log_level=logging.DEBUG):

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        self.config = config

        es = Elasticsearch([{'host': self.config['elastic_host'], 'port': self.config['elastic_port'], 'scheme': self.config['elastic_scheme']}])

        self.index_name = index_name
        es_handler = ElasticsearchHandler(es, index_name)
        es_handler.setLevel(log_level)

    def log(self, level, message):
        """
            Logs a message with the specified log level.

            Args:
                level (str): The log level, e.g. 'debug', 'info', 'warning', 'error', 'critical'.
                message (str): The log message.
        """

        if level == 'debug':
            self.logger.debug(message)
        elif level == 'info':
            self.logger.info(message)
        elif level == 'warning':
            self.logger.warning(message)
        elif level == 'error':
            self.logger.error(message)
        elif level == 'critical':
            self.logger.critical(message)


class ElasticsearchHandler(logging.Handler):
    def __init__(self, es, index_name, datefmt='%Y-%m-%d %H:%M:%S'):
        super().__init__()
        self.index_name = index_name
        self.es = es
        self.f = ElasticsearchFormatter(datefmt=datefmt)

    def emit(self, record):
        try:
            log_entry = self.f.format(record)
            self.es.index(index=self.index_name, body=log_entry, timeout="60s")
        except Exception as e:
            logging.exception("Error while handling log record: %s", e)

class ElasticsearchFormatter(logging.Formatter):
    def format(self, record):
        return {'timestamp': self.formatTime(record), 'message': record.getMessage()}

