from data.database import TableModel, connect_db
from sqlalchemy import Column, Integer, String
import logging
import os
import sys
from datetime import datetime
from utils.config import config


# 定义日志模型
class LogEntry(TableModel):
    level = Column(String)
    message = Column(String)


# 自定义日志处理器
class DatabaseLogHandler(logging.Handler):
    def emit(self, record):
        LogEntry.create(
            level=record.levelname,
            message=self.format(record)
        )


# 配置日志记录
def get_log():
    path = os.path.join(os.path.dirname(__file__), '..', 'output')
    if not os.path.exists(path):
        os.mkdir(path)

    level = logging.INFO if config['log_level'] == 'info' else logging.DEBUG
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=os.path.join(os.path.dirname(__file__), '..', 'output', 'server.log')
    )

    # 添加自定义的日志处理器
    db_handler = DatabaseLogHandler()
    db_handler.setLevel(level)
    db_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(db_handler)

    return logging.getLogger()


log = get_log()

    # Create a custom handler that writes to stderr
class StderrHandler(logging.StreamHandler):
    def __init__(self):
        super().__init__(sys.stderr)

    def format(self, record):
        # Customize the log format if needed
        return f"{record.levelname}: {record.getMessage()}"
def get_log2():
    # Configure logging
    logger = logging.getLogger(__name__)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    # Add the custom handler
    handler = StderrHandler()
    logger.addHandler(handler)
    logger.propagate = False

    return logger

log2 = get_log2()

if __name__ == '__main__':
    connect_db()

    log.info('test')

    for _log in LogEntry.query_all():
        level = _log.level
        message = _log.message
        timestamp = _log.create_time
        print(f'level: {level}, message: {message}, timestamp: {timestamp}')
