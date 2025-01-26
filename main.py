from utils.log import log
from server.server import start_server
from data.database import connect_db

if __name__ == '__main__':
    connect_db()
    log.info('needle_digital_assistant started.')
    start_server()
