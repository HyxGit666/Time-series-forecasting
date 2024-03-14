import asyncio
from asyncio.windows_events import ProactorEventLoop

from fastapi import FastAPI
from uvicorn import Config, Server

import configparser
# Access the parameters
config = configparser.ConfigParser()
config.read('deploy.ini')
HOST = config.get('FastAPI', 'host')
PORT = config.getint('FastAPI', 'port')

# Ref Link: https://github.com/tiangolo/fastapi/issues/4361
# customize a proactor server to implement async long running activities with event handler
class ProactorServer(Server):
    def run(self, sockets=None):
        loop = ProactorEventLoop()
        asyncio.set_event_loop(loop)
        asyncio.run(self.serve(sockets=sockets))

# TODO not really reloading
if __name__ == '__main__':
    config = Config(app="main:app", host=HOST, port=PORT,reload=True,log_level="info")
    server = ProactorServer(config=config)
    server.run()