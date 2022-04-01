import asyncio
import os
import pickle
import queue
import ssl
import time
from logging import log, ERROR

import websockets as websockets
from websockets.exceptions import ConnectionClosedError
from websockets.legacy.client import WebSocketClientProtocol

from configs.api import video_token, WS_PORT_VIDEO

PING_TIMEOUT = 10
PING_INTERVAL = 5
CLOSE_TIMEOUT = 20


class VideoStreamerThreadBody:
    """
    Thread body to use to interact with Frames-gather Server-Side WSS
    """
    def __init__(self, serv_addr='localhost', host_id='host_name', ca_file='ca_cert.pem'):
        """
        :param serv_addr:
        :param host_id: Hostname used to be identified from Server-Side
        :param ca_file: Full-chain certificate to believe in
        """
        self.q_frames = queue.Queue(maxsize=30)
        self.uri = f"wss://{serv_addr}:{WS_PORT_VIDEO}"
        self.ca_file = ca_file
        self.process = True
        self.my_name = host_id

    def __call__(self):
        """
        Setup WSS client-side and define the client-side protocol

        :return:
        """
        if not bool(os.getenv("DEBUG")):
            return

        async def send_frames():
            uri = self.uri
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ca_cert_pem = self.ca_file  

            print(f'FILE = {ca_cert_pem}')
            ssl_context.load_verify_locations(cafile=ca_cert_pem)

            # TODO: Remove in production
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            while self.process:
                flooding = True
                try:
                    life_start = time.time()
                    await asyncio.sleep(5)
                    async with websockets.connect(uri,
                                                  ssl=ssl_context,
                                                  ping_timeout=PING_TIMEOUT,
                                                  ping_interval=PING_INTERVAL,
                                                  close_timeout=CLOSE_TIMEOUT
                                                  ) as ws:
                        ws: WebSocketClientProtocol
                        # ws.ping()
                        print('Connection Opened')
                        # ws.connection_lost()
                        ack = None
                        read_ack = 0
                        empty_cnt = 0
                        msg = video_token
                        try:
                            await ws.send(msg)
                        except:
                            flooding = False
                            print("Unauthorized to send video")
                        while flooding and ws.open:
                            try:

                                if ack is not None and not ack.cr_running:
                                    r = await ack
                                    ack = None
                                    if r != 'OK':
                                        raise Exception('Server Connection Lost (in time and space)')

                                if ack is not None and ack.cr_running:
                                    print('RUNNING')

                                if self.q_frames.empty():
                                    # time.sleep(1/10)
                                    empty_cnt += 1
                                    await ws.ping()
                                    await asyncio.sleep(1 / 20)
                                    continue

                                if empty_cnt > 0:
                                    # print(f'Skip count: {empty_cnt}')
                                    empty_cnt = 0

                                read_ack = (read_ack + 1) % 100

                                f = self.q_frames.get_nowait()
                                msg = pickle.dumps((self.my_name, f))
                                # await ping
                                await ws.send(msg)
                                # print("sent msg")

                                if read_ack == 0:
                                    ack = ws.recv()

                                # p = await ws.pong('aaa')
                            except Exception as e:
                                log(level=ERROR, msg=f'VideoStreamer: {e}')
                                flooding = False
                                await ws.close_connection()
                        print('Connection Closed')

                    life_end = time.time()
                    life_time = int(life_end - life_start)
                    print(f'# Lifetime: {life_time}s')
                except Exception as e:
                    # except ConnectionRefusedError as e:
                    log(ERROR, f'[WS-Sender]: {e}')

        evt_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(evt_loop)

        asyncio.get_event_loop().run_until_complete(send_frames())

        # while self.process:
        #     asyncio.get_event_loop().run_until_complete(send_frames())
