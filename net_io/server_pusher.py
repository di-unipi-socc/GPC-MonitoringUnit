import os
import pickle
import time
from socket import gethostname
from threading import Thread
import logging as log

import requests

from configs.api import ca_cert
from counts.mucounts import MUCounts, IN, OUT
from net_io.videostream_websoc import VideoStreamerThreadBody

DUMP_FILE = 'unsent_updates.bin'
UPDATE_TIME_LIMIT = 10
UPD_RATE = 1/10


class ServerPusherThreadBody:
    """
    Thread that perform all interactions with Collector Service.
    Send Counts Updates, and forward debug frames stream.
    """
    def __init__(self, server_addr, server_port, server_auth):
        self.update_time = UPDATE_TIME_LIMIT
        self.upd_timestamp = 0
        self.process = True

        self.server_addr = server_addr
        self.server_port = server_port
        self.server_auth = server_auth
        self.device_name = gethostname()
        self.ca_file = ca_cert

        self.updates = MUCounts(self.device_name)
        self.send_ls = []
        if os.path.exists(DUMP_FILE):
            self.__restore_file__(DUMP_FILE)

        self.streamer = None
        if bool(os.getenv("DEBUG")):
            self.streamer = VideoStreamerThreadBody(serv_addr=self.server_addr,
                                                    host_id=self.device_name,
                                                    ca_file=self.ca_file)

    def __call__(self):
        """
        Main thread-loop, that perform Update's send
        :return:
        """
        self.th_streamer = Thread(target=self.streamer)
        self.th_streamer.start()
        while self.process:
            try:
                t_start = time.time()
                if self.any_update() or self.may_send_update():
                    res = self.send_updates()
                    if not res:
                        time.sleep(5)
                t_send = time.time() - t_start
                tt_sleep = (UPD_RATE-t_send) if t_send <= UPD_RATE else 0
                time.sleep(tt_sleep)
            except Exception as e:
                log.error(f'[ServerPusherThreadBody] {str(e)}')

    def enqueue_img(self, image):
        """
        Add a new debug frame to frame-stream queue,

        :param image:
        :return:
        """
        if bool(os.getenv("DEBUG")) and self.streamer.q_frames.qsize() < self.streamer.q_frames.maxsize:
            try:
                self.streamer.q_frames.put_nowait(image)
            except:
                pass
            return

    def add_update(self, n_in, n_out):
        """
        Add new Enter/Exit Counts Estimation

        :param n_in:
        :param n_out:
        :return:
        """
        if n_in > 0:
            self.updates.add_record(IN, n_in)
        if n_out > 0:
            self.updates.add_record(OUT, n_out)

    def __dump_file__(self, file_path):
        """
        Save Counts Estimation in a temp.bin file, to be restored in case of unexpected MU's crash/reboot

        :param file_path:
        :return:
        """
        dump_ok = False
        while not dump_ok:
            try:
                if len(self.send_ls) > 0:
                    with open(file_path, 'wb') as file:
                        ls_to_storage = self.send_ls
                        pickle.dump(ls_to_storage, file)
                dump_ok = True
            except Exception as e:
                log.error(str(e))

    def __restore_file__(self, file_path):
        """
        Restore file with unsent Counts Estimation

        :param file_path:
        :return:
        """
        assert len(self.send_ls) == 0
        try:
            with open(file_path, 'rb') as file:
                unsent_ls = pickle.load(file)
                for j_str in unsent_ls:
                    self.updates.rollback(j_str)
        except Exception as e:
            log.info(f'Reloading updates File: {str(e)}')
            return False

        return True

    def __send_updates__(self, status_j):
        """
        Perform the REST-API Call to right endpoint, to send last Counts estimations performed

        :param status_j:
        :return: True if the update is sent, False otherwise
        """
        try:
            r = requests.post(url=f'https://{self.server_addr}:{self.server_port}/update',
                              verify=True, data=status_j, auth=self.server_auth, timeout=5)
            if r.status_code != 200:
                raise Exception(f'Invalid server response({str(r.status_code)}) for {r.url}: \n{str(r.text)}')
            return True
        except Exception as e:
            log.error(f'ServerPusher Fail send_updates: \n{str(e)}')
            return False

    def send_updates(self):
        """
        Function that try to send last Counts Estimation.
        If it fail, save unsent updates into a temporary file, and try to send it
        in a successive round

        :return:
        """
        status_j, n_records = self.updates.jsonify()

        sent_ok = self.__send_updates__(status_j)

        if (not sent_ok) and n_records > 0:
            self.send_ls.append(status_j)

        if sent_ok and len(self.send_ls) > 0:
            sent_ok = self.__retry_sends__()

        if not sent_ok:
            self.__dump_file__(DUMP_FILE)
            return False

        # self.updates.__clear__()
        if os.path.exists(DUMP_FILE):
            os.remove(DUMP_FILE)

        self.refresh_update_timestamp()
        return True

    def __retry_sends__(self):
        sent_ok = True
        while len(self.send_ls) > 0 and sent_ok:
            status_j = self.send_ls.pop(0)
            sent_ok = self.__send_updates__(status_j)
            if not sent_ok:
                self.send_ls.append(status_j)
        return sent_ok

    def any_update(self):
        if self.updates.entrances.qsize() + self.updates.exits.qsize() > 0:
            return True
        return False

    def refresh_update_timestamp(self):
        self.upd_timestamp = int(time.time())

    def may_send_update(self):
        t_now = int(time.time())
        if t_now - self.upd_timestamp < self.update_time:
            return False
        return True

