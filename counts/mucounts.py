import json
import queue
import time

IN = 'enterances'
OUT = 'exits'

MAX_Q_LEN = 10000


class MUCounts:
    """
    Class object that store People Counts Estimation.
    Offer methods to add/retrieve Counts Estimated, and format a message update to send to Collector Service.
    """
    def __init__(self, device_id, json_data=None):
        self.entrances = queue.Queue(maxsize=MAX_Q_LEN)
        self.exits = queue.Queue(maxsize=MAX_Q_LEN)
        self.device_id = device_id
        if json_data is not None:
            self.load(json_data)

    def add_record(self, type_count, counts):
        if counts <= 0:
            return
        if type_count == IN:
            try:
                self.entrances.put_nowait((counts, int(time.time())))
            except:
                self.entrances.put((counts, int(time.time())))
        elif type_count == OUT:
            try:
                self.exits.put_nowait((counts, int(time.time())))
            except:
                self.exits.put((counts, int(time.time())))
        else:
            raise Exception(f'Invalid type of Counts: {type_count}')

    def load(self, j_data):
        """
        Restore People Counts Estimation from a given Json

        :param j_data:
        :return:
        """
        if not ('entrances' in j_data):
            raise Exception('NO entrances field in JSON')
        if not ('exits' in j_data):
            raise Exception('NO exits field in JSON')
        if not ('device_id' in j_data):
            raise Exception('NO device_id field in JSON')

        self.device_id = j_data['device_id']

        for e in j_data['entrances']:
            self.entrances.put((e[0], e[1]))

        for e in j_data['exits']:
            self.exits.put((e[0], e[1]))

    def jsonify(self):
        entrances = []
        exits = []
        while not self.entrances.empty():
            e = self.entrances.get()
            entrances.append(e)
        while not self.exits.empty():
            e = self.exits.get()
            exits.append(e)
        obj_d = {'device_id': self.device_id, 'entrances': entrances, 'exits': exits}
        obj_j = json.dumps(obj_d)
        return obj_j, len(entrances) + len(exits)

    def rollback(self, j_str: str):
        j_obj = json.loads(j_str)
        self.load(j_obj)

    def __clear__(self):
        pass
        # self.entrances.clear()
        # self.exits.clear()
