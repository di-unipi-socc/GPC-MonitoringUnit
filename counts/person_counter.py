import time

from mc_tracker.sct import TrackedObj
import cv2 as cv

from utils.misc import COLOR_PALETTE


def get_area(rect=(0, 0, 0, 0)):
    left, top, right, bottom = rect
    return abs(left - right) * abs(bottom - top)


def area_intersect(rect1=(0, 0, 0, 0), rect2=(0, 0, 0, 0)):
    """
    :param rect1:
    :param rect2:
    :return: Intersection Area of 2 rectangles
    """
    l1, t1, r1, b1 = rect1
    l2, t2, r2, b2 = rect2
    x_dist = min(r1, r2) - max(l1, l2)
    y_dist = min(b1, b2) - max(t1, t2)
    overlap_area = 0
    if x_dist > 0 and y_dist > 0:
        overlap_area = x_dist * y_dist
    return overlap_area


class Person:
    """
    Class reference that represent a Silhouette.
    Each silhouette is associated with random-incremental id, assigned on creation-time.
    It contains some utility fields to estimate entrances/exits into/from the building.
    """
    def __init__(self, detection: TrackedObj, gate_area=(0, 0, 0, 0)):
        self.id = int(detection.label.split(' ')[-1]) if isinstance(detection.label, str) else int(detection.label)
        assert self.id >= 0

        # Utility field that indicates how many frames without appear can be considered present
        # in people counts enter/exits estimation.
        # This is due to possibly error in silhouette-id re-assignation
        self.life_frames = 30
        self.skipable_frames = self.life_frames

        self.current_location = detection.rect
        self.out_degree_0 = self.calc_out_degree(gate_area)
        self.out_degree = self.out_degree_0

    def calc_out_degree(self, gate_rect=(0, 0, 0, 0)):
        """
        Establish percentage of silhouette's area considered "outside building".

        :param gate_rect:
        :return:
        """
        intersection = area_intersect(self.current_location, gate_rect)
        curr_area = get_area(self.current_location)
        out_d = int((intersection / curr_area) * 100)
        return out_d

    def update_out(self, gate_rect=(0, 0, 0, 0)):
        out_d = self.calc_out_degree(gate_rect)
        assert 0 <= out_d <= 100
        self.out_degree = out_d

    def reset_out(self, gate_rect=(0, 0, 0, 0)):
        self.out_degree_0 = self.calc_out_degree(gate_rect)

    def update_position(self, new_rect=(0, 0, 0, 0), gate_rect=(0, 0, 0, 0)):
        self.current_location = new_rect
        self.update_out(gate_rect)
        self.reset_skip_frames()

    def miss_frame(self):
        """
        Report that for one frame this ID is not present in the frame
        :return:
        """
        if self.skipable_frames > 0:
            self.skipable_frames -= 1

    def reset_skip_frames(self):
        self.skipable_frames = self.life_frames


class GateWatcher:
    """
    Class that perform People Count Enter/Exits Estimations.
    Contain fields to keep track of IDs considered inside or outside, and try to establish when an ID
    perform an entrance/exit action
    """
    def __init__(self, gateway_rect=(0, 0, 0, 0), in_out_thresh=50, setup=False):
        assert 0 <= in_out_thresh <= 100
        # Out Degree threshold that establish if an ID is inside or outside the building
        self.in_out_th = in_out_thresh
        self.n_persons = 0
        self.n_in = 0
        self.n_out = 0

        # Dicts of IDs, that divide IDs in inside/outside IDs
        self.outsiders = {}
        self.insiders = {}

        self.gate_rect = gateway_rect
        self.setup = setup

    def get_counters(self):
        """
        :return: current People Counts Estimation
        """
        counts = (self.n_in, self.n_out)
        self.__reset_counters__()
        return counts

    def __reset_counters__(self):
        self.n_persons = 0
        self.n_in = 0
        self.n_out = 0

    def update(self, frame, objects):
        """
        Perform the Counts Estimation and enrich Frame with debugging information.

        :param frame:
        :param objects: Silhouettes detected into the frame
        :return:
        """
        self.update_frame(objects)
        self.n_persons = self.n_in - self.n_out
        self.print_gateway(frame)
        if self.setup:
            self.print_grid(frame)

    def update_frame(self, detections):
        """
        Perform the Enter/Exits estimations
        :param detections: Silhouettes detected
        :return:
        """
        ids_in_frame = set()
        detection_dict = {}
        for d in detections:
            # Retrieve Silhouettes IDs
            id = int(d.label.split(' ')[-1]) if isinstance(d.label, str) else int(d.label)
            if id >= 0:
                ids_in_frame.add(id)
                detection_dict[str(id)] = d

        # Update the state of IDs considered inside/outside, and add new IDs, and check if some
        # ID is entered/exited in the building
        self.update_outsiders(detection_dict, ids_in_frame)
        self.update_insiders(detection_dict, ids_in_frame)
        self.add_new_entries(detection_dict)

        self.check_correctness()

    def update_outsiders(self, detect_dict: dict, ids_list: set):
        """
        Update ID's States of silhouettes considered outside (silhouette-box position, skip-frames, ...).
        Check if some ID is transit from "outside" to "inside"

        :param detect_dict:
        :param ids_list:
        :return:
        """
        miss_id = []
        for id in self.outsiders.keys():
            if int(id) in ids_list:
                self.outsiders[id].update_position(detect_dict[id].rect, self.gate_rect)
                self.try_enter(self.outsiders[id])
                ids_list.remove(int(id))
                del detect_dict[id]
            else:
                self.outsiders[id].miss_frame()
                if self.outsiders[id].skipable_frames <= 0:
                    miss_id.append(id)
        for id in self.insiders.keys():
            if id in self.outsiders.keys():
                del self.outsiders[id]
        for id in miss_id:
            del self.outsiders[id]
        # return detect_dict, ids_list

    def update_insiders(self, detect_dict: dict, ids_list: set):
        """
        Update ID's States of silhouettes considered inside (silhouette-box position, skip-frames, ...)
        Check if some ID is transit from "inside" to "outside"

        :param detect_dict:
        :param ids_list:
        :return:
        """
        miss_id = []
        for id in self.insiders.keys():
            if int(id) in ids_list:
                self.insiders[id].update_position(detect_dict[id].rect, self.gate_rect)
                self.try_exit(self.insiders[id])
                ids_list.remove(int(id))
                del detect_dict[id]
            else:
                self.insiders[id].miss_frame()
                if self.insiders[id].skipable_frames <= 0:
                    miss_id.append(id)
        for id in self.outsiders.keys():
            if id in self.insiders.keys():
                del self.insiders[id]
        for id in miss_id:
            del self.insiders[id]
        # return detect_dict, ids_list

    def add_new_entries(self, detect_dict: dict):
        for id in detect_dict.keys():
            assert int(id) not in self.outsiders
            assert int(id) not in self.insiders
            person = Person(detect_dict[id], self.gate_rect)
            if person.out_degree_0 >= self.in_out_th:
                self.outsiders[id] = person
            else:
                self.insiders[id] = person

    def try_enter(self, p: Person):
        """
        Establish if an "outsider-ID" was transit from outside to inside

        :param p:
        :return:
        """
        assert p.id >= 0
        assert p.out_degree_0 >= self.in_out_th
        assert str(p.id) in self.outsiders.keys()
        assert str(p.id) not in self.insiders.keys()
        if p.out_degree < self.in_out_th:
            # del self.outsiders[str(p.id)]
            p.reset_out(self.gate_rect)
            self.insiders[str(p.id)] = p
            self.n_in += 1
            # print(self.n_persons)

    def try_exit(self, p: Person):
        """
        Establish if an "insider-ID" was transit from inside to outside
        :param p:
        :return:
        """
        assert p.id >= 0
        assert p.out_degree_0 < self.in_out_th
        assert str(p.id) in self.insiders.keys()
        assert str(p.id) not in self.outsiders.keys()
        if p.out_degree >= self.in_out_th:
            # del self.insiders[str(p.id)]
            p.reset_out(self.gate_rect)
            self.outsiders[str(p.id)] = p
            self.n_out += 1

    def print_gateway(self, frame):
        left, top, right, bottom = self.gate_rect
        box_color = COLOR_PALETTE[1 % len(COLOR_PALETTE)]
        cv.rectangle(frame, (left, top), (right, bottom), box_color, thickness=3)

    def print_grid(self, frame):
        """
        Print a grid on frame to help in gate coordinate setting in setup process
        :param frame:
        :return:
        """
        h_max, w_max, _ = frame.shape
        # cv.line(frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 1, 1)
        for x in range(0, w_max, 10):
            t = 2 if (x % 50) == 0 else 1
            cv.line(frame, (x, 0), (x, h_max), (255, 0, 0), t, 1)
        for y in range(0, h_max, 10):
            t = 2 if (y % 50) == 0 else 1
            cv.line(frame, (0, y), (w_max, y), (255, 0, 0), t, 1)

    def check_correctness(self):
        for p in self.insiders.values():
            p: Person
            assert p.out_degree_0 < self.in_out_th
        for p in self.outsiders.values():
            p: Person
            assert p.out_degree_0 >= self.in_out_th


class PersonCounter:
    """
    It collect all counts estimation performed by provided GateWatchers
    """
    def __init__(self, watchers_list):
        self.global_count = 0
        self.global_in = 0
        self.global_out = 0
        self.watchers = watchers_list

    def update(self, frames, all_objects):
        """
        Perform Counts Estimation on Entrances and Exits

        :param frames: current frames
        :param all_objects: objects detected with assigned IDs
        :return:
        """
        assert len(frames) == len(all_objects) == len(self.watchers)
        for i, (frame, objects) in enumerate(zip(frames, all_objects)):
            self.watchers[i].update(frame, objects)

        count_in = 0
        count_out = 0
        for w in self.watchers:
            w: GateWatcher
            n_in, n_out = w.get_counters()
            count_in += n_in
            count_out += n_out
        self.global_in = count_in
        self.global_out = count_out
        self.global_count = count_in - count_out

    def get_counters(self):
        counts = (self.global_in, self.global_out, int(time.time()))
        self.__reset_counters__()
        return counts

    def __reset_counters__(self):
        self.global_count = 0
        self.global_in = 0
        self.global_out = 0
