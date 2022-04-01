import time

import cv2

GREEN = (0, 255, 0)
RED = (0, 0, 255)


class CrowdGuardian:
    """
    Class that coordinate all CrowdWatchers
    """
    def __init__(self, watchers_ls):
        self.watchers = watchers_ls
        self.alerts = []

    def check(self, all_detecton):
        r_list = []
        for w, d in zip(self.watchers, all_detecton):
            w: CrowdWatcher
            n = len(d)
            r = w.check(n)
            r_list.append(r)
        self.alerts = r_list
        return self.any_alert()

    def any_alert(self):
        alert = False
        for a in self.alerts:
            alert = alert or a

        return alert

    def render_alerts(self, frames):
        for frame, watcher in zip(frames, self.watchers):
            watcher.render_alert(frame)


class CrowdWatcher:
    """
    Class that implement methods to check crowding formation in front of an entrance.
    """
    def __init__(self, src, max_threshold, alert_timeout_sec=30, skippable_check=5):
        self.src = src
        self.max = max_threshold
        self.timeout = alert_timeout_sec
        self.skip_check = skippable_check

        self.start_alert_t = None
        self.alert_time_elapsed = 0
        self.alert = False
        self.skip = skippable_check

    def reset(self):
        self.start_alert_t = None
        self.alert_time_elapsed = 0
        self.alert = False
        self.skip = self.skip_check

    def check(self, n_persons):
        """
        Check if in front of the gate was created a crowding situation for more than defined seconds

        :param n_persons:
        :return:
        """
        if n_persons >= self.max:
            # Threshold passed
            self.skip = self.skip_check  # reset skippable frame

            if not self.start_alert_t:
                # timer is not already initialized => initialize it
                self.start_alert_t = int(time.time())
                return self.alert

            if self.alert:
                # Alerti is already turned on
                return self.alert

            # timer is already initialized => update it
            now = int(time.time())
            self.alert_time_elapsed = now - self.start_alert_t

            if self.alert_time_elapsed >= self.timeout:
                # It's enough time with so much person detected => Time to alert!
                self.alert = True
                return self.alert
        else:
            # Threshold respected
            if self.alert_time_elapsed is not None:
                # timer is going on, and maybe could be some miss-detection
                self.skip -= 1
            if self.skip < 0:
                # all skip check are gone, maybe the counting is correct
                self.reset()
        return self.alert

    def render_alert(self, frame):
        """
        Draw a Green/Red rectangle on frame's border, to signal crowding situation

        :param frame:
        :return:
        """
        h_max, w_max, _ = frame.shape
        color = None
        if self.alert:
            color = RED
        else:
            color = GREEN
        cv2.rectangle(frame, (3, 3), (w_max-3, h_max-3), color, thickness=3)

