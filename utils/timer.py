from datetime import datetime
import logging
import time


class Timer:
    """
    Utility class used to check execution time of methods/functions
    """
    def __init__(self, name, rec_max=1000):
        self.name = name
        self.last_ts = time.time_ns()
        self.times_repo = {}
        self.max_len = rec_max

    def start(self):
        self.last_ts = time.time_ns()

    def check(self, task=None):
        now = time.time_ns()
        diff = now - self.last_ts
        if task and type(task) == str:
            if task not in self.times_repo:
                time_ls = []
                self.times_repo[task] = time_ls

            time_ls = self.times_repo[task]
            if len(time_ls) >= self.max_len:
                time_ls.pop(0)
            time_ls.append(diff)

        return diff

    def print_check(self, taskname=''):
        timestamp = self.check()
        if timestamp > 0:
            print(f"[{datetime.fromtimestamp(time.time()).second}] Timer {self.name}:{taskname} {timestamp}ns")

    def avg_calc(self, print_avg=False):
        self.avgs = {}
        self.min = {}
        self.max = {}
        for task in self.times_repo.keys():
            time_ls = self.times_repo[task]
            t = int(sum(time_ls) / len(time_ls))
            t = round(float(t / 1000000), 3)
            self.avgs[task] = t
            ls_sort = sorted(time_ls)
            self.min[task] = int(ls_sort[0] / 1000000)
            self.max[task] = int(ls_sort[-1] / 1000000)
        if print_avg:
            self.avg_print()

    def avg_print(self):
        if not self.avgs:
            return
        tups = []
        for task, t in zip(self.avgs.keys(), self.avgs.values()):
            tups.append((task, t))

        tups_ord = sorted(tups, key=(lambda x: x[1]))
        print(f'\n#################\n[{self.name} Timer]')
        for task, t in tups_ord:
            print(f"avg time:\t{t}ms \tmin:{self.min[task]} \tMAX:{self.max[task]} \t[{task}]")
        print('')

    def avg_log(self, file_path=None):
        if not self.avgs:
            return
        tups = []
        for task, t in zip(self.avgs.keys(), self.avgs.values()):
            tups.append((task, t))

        tups_ord = sorted(tups, key=(lambda x: x[1]))
        to_print = f'\n#################\n[{self.name} Timer] \t@{str(datetime.now().replace(microsecond=0))}\n'
        for task, t in tups_ord:
            to_print += f"avg time:\t{t}ms \tmin:{self.min[task]} \tMAX:{self.max[task]} \t[{task}]\n"
        to_print += '\n'
        if not file_path:
            logging.info(to_print)
        else:
            with open(file_path, 'a') as file:
                file.write(to_print)
