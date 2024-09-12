import time
from tabulate import tabulate


class times_and_tags:
    def __init__(self):
        self.times = []
        self.tags = []

    def add(self, tag):
        self.times.append(time.time())
        self.tags.append(tag)

    def print(self):
        # add last time (if needed)
        if len(self.tags) == len(self.times):
            self.times.append(time.time())

        # delta t
        dt = [self.times[i + 1] - self.times[i] for i in range(len(self.times) - 1)]

        # delta t percentage
        sum_dt = sum(dt)
        p_dt = [x*100/sum_dt for x in dt]

        # create data for table
        data = []
        headers = ["Pos", "Tag", "Time [s]", "Time [%]"]
        for i in range(len(self.times) - 1):
            data.append([i, self.tags[i], dt[i], p_dt[i]])
        data.append(["x", "SUM", sum_dt, 100])

        # print table
        print(tabulate(data, headers=headers, tablefmt='orgtbl'))
        print("")

    def reset(self):
        self.times = []
        self.tags = []
