import multiprocessing
from lxyTools.progressBar import simpleBar
import time


def processRun(obj, i):
    for val in obj.processPeriodlist[i]:
        # print(obj.progressCount.value)

        re = obj.fun(val)
        obj.result.put(re)
        obj.progressCount[i] += 1

class mutiProcessLoop:
    def __init__(self, fun, period, n_process=4, silence=False):
        self.period = period
        self.n_process = n_process
        self.silence = silence
        self.fun = fun
        self.processPeriodlist = []
        self.processlist = []
        self.manager = multiprocessing.Manager()
        self.result = multiprocessing.Queue()
        self.progressCount = self.manager.Array('i', [0]*n_process)
        for i in range(n_process):
            self.processPeriodlist.append([])
            self.processlist.append(
                multiprocessing.Process(
                    target=processRun,
                    args=(self, i))
                )
        for i in range(len(period)):
            self.processPeriodlist[i % n_process].append(period[i])

        if not silence:
            self.bar = simpleBar(len(period))

    def run(self):
        for process in self.processlist:
            process.start()

        resultlist = []
        while 1:

            while not self.result.empty():
                resultlist.append(self.result.get_nowait())

            val = 0
            for i in self.progressCount:
                val += i
            time.sleep(0.5)

            if not self.silence: self.bar.update(val)
            if val >= len(self.period):
                if not self.silence: self.bar.finish()
                break

        while not self.result.empty():
            resultlist.append(self.result.get_nowait())

        for process in self.processlist:
            # process.join()
            process.terminate()



        return resultlist
