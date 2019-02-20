import sys
import time


class simpleBar:

    def __init__(self, taskSum, barlength=20):
        self.taskSum = taskSum
        self.lastRatio = 0
        self.lastTime = time.time()
        self.startTime = time.time()
        self.barlength = barlength
        self.outlen = 0

    def update(self, val):

        ratio = val/self.taskSum
        finishedNum = int(ratio*self.barlength)
        unfinishedNum = self.barlength - finishedNum
        outbar = '/'+'*'*finishedNum+'-'*unfinishedNum+'/'
        diffRatio = ratio - self.lastRatio
        self.lastRatio = ratio

        timeNow = time.time()
        diffTime = timeNow - self.lastTime
        self.lastTime = timeNow

        elapsedTime = timeNow - self.startTime

        if diffRatio != 0:
            leftTime = int((1 - ratio)/diffRatio*diffTime)

            outbar += '  '
            outbar += 'time left:'
            outbar += str(int(leftTime/3600))+':'
            outbar += str(int(leftTime % 3600/60))+':'
            outbar += str(int(leftTime % 60))+'  '

            outbar += '  '
            outbar += 'elapsed time:'
            outbar += str(int(elapsedTime/3600))+':'
            outbar += str(int(elapsedTime % 3600/60))+':'
            outbar += str(int(elapsedTime % 60))+'  '

            sys.stdout.write(' ' * self.outlen + '\r')
            sys.stdout.flush()
            self.outlen = len(outbar)
            sys.stdout.write(outbar + '\r')
            sys.stdout.flush()

    def finish(self):
        print()
