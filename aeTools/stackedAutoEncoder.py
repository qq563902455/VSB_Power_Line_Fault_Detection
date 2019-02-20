import numpy as np


class stackedLabelDenoisingAE:

    def __init__(self, aelist=[]):
        self.aelist = aelist

    def train(self, epoch, batch_size, X, y, y_true, sess, testX=-1, testy=-1):

        trainX = X
        inputTestX = testX
        for model in self.aelist:
            model.train(epoch, batch_size, trainX, y, y_true,
                        sess, inputTestX, testy)
            inputTestX = model.encoder(inputTestX, testy, sess)
            trainX = model.encoder(trainX, y, sess)

    def encoder(self, X, y, sess):
        outlist = []
        inputX = X
        for model in self.aelist:
            inputX = model.encoder(inputX, y, sess)
            outlist.append(inputX)
        return np.concatenate(outlist, axis=1)

    def decoder(self, X, y, sess):
        outlist = []
        inputX = X
        for model in self.aelist:
            inputX = model.decoder(inputX, y, sess)
            outlist.append(inputX)
        return np.concatenate(outlist, axis=1)
