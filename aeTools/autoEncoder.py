import tensorflow as tf


class autoEncoder:

    def __init__(self, hide_size, input_size,
                 sparsity=0, gamma=0.5, p=0.05):
        self.x = tf.placeholder("float", [None, input_size])
        self.eW = tf.Variable(tf.random_normal([input_size, hide_size],
                              seed=2015))
        self.eb = tf.Variable(tf.random_normal([hide_size], seed=123))

        self.dW = tf.Variable(tf.random_normal([hide_size, input_size],
                              seed=478))
        self.db = tf.Variable(tf.random_normal([input_size], seed=654))

        self.encoderOut = tf.nn.sigmoid(tf.add(
            tf.matmul(self.x, self.eW), self.eb))

        self.decoderOut = tf.nn.sigmoid(tf.add(
            tf.matmul(self.encoderOut, self.dW), self.db))

        pj = tf.reduce_mean(self.encoderOut, 0)
        sparse_penalty = sparsity*tf.reduce_sum(p*tf.log(p/pj) +
                                                (1-p)*tf.log((1-p)/(1-pj)))
        l2_penalty = gamma * (tf.reduce_sum(tf.pow(self.eW, 2)) +
                              tf.reduce_sum(tf.pow(self.dW, 2)))/2

        self.cost = tf.reduce_sum(tf.pow(self.x - self.decoderOut, 2)) +\
            l2_penalty + sparse_penalty
        self.optimizer_ae = tf.train.AdamOptimizer(0.01).minimize(self.cost)

    def train(self, epoch, batch_size, X, sess, testX=-1):
        for count in range(epoch):
            for i in range(int(X.shape[0]/batch_size)):
                xval = X[i*batch_size:(i*batch_size+batch_size), :]
                sess.run(self.optimizer_ae, feed_dict={self.x: xval})
            if type(testX) != int:
                print('cost: ', sess.run(self.cost, feed_dict={self.x: testX}))

    def decoder(self, X, sess):
        return sess.run(self.decoderOut, feed_dict={self.x: X})

    def encoder(self, X, sess):
        return sess.run(self.encoderOut, feed_dict={self.x: X})


class labelDenoisingAutoEncoder:

    def __init__(self, hide_size, input_size, output_size,
                 beta=2, gamma=0.5, Lambda=1.5, sparsity=0, p=0.05):
        self.x = tf.placeholder("float", [None, input_size])
        self.y = tf.placeholder("float", [None, 1])
        self.y_true = tf.placeholder("float", [None, output_size])

        eW = tf.Variable(tf.random_normal([input_size+1, hide_size],
                                          stddev=0.5))
        eb = tf.Variable(tf.random_normal([hide_size], stddev=0.5))

        dWx = tf.Variable(tf.random_normal([hide_size, input_size],
                                           stddev=0.5))
        dbx = tf.Variable(tf.random_normal([input_size], stddev=0.5))

        dWy = tf.Variable(tf.zeros([hide_size, output_size]))
        dby = tf.Variable(tf.zeros([output_size]))

        self.encoderOut = tf.nn.sigmoid(
            tf.matmul(tf.concat([self.x, self.y], 1), eW) + eb)
        self.decoderOut = tf.nn.sigmoid(tf.matmul(self.encoderOut, dWx) + dbx)
        self.softmaxOut = tf.nn.softmax(tf.matmul(self.encoderOut, dWy) + dby)

        self.AE_se = tf.reduce_sum(tf.pow(self.x - self.decoderOut, 2))
        cross_entropy = -Lambda*tf.reduce_sum(
            self.y_true*tf.log(self.softmaxOut))
        sigmoid_l2 = gamma*(tf.reduce_sum(tf.pow(eW, 2)) +
                            tf.reduce_sum(tf.pow(dWx, 2)))
        softmax_l2 = beta*tf.reduce_sum(tf.pow(dWy, 2))

        self.cost = self.AE_se + cross_entropy + sigmoid_l2 + softmax_l2
        self.optimizer = tf.train.AdamOptimizer(0.01).minimize(self.cost)

    def train(self, epoch, batch_size, X, y, y_true, sess, testX=-1, testy=-1):
        for count in range(epoch):
            for i in range(int(X.shape[0]/batch_size)):
                xval = X[i*batch_size:(i*batch_size+batch_size), :]
                yval = y[i*batch_size: (i*batch_size+batch_size), :]
                yval_true = y_true[i*batch_size: (i*batch_size+batch_size), :]
                sess.run(
                    self.optimizer, feed_dict={self.x: xval,
                                               self.y: yval,
                                               self.y_true: yval_true})

            if type(testX) != int:
                print('se: ', sess.run(self.AE_se, feed_dict={self.x: testX,
                                                              self.y: testy}))

    def decoder(self, X, y, sess):
        return sess.run(self.decoderOut, feed_dict={self.x: X,
                                                    self.y: y})

    def encoder(self, X, y, sess):
        return sess.run(self.encoderOut, feed_dict={self.x: X,
                                                    self.y: y})


class aeWithModel(autoEncoder):
    def __init__(self, model, hide_size, input_size,
                 sparsity=0, gamma=0.5, p=0.05):
        self.model = model
        autoEncoder.__init__(self, hide_size, input_size,
                             sparsity, gamma, p)

    def modelFit(self, X, y, sess, epoch, batch_size, valid=-1, metric=-1):
        x_represention = self.encoder(X, sess)
        if type(valid) != int:
            valid_en = self.encoder(valid[0], sess)

        self.model.fit(x_represention, y,
                       sess=sess, epoch=epoch, batch_size=batch_size,
                       valid=(valid_en, valid[1]), metric=metric)

    def predict(self, X, sess):
        x_represention = self.encoder(X, sess)
        return self.model.predict(x_represention, sess)
