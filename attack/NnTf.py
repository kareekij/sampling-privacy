import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import time

class NN:
    def __init__(self, xDim, yDim):
        with tf.device('/cpu:0'):
            self.xDim = xDim
            self.yDim = yDim
            size = (xDim+yDim)/2
            # size = xDim
            # Initial all variable and equation graph
            self.x = tf.placeholder(tf.float32, [None, xDim])
            self.y = tf.placeholder(tf.float32, [None, yDim])

            self.wH1 = tf.Variable(tf.random_normal([xDim, size]), trainable=True)
            self.y1 = tf.sigmoid(tf.matmul(self.x, self.wH1))

            # self.wH2 = tf.Variable(tf.random_normal([size, size]), trainable=True)
            # self.y2 = tf.sigmoid(tf.matmul(self.y1, self.wH2))
            #
            # self.wH3 = tf.Variable(tf.random_normal([size, size]), trainable=True)
            # self.y3 = tf.sigmoid(tf.matmul(self.y2, self.wH3))
            #
            # self.wH4 = tf.Variable(tf.random_normal([size, size]), trainable=True)
            # self.y4 = tf.sigmoid(tf.matmul(self.y3, self.wH4))
            #
            # self.wH5 = tf.Variable(tf.random_normal([size, size]), trainable=True)
            # self.y5 = tf.sigmoid(tf.matmul(self.y4, self.wH5))

            self.W = tf.Variable(tf.random_normal([size, yDim]), trainable=True)
            self.y_ = tf.sigmoid(tf.matmul(self.y1, self.W))

            self.costFunction = tf.reduce_sum(tf.scalar_mul(0.5, tf.square(tf.subtract(self.y, self.y_))))
            self.optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(self.costFunction)
            self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

            # Initial session
            self.init = tf.global_variables_initializer()
            # self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
            self.sess = tf.Session()
            self.sess.run(self.init)

            # For Tensorboard
            # tf.summary.scalar("Cost", self.costFunction)
            # tf.summary.scalar("Accuracy", self.accuracy)
            # self.merge = tf.summary.merge_all()
            # self.write = tf.summary.FileWriter("./logs", self.sess.graph)

    def fit(self, x, y, epoch):
        # Train
        for _ in range(epoch):
            tmp, cost = self.sess.run([self.optimizer,self.costFunction], feed_dict={self.x: x, self.y: y})
            # summary, tmp, cost = self.sess.run([self.merge, self.optimizer, self.costFunction], feed_dict={self.x: x, self.y: y})
            # self.write.add_summary(summary, _)
            if cost < self.xDim*self.yDim*0.01:
                print _
                break

        # Test the model
        print("Training Accuracy: {0}".format(self.sess.run(self.accuracy, feed_dict={self.x: x, self.y: y})))

    def predict(self, x):
        return self.sess.run(tf.argmax(self.y_, 1), feed_dict={self.x: x})

if __name__ == '__main__':

    # Train Data
    trainX = [[0.0, 0.0, 0.0],
              [0.0, 0.0, 1.0],
              [0.0, 1.0, 0.0],
              [0.0, 1.0, 1.0],
              [1.0, 0.0, 0.0],
              [1.0, 0.0, 1.0],
              [1.0, 1.0, 0.0],
              [1.0, 1.0, 1.0]]
    trainY = [[1.0, 0.0],
              [0.0, 1.0],
              [1.0, 0.0],
              [0.0, 1.0],
              [0.0, 1.0],
              [0.0, 1.0],
              [1.0, 0.0],
              [1.0, 0.0]]

    testX = [[0.7, 1.0, 0.2],
              [0.5, 0.3, 1.0]]

    nn = NN(3, 2)
    start_time = time.time()
    nn.fit(trainX, trainY, 1000)
    predict = nn.predict(testX)
    print predict
    print("--- %s seconds ---" % (time.time() - start_time))

