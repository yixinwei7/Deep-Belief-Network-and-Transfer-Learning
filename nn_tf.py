import numpy as np
import math
import tensorflow as tf
import matplotlib.pyplot as plt
from bn import bn_layer
import pandas as pd
class NN(object):
    """Docstring for NN. """

    def __init__(self, sizes, hidden,opts, X, Y,X_test,Y_test, X_normal, Y_normal,X_migration_test,Y_migration_test,X_data_migration_test,Y_data_migration_test,size_add):

        self._sizes = sizes
        self._opts = opts
        self.hidden = hidden
        self._X = X
        self._Y = Y
        self._X_test = X_test
        self._Y_test = Y_test
        self._X_normal = X_normal
        self._Y_normal = Y_normal
        self._X_migration_test = X_migration_test
        self._Y_migration_test = Y_migration_test

        self._X_data_migration_test = X_data_migration_test
        self._Y_data_migration_test = Y_data_migration_test
        self._size_add = size_add
        self.w_list = []
        self.b_list = []
        self.w_add_list = []
        self.b_add_list = []
        self.a1 =[]
        self.b1 =[]
        # self.w_migration = []
        # self.b_migration = []
        input_size = X.shape[1]


        for size in self._sizes +self.hidden+ [Y.shape[1]]:  # 400  50  10

            max_range = 4 * math.sqrt(6. / (input_size + size))  # 一种方便快速的初始化方法

            # Initialize weights through a random uniform distribution
            self.w_list.append(np.random.uniform(-max_range, max_range, [input_size, size]).astype(np.float32))
            # Initialize bias as zeroes
            self.b_list.append(np.zeros([size], np.float32))

            input_size = size

        input_size_add = 6
        for size in  self._size_add + [Y.shape[1]]:
            max_range = 4 * math.sqrt(6. / (input_size_add + size))  # 一种方便快速的初始化方法
            self.w_add_list.append(np.random.uniform(-max_range, max_range, [input_size_add, size]).astype(np.float32))
            self.b_add_list.append(np.zeros([size], np.float32))
            input_size_add = size




    def load_from_dbn(self, dbn):

        assert len(dbn._sizes) == len(self._sizes)
        for i in range(len(self._sizes)):  # Check if for each RBN the expected sizes are correct
            assert dbn._sizes[i] == self._sizes[i]
        for i in range(len(self._sizes)):  # If everything is correct, bring over the weights and biases
            print(i)  # 0  1
            print(dbn.rbm_list[i].w)
            self.w_list[i] = dbn.rbm_list[i].w
            self.b_list[i] = dbn.rbm_list[i].hb



    def train(self):                  #训练过程

        _a = [None] * (len(self._sizes) + 3)
        _y_model = [None]
        _w = [None] * (len(self._sizes) + 3)
        _b = [None] * (len(self._sizes) + 3)
        _a[0] = tf.placeholder("float", [None, self._X.shape[1]])
        y = tf.placeholder("float", [None, self._Y.shape[1]])

        # Define variables and activation functoin

        for i in range(len(self._sizes) + 3):
            #print("w b  len")
            #print(i)  # 0  1  2  3
            _w[i] = tf.Variable(self.w_list[i])
            _b[i] = tf.Variable(self.b_list[i])

        for i in range(1, len(self._sizes)+1):
            #print('_a  len')
            #print(i)  # 1  2  3
            _a[i] = tf.nn.sigmoid(tf.matmul(_a[i - 1], _w[i - 1]) + _b[i - 1])

        # batch_mean, batch_var = tf.nn.moments(tf.matmul(_a[2], _w[2]) + _b[2], [0])
        # _a[3] = tf.nn.sigmoid(tf.nn.batch_normalization(tf.matmul(_a[2], _w[2]) + _b[2], batch_mean, batch_var, 0, 1, variance_epsilon=1e-3))

        _a[4] = tf.matmul(_a[3], _w[3]) + _b[3]
        batch_norm =bn_layer(_a[4],1)
        _a[4] = tf.nn.sigmoid(batch_norm.batch_norm_layer())


        _a[5] = tf.matmul(_a[4], _w[4]) + _b[4]
        batch_norm1 = bn_layer(_a[5], 1)
        _a[5] = tf.nn.sigmoid(batch_norm1.batch_norm_layer())
        _y_model = tf.nn.softmax(tf.matmul(_a[5], _w[5]) + _b[5])

        # Define the cost function
        cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(_y_model), 1))
        # Define the training operation (Momentum Optimizer minimizing the Cost function)

        # train_op = tf.train.MomentumOptimizer(
        #     self._opts._learning_rate, self._opts._momentum).minimize(cost)

        train_op = tf.train.GradientDescentOptimizer(self._opts._learning_rate).minimize(cost)
        # Prediction operation
        predict_op = tf.argmax(_y_model, 1)   #根据axis取值的不同返回每行或者每列最大值的索引         1为行，0为列
        actuals = tf.argmax(y, 1)

        ones_like_actuals = tf.ones_like(actuals)
        zeros_like_actuals = tf.zeros_like(actuals)
        ones_like_predictions = tf.ones_like(predict_op)
        zeros_like_predictions = tf.zeros_like(predict_op)

        tn = tf.reduce_sum(tf.cast(
            tf.logical_and(tf.equal(actuals, ones_like_actuals), tf.equal(predict_op, ones_like_predictions)),
            "float"))

        fp = tf.reduce_sum(
            tf.cast(tf.logical_and(tf.equal(actuals, ones_like_actuals), tf.equal(predict_op, zeros_like_predictions)),
                    "float"))
        fpr = tf.divide(fp,tf.add(fp,tn),name='prob2')
#

        with tf.Session() as sess:
            l_train_ce = []
            l_train_yan = []
            llo = []
            w_value = []
            # print(_a[-1].eval())
            sess.run(tf.initialize_all_variables())
            sample1=np.arange(self._opts._epoches)
            for i in range(self._opts._epoches):
                for start, end in zip(
                        range(0, len(self._X), self._opts._batchsize),
                        range(self._opts._batchsize, len(self._X)+self._opts._batchsize, self._opts._batchsize)):
                    sess.run(train_op, feed_dict={
                        _a[0]: self._X[start:end], y: self._Y[start:end]})
                ll= sess.run(cost, feed_dict={_a[0]: self._X[start:end], y: self._Y[start:end]})

                llo.append(ll)


                #  print(len(self._X))  55000
                for i in range(len(self._sizes) + 3):
                    self.w_list[i] = sess.run(_w[i])
                    self.b_list[i] = sess.run(_b[i])
                #w_value5 = sess.run(_w[5],feed_dict={_a[0]: self._X[start:end], y: self._Y[start:end]})

                    # self.a1[i] = sess.run(_w[i])
                    # self.b1[i] = sess.run(_b[i])
                print(np.mean(np.argmax(self._Y, axis=1) ==
                              sess.run(predict_op, feed_dict={
                                  _a[0]: self._X, y: self._Y})))

                train_ce = np.mean(np.argmax(self._Y, axis=1) ==
                                   sess.run(predict_op, feed_dict={
                                       _a[0]: self._X, y: self._Y}))

                l_train_ce.append(train_ce)

                train_yan = np.mean(np.argmax(self._Y_test, axis=1) ==
                                     sess.run(predict_op, feed_dict={
                                         _a[0]: self._X_test, y: self._Y_test}))
                l_train_yan.append(train_yan)
            # print("训练后模型的参数")
            # print(w_value5)

            plt.plot(sample1, l_train_ce, marker="*", linewidth=1, linestyle="--", color="red")
            plt.plot(sample1, l_train_yan, marker="*", linewidth=1, linestyle="--", color="blue")
            plt.title("bp The acc of the train set and verification set")
            plt.xlabel("Sampling Point")
            plt.ylabel("acc")
            plt.grid(True)
            plt.show()

            plt.plot(sample1, llo, marker="*", linewidth=1, linestyle="--", color="red")
            plt.title("bp The variation of the loss")
            plt.xlabel("Sampling Point")
            plt.ylabel("Loss")
            plt.grid(True)
            plt.show()

            print('''''''')
            print(np.mean(np.argmax(self._Y_test, axis=1) ==
                                     sess.run(predict_op, feed_dict={
                                         _a[0]: self._X_test, y: self._Y_test})))
            print('训练集假阳性率',sess.run(fpr,feed_dict={
                                  _a[0]: self._X, y: self._Y}))
            print('验证集假阳性率', sess.run(fpr, feed_dict={
                                         _a[0]: self._X_test, y: self._Y_test}))


            print("迁移后的数据 先用来对未迁移前的模型进行测试")
            print(np.mean(np.argmax(self._Y_normal, axis=1) ==
                                     sess.run(predict_op, feed_dict={
                                         _a[0]: self._X_normal, y: self._Y_normal})))

            print(np.mean(np.argmax(self._Y_migration_test, axis=1) ==
                          sess.run(predict_op, feed_dict={
                              _a[0]: self._X_migration_test, y: self._Y_migration_test})))




    def train_migration(self):                         #在分类器前再加一层隐藏层，固定之前所有层，对于目标域的数据有了学习的空间
        _a = [None] * (len(self._sizes) + 3)
        _w = [None] * len(self.w_list)
        _b = [None] * len(self.b_list)

        _w_add = [None]*len(self.w_add_list)
        _b_add = [None]*len(self.b_add_list)

        _y_model = [None]
        _a[0] = tf.placeholder("float", [None, self._X.shape[1]])
        for i in range(len(self.w_list)):
            _w[i] = tf.constant(self.w_list[i])
            _b[i] = tf.constant(self.b_list[i])

        for i in range(1, len(self._sizes)+1 ):
            #print(len(self._sizes))  # 2  2
            #print(i)  # 1  2

            print("开始迁移，保持前向参数不变")  # 1  2

            _a[i] = tf.nn.sigmoid(tf.matmul(_a[i - 1], _w[i - 1]) + _b[i - 1])


        _a_migration = _a[3]
        y_migration = tf.placeholder("float", [None, self._Y_normal.shape[1]])


        _a[4] = tf.matmul(_a[3], _w[3]) + _b[3]
        batch_norm2 = bn_layer(_a[4], 1)
        _a[4] = tf.nn.sigmoid(batch_norm2.batch_norm_layer())

        _a[5] = tf.matmul(_a[4], _w[4]) + _b[4]
        batch_norm3 = bn_layer(_a[5], 1)
        _a[5] = tf.nn.sigmoid(batch_norm3.batch_norm_layer())

        _f1 = [None]
        _f2 = [None]



        for i in range(len(self._size_add) + 1):
            _w_add[i] = tf.Variable(self.w_add_list[i])
            _b_add[i] = tf.Variable(self.b_add_list[i])



        _f1 = tf.matmul(_a[5], _w_add[0]) + _b_add[0]
        batch_norm4 = bn_layer(_f1, 1)
        _f1 = tf.nn.sigmoid(batch_norm4.batch_norm_layer())


        _f2 = tf.nn.softmax(tf.matmul(_f1, _w_add[1]) + _b_add[1])


        cost_migration =  tf.reduce_mean(-tf.reduce_sum(y_migration * tf.log(_f2), 1))

        # train_op_migration = tf.train.MomentumOptimizer(
        #     self._opts._learning_rate_migration, self._opts._momentum).minimize(cost_migration)

        train_op_migration = tf.train.GradientDescentOptimizer(self._opts._learning_rate_migration).minimize(cost_migration)

        predict_op_migration = tf.argmax(_f2, 1)
        actuals = tf.argmax(y_migration, 1)

        ones_like_actuals = tf.ones_like(actuals)
        zeros_like_actuals = tf.zeros_like(actuals)
        ones_like_predictions = tf.ones_like(predict_op_migration)
        zeros_like_predictions = tf.zeros_like(predict_op_migration)

        tn = tf.reduce_sum(tf.cast(
            tf.logical_and(tf.equal(actuals, ones_like_actuals), tf.equal(predict_op_migration, ones_like_predictions)),
            "float"))

        fp = tf.reduce_sum(
            tf.cast(tf.logical_and(tf.equal(actuals, ones_like_actuals), tf.equal(predict_op_migration, zeros_like_predictions)),
                    "float"))
        fpr = tf.divide(fp, tf.add(fp, tn), name='prob2')


        with tf.Session() as sess:
            lo =[]
            lce =[]
            lyan = []
            sess.run(tf.initialize_all_variables())
            sample = np.arange(self._opts._epoches_migration)
            train_max = 0
            train_i = 0
            yanzhen_max = 0
            yanzhen_i = 0
            for i in range(self._opts._epoches_migration):
                for start, end in zip(range(0, len(self._X_normal), self._opts._batchsize_migration),
                                      range(self._opts._batchsize_migration, len(self._X_normal)+self._opts._batchsize_migration, self._opts._batchsize_migration)):
                    #print(start,end)
                    sess.run(train_op_migration, feed_dict={ _a[0]: self._X_normal[start:end], y_migration: self._Y_normal[start:end]})

                l=sess.run(cost_migration, feed_dict={_a[0]: self._X_normal[start:end], y_migration: self._Y_normal[start:end]})
                lo.append(l)
                #  print(len(self._X))  55000

                self.w_add_list[0] = sess.run(_w_add[0])
                self.b_add_list[0] = sess.run(_b_add[0])
                self.w_add_list[1] = sess.run(_w_add[1])
                self.b_add_list[1] = sess.run(_b_add[1])


                # if i % 100 == 0:
                #     print("迁移后训练集的准确率："+ "在第"+str(i)+"代"+str(np.mean(np.argmax(self._Y_normal, axis=1) ==
                #               sess.run(predict_op_migration, feed_dict={
                #                   _a[0]: self._X_normal, y_migration: self._Y_normal}))))
                #
                #     print("迁移后验证集的准确率：" + "在第"+str(i)+"代"+str(np.mean(np.argmax(self._Y_migration_test, axis=1) ==
                #             sess.run(predict_op_migration, feed_dict={
                #                 _a[0]: self._X_migration_test, y_migration: self._Y_migration_test}))))


                ce=  np.mean(np.argmax(self._Y_normal, axis=1) ==
                              sess.run(predict_op_migration, feed_dict={
                                  _a[0]: self._X_normal, y_migration: self._Y_normal}))
                lce.append(ce)

                yan = np.mean(np.argmax(self._Y_migration_test, axis=1) ==
                            sess.run(predict_op_migration, feed_dict={
                                _a[0]: self._X_migration_test, y_migration: self._Y_migration_test}))
                lyan.append(yan)

                if ce > train_max:
                    train_max = ce
                    train_i = i
                    print("迁移后   训练集  当前最优准确率：" + "在第" + str(train_i) + " (epoches) " + str(train_max))

                if yan > yanzhen_max:
                    yanzhen_max = yan
                    yanzhen_i = i
                    print("迁移后   验证集  当前最优准确率：" + "在第" + str(yanzhen_i) + " (epoches) " + str(yanzhen_max))

            print('训练集假阳性率', sess.run(fpr, feed_dict={
                                  _a[0]: self._X_normal, y_migration: self._Y_normal}))
            print('验证集假阳性率', sess.run(fpr, feed_dict={
                                _a[0]: self._X_migration_test, y_migration: self._Y_migration_test}))

            print('3月2日测试集假阳性率', sess.run(fpr, feed_dict={
                _a[0]: self._X_data_migration_test, y_migration: self._Y_data_migration_test}))

            x_data_set = pd.DataFrame(data=lo, index=None, columns=None)  # 这种方法默认行与列索引为0，1，2等
            x_data_set.to_csv('F:/python代码/迁移学习/GBRBM/迁移rbm后加隐藏层/loss11.csv', encoding="utf-8")  # 读入csv文件




            plt.plot(sample, lo,  linewidth=1, linestyle="--", color="red")
            plt.title("fine_turn_The variation of the loss")
            plt.xlabel("Sampling Point")
            plt.ylabel("Loss")
            plt.grid(True)
            plt.show()

            plt.plot(sample, lce,  linewidth=1, linestyle="--", color="red")
            plt.plot(sample, lyan,  linewidth=1, linestyle="--", color="blue")
            plt.title("fine_turn_The acc of the train set and verification set")
            plt.xlabel("Sampling Point")
            plt.ylabel("acc")
            plt.grid(True)
            plt.show()


    def train_migration_all(self):                                      #开放所有层，以很小的学习率再次微调
        _a = [None] * (len(self._sizes) + 3)
        _w = [None] * len(self.w_list)
        _b = [None] * len(self.b_list)

        _w_add = [None] * len(self.w_add_list)
        _b_add = [None] * len(self.b_add_list)
        _f1 = [None]
        _f2 = [None]

        _a[0] = tf.placeholder("float", [None, self._X.shape[1]])
        y_model = tf.placeholder("float", [None, self._Y_normal.shape[1]])
        for i in range(len(self._sizes) + 2):
            #print("w b  len")
            #print(i)  # 0  1  2  3
            _w[i] = tf.Variable(self.w_list[i])
            _b[i] = tf.Variable(self.b_list[i])

        for i in range(len(self._size_add)+1):
            _w_add[i] = tf.Variable(self.w_add_list[i])
            _b_add[i] = tf.Variable(self.b_add_list[i])

        for i in range(1, len(self._sizes)+1):
            #print('_a  len')
            #print(i)  # 1  2  3
            _a[i] = tf.nn.sigmoid(tf.matmul(_a[i - 1], _w[i - 1]) + _b[i - 1])

        _a[4] = tf.matmul(_a[3], _w[3]) + _b[3]
        batch_norm5 = bn_layer(_a[4], 1)
        _a[4] = tf.nn.sigmoid(batch_norm5.batch_norm_layer())

        _a[5] = tf.matmul(_a[4], _w[4]) + _b[4]
        batch_norm6 = bn_layer(_a[5], 1)
        _a[5] = tf.nn.sigmoid(batch_norm6.batch_norm_layer())

        _f1 = tf.matmul(_a[5], _w_add[0]) + _b_add[0]
        batch_norm4 = bn_layer(_f1, 1)
        _f1 = tf.nn.sigmoid(batch_norm4.batch_norm_layer())

        _f2 = tf.nn.softmax(tf.matmul(_f1, _w_add[1]) + _b_add[1])

        cost_migration = tf.reduce_mean(-tf.reduce_sum(y_model * tf.log(_f2), 1))

        train_op_migration = tf.train.GradientDescentOptimizer(self._opts._learning_rate_migration_min).minimize(
            cost_migration)

        predict_op_migration = tf.argmax(_f2, 1)
        actuals = tf.argmax(y_model, 1)

        ones_like_actuals = tf.ones_like(actuals)
        zeros_like_actuals = tf.zeros_like(actuals)
        ones_like_predictions = tf.ones_like(predict_op_migration)
        zeros_like_predictions = tf.zeros_like(predict_op_migration)

        tn = tf.reduce_sum(tf.cast(
            tf.logical_and(tf.equal(actuals, ones_like_actuals),
                           tf.equal(predict_op_migration, ones_like_predictions)),
            "float"))

        fp = tf.reduce_sum(
            tf.cast(tf.logical_and(tf.equal(actuals, ones_like_actuals),
                                   tf.equal(predict_op_migration, zeros_like_predictions)),
                    "float"))
        fpr = tf.divide(fp, tf.add(fp, tn), name='prob2')

        with tf.Session() as sess:
            lo_all =[]
            lce_all =[]
            lyan_all = []
            sess.run(tf.initialize_all_variables())
            sample = np.arange(self._opts._epoches_migration_all)
            train_max = 0
            train_i = 0
            yanzhen_max = 0
            yanzhen_i = 0
            yanzhen_32_max = 0
            for i in range(self._opts._epoches_migration_all):
                for start, end in zip(range(0, len(self._X_normal), self._opts._batchsize_migration),
                                      range(self._opts._batchsize_migration, len(self._X_normal)+self._opts._batchsize_migration, self._opts._batchsize_migration)):
                    #print(start,end)
                    sess.run(train_op_migration, feed_dict={ _a[0]: self._X_normal[start:end], y_model: self._Y_normal[start:end]})

                l=sess.run(cost_migration, feed_dict={_a[0]: self._X_normal[start:end], y_model: self._Y_normal[start:end]})
                lo_all.append(l)
                #  print(len(self._X))  55000

                self.w_add_list[0] = sess.run(_w_add[0])
                self.b_add_list[0] = sess.run(_b_add[0])
                self.w_add_list[1] = sess.run(_w_add[1])
                self.b_add_list[1] = sess.run(_b_add[1])


                # if i % 100 == 0:
                #     print("迁移后训练集的准确率："+ "在第"+str(i)+"代"+str(np.mean(np.argmax(self._Y_normal, axis=1) ==
                #               sess.run(predict_op_migration, feed_dict={
                #                   _a[0]: self._X_normal, y_migration: self._Y_normal}))))
                #
                #     print("迁移后验证集的准确率：" + "在第"+str(i)+"代"+str(np.mean(np.argmax(self._Y_migration_test, axis=1) ==
                #             sess.run(predict_op_migration, feed_dict={
                #                 _a[0]: self._X_migration_test, y_migration: self._Y_migration_test}))))


                ce=  np.mean(np.argmax(self._Y_normal, axis=1) ==
                              sess.run(predict_op_migration, feed_dict={
                                  _a[0]: self._X_normal, y_model: self._Y_normal}))
                lce_all.append(ce)

                yan = np.mean(np.argmax(self._Y_migration_test, axis=1) ==
                            sess.run(predict_op_migration, feed_dict={
                                _a[0]: self._X_migration_test, y_model: self._Y_migration_test}))
                lyan_all.append(yan)


                yan_3_2 = np.mean(np.argmax(self._Y_data_migration_test, axis=1) ==
                            sess.run(predict_op_migration, feed_dict={
                                _a[0]: self._X_data_migration_test, y_model: self._Y_data_migration_test}))

                if ce > train_max:
                    train_max = ce
                    train_i = i
                    print("迁移后   训练集  当前最优准确率：" + "在第" + str(train_i) + " (epoches) " + str(train_max))

                if yan > yanzhen_max:
                    yanzhen_max = yan
                    yanzhen_i = i
                    print("迁移后   验证集  当前最优准确率：" + "在第" + str(yanzhen_i) + " (epoches) " + str(yanzhen_max))

                if yan_3_2 > yanzhen_32_max:
                    yanzhen_32_max = yan_3_2
                    yanzhen_32_i = i
                    print("迁移后   验证集2月2  当前最优准确率：" + "在第" + str(yanzhen_32_i) + " (epoches) " + str(yanzhen_32_max))

            print('训练集假阳性率', sess.run(fpr, feed_dict={
                                  _a[0]: self._X_normal, y_model: self._Y_normal}))
            print('验证集假阳性率', sess.run(fpr,feed_dict={
                                _a[0]: self._X_migration_test, y_model: self._Y_migration_test}))

            print('3月2日测试集假阳性率', sess.run(fpr, feed_dict={
                _a[0]: self._X_data_migration_test, y_model: self._Y_data_migration_test}))


            x_data_set = pd.DataFrame(data=lo_all, index=None, columns=None)  # 这种方法默认行与列索引为0，1，2等
            x_data_set.to_csv('F:/python代码/迁移学习/GBRBM/迁移rbm后加隐藏层/loss12.csv', encoding="utf-8")  # 读入csv文件


            plt.plot(sample, lo_all,  linewidth=1, linestyle="--", color="red")
            plt.title("fine_turn_The variation of the loss")
            plt.xlabel("Sampling Point")
            plt.ylabel("Loss")
            plt.grid(True)
            plt.show()

            plt.plot(sample, lce_all,  linewidth=1, linestyle="--", color="red")
            plt.plot(sample, lyan_all,  linewidth=1, linestyle="--", color="blue")
            plt.title("fine_turn_The acc of the train set and verification set")
            plt.xlabel("Sampling Point")
            plt.ylabel("acc")
            plt.grid(True)
            plt.show()