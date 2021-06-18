
import tensorflow as tf
import numpy as np

class  bn_layer(object):
   def __init__(self,x1, train_phase):
       self.x1 = x1
       self.train_phase = train_phase



   def batch_norm_layer(self):
      with tf.variable_scope('V1'):
        # 新建两个变量，平移、缩放因子
        beta = tf.Variable(tf.constant(0.0, shape=[self.x1.shape[-1]]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[self.x1.shape[-1]]), name='gamma', trainable=True)

        # 计算此次批量的均值和方差
        axises = np.arange(len(self.x1.shape) - 1)
        batch_mean, batch_var = tf.nn.moments(self.x1, [0], name='moments')

        # 滑动平均做衰减
        ema = tf.train.ExponentialMovingAverage(decay=0.5)


        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
                # train_phase 训练还是测试的flag
                # 训练阶段计算runing_mean和runing_var，使用mean_var_with_update（）函数
                # 测试的时候直接把之前计算的拿去用 ema.average(batch_mean)
        # mean, var = tf.cond(self.train_phase, mean_var_with_update,lambda: (ema.average(batch_mean), ema.average(batch_var)))
        if self.train_phase ==1:
            mean,var = mean_var_with_update()
        else:
            mean= ema.average(batch_mean)
            var = ema.average(batch_var)


        normed = tf.nn.batch_normalization(self.x1, mean, var, beta, gamma, 1e-3)

      return normed
