'''
author:xupanxiong@qq.com
cnnforward-file, build the network -- add bn
'''

import tensorflow as tf
import zjl_config as zjlconf
IMAGE_WIDTH = zjlconf.IMAGE_WIDTH
IMAGE_HIGH = zjlconf.IMAGE_HIGH
NUM_CHANNELS = 3
OUTPUT_NODE = 2

def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b

#用于批标准化
def mean_var_with_update(ema, fc_mean, fc_var):
    ema_apply_op = ema.apply([fc_mean, fc_var])
    with tf.control_dependencies([ema_apply_op]):
        return tf.identity(fc_mean), tf.identity(fc_var)

def conv2d(x,w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# #带批标准化的卷积层
# def Conv2d_BN(scope_name, inputs, filter_size, in_channels, output_channels, strides_moves, padding_mode, activation):
#     with tf.variable_scope(scope_name) as scope:
#         kernel = get_weight(shape=[filter_size, filter_size, in_channels, output_channels],regularizer=None)
#         conv = tf.nn.conv2d(inputs, kernel, strides=[1, strides_moves, strides_moves, 1], padding=padding_mode)
#         biases = get_bias(shape=[output_channels])
#         pre_activation = tf.nn.bias_add(conv, biases)
#         fc_mean, fc_var = tf.nn.moments(pre_activation, axes=[0, 1, 2])
#         scale = tf.Variable(tf.ones([output_channels]))
#         shift = tf.Variable(tf.zeros([output_channels]))
#         epsilon = 0.001
#         ema = tf.train.ExponentialMovingAverage(decay=0.5)
#         # update mean and var when the value of mode is TRAIN, or back to the previous Moving Average of fc_mean and fc_var
#         mean, var = tf.cond(train_mode, lambda: (mean_var_with_update(ema, fc_mean, fc_var)), lambda: (ema.average(fc_mean), ema.average(fc_var)))
#         pre_activation_with_BN = tf.nn.batch_normalization(pre_activation, mean, var, shift, scale, epsilon)
#     return activation(pre_activation_with_BN, name=scope.name)

def forward(x, train, regularizer):
    #构建网络
    with tf.variable_scope("conv1_1"):
        kernel1_1 = tf.Variable(tf.truncated_normal([3, 3, 3, 64], mean=0.0, stddev=1.0, dtype=tf.float32))
        conv1_1 = tf.nn.conv2d(x, kernel1_1, [1,1,1,1], padding="SAME", name="CONV1_1")
        print("conv1_1",conv1_1)
        bias1_1 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[64], name="BIAS1_1"))
        conv1_1 = tf.nn.bias_add(conv1_1, bias1_1)
        #加入批标准化
        fc_mean, fc_var = tf.nn.moments(conv1_1, axes=[0, 1, 2])
        scale = tf.Variable(tf.ones([64]))
        shift = tf.Variable(tf.zeros([64]))
        epsilon = 0.001
        ema = tf.train.ExponentialMovingAverage(decay=0.5)
        #print(type(train))
        if train:
            mean,var = mean_var_with_update(ema, fc_mean, fc_var)
        else:
            mean = ema.average(fc_mean)
            var = ema.average(fc_var)
        pre_activation_with_BN = tf.nn.batch_normalization(conv1_1, mean, var, shift, scale, epsilon)
        conv1_1 = tf.nn.relu(pre_activation_with_BN)

    with tf.variable_scope("conv1_2"):
        kernel1_2 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], mean=0.0, stddev=1.0, dtype=tf.float32))
        conv1_2 = tf.nn.conv2d(conv1_1, kernel1_2, [1,1,1,1], padding="SAME", name="CONV1_2")
        print("conv1_2", conv1_2)
        bias1_2 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[64], name="BIAS1_2"))
        conv1_2 = tf.nn.bias_add(conv1_2, bias1_2)
        # 加入批标准化
        fc_mean, fc_var = tf.nn.moments(conv1_2, axes=[0, 1, 2])
        scale = tf.Variable(tf.ones([64]))
        shift = tf.Variable(tf.zeros([64]))
        epsilon = 0.001
        ema = tf.train.ExponentialMovingAverage(decay=0.5)
        if train:
            mean,var = mean_var_with_update(ema, fc_mean, fc_var)
        else:
            mean = ema.average(fc_mean)
            var = ema.average(fc_var)
        pre_activation_with_BN = tf.nn.batch_normalization(conv1_2, mean, var, shift, scale, epsilon)
        conv1_2 = tf.nn.relu(pre_activation_with_BN)

    maxpool1 = tf.nn.max_pool(conv1_2, [1,2,2,1],[1,2,2,1],padding="SAME",name="maxpool1")
    print("maxpool1",maxpool1)

    with tf.variable_scope("conv2_1"):
        kernel2_1 = tf.Variable(tf.truncated_normal([3, 3, 64, 128], mean=0.0, stddev=1.0, dtype=tf.float32))
        conv2_1 = tf.nn.conv2d(maxpool1, kernel2_1, [1,1,1,1], padding="SAME", name="CONV2_1")
        print("conv2_1", conv2_1)
        bias2_1 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[128], name="BIAS2_1"))
        conv2_1 = tf.nn.bias_add(conv2_1, bias2_1)
        # 加入批标准化
        fc_mean, fc_var = tf.nn.moments(conv2_1, axes=[0, 1, 2])
        scale = tf.Variable(tf.ones([128]))
        shift = tf.Variable(tf.zeros([128]))
        epsilon = 0.001
        ema = tf.train.ExponentialMovingAverage(decay=0.5)
        if train:
            mean,var = mean_var_with_update(ema, fc_mean, fc_var)
        else:
            mean = ema.average(fc_mean)
            var = ema.average(fc_var)
        pre_activation_with_BN = tf.nn.batch_normalization(conv2_1, mean, var, shift, scale, epsilon)
        conv2_1 = tf.nn.relu(pre_activation_with_BN)

    with tf.variable_scope("conv2_2"):
        kernel2_2 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], mean=0.0, stddev=1.0, dtype=tf.float32))
        conv2_2 = tf.nn.conv2d(conv2_1, kernel2_2, [1,1,1,1], padding="SAME", name="CONV2_2")
        print("conv2_2", conv2_2)
        bias2_2 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[128], name="BIAS2_2"))
        conv2_2 = tf.nn.bias_add(conv2_2, bias2_2)
        # 加入批标准化
        fc_mean, fc_var = tf.nn.moments(conv2_2, axes=[0, 1, 2])
        scale = tf.Variable(tf.ones([128]))
        shift = tf.Variable(tf.zeros([128]))
        epsilon = 0.001
        ema = tf.train.ExponentialMovingAverage(decay=0.5)
        if train:
            mean,var = mean_var_with_update(ema, fc_mean, fc_var)
        else:
            mean = ema.average(fc_mean)
            var = ema.average(fc_var)
        pre_activation_with_BN = tf.nn.batch_normalization(conv2_2, mean, var, shift, scale, epsilon)
        conv2_2 = tf.nn.relu(pre_activation_with_BN)

    maxpool2 = tf.nn.max_pool(conv2_2, [1,2,2,1],[1,2,2,1],padding="SAME",name="maxpool2")
    print("maxpool2", maxpool2)

    with tf.variable_scope("conv3_1"):
        kernel3_1 = tf.Variable(tf.truncated_normal([3,3,128,256], 0.0, 1.0, dtype=tf.float32))
        conv3_1 = tf.nn.conv2d(maxpool2, kernel3_1, [1,1,1,1], padding="SAME", name="CONV3_1")
        print("conv3_1", conv3_1)
        bias3_1 = tf.Variable(tf.constant(0.0, dtype=tf.float32,shape=[256], name="BIAS3_1"))
        conv3_1 = tf.nn.bias_add(conv3_1, bias3_1)
        # 加入批标准化
        fc_mean, fc_var = tf.nn.moments(conv3_1, axes=[0, 1, 2])
        scale = tf.Variable(tf.ones([256]))
        shift = tf.Variable(tf.zeros([256]))
        epsilon = 0.001
        ema = tf.train.ExponentialMovingAverage(decay=0.5)
        if train:
            mean,var = mean_var_with_update(ema, fc_mean, fc_var)
        else:
            mean = ema.average(fc_mean)
            var = ema.average(fc_var)
        pre_activation_with_BN = tf.nn.batch_normalization(conv3_1, mean, var, shift, scale, epsilon)
        conv3_1 = tf.nn.relu(pre_activation_with_BN)

    with tf.variable_scope("conv3_2"):
        kernel3_2 = tf.Variable(tf.truncated_normal([3,3,256,256],mean=0.0, stddev=1.0,dtype=tf.float32))
        conv3_2 = tf.nn.conv2d(conv3_1, kernel3_2, [1,1,1,1], padding="SAME", name="CONV3_2")
        print("conv3_2", conv3_2)
        bias3_2 = tf.Variable(tf.constant(0.0, dtype=tf.float32,shape=[256],name="BIAS3_2"))
        conv3_2 = tf.nn.bias_add(conv3_2,bias3_2)
        # 加入批标准化
        fc_mean, fc_var = tf.nn.moments(conv3_2, axes=[0, 1, 2])
        scale = tf.Variable(tf.ones([256]))
        shift = tf.Variable(tf.zeros([256]))
        epsilon = 0.001
        ema = tf.train.ExponentialMovingAverage(decay=0.5)
        if train:
            mean,var = mean_var_with_update(ema, fc_mean, fc_var)
        else:
            mean = ema.average(fc_mean)
            var = ema.average(fc_var)
        pre_activation_with_BN = tf.nn.batch_normalization(conv3_2, mean, var, shift, scale, epsilon)
        conv3_2 = tf.nn.relu(pre_activation_with_BN)

    with tf.variable_scope("conv3_3"):
        kernel3_3 = tf.Variable(tf.truncated_normal([3, 3, 256, 256], mean=0.0, stddev=1.0, dtype=tf.float32))
        conv3_3 = tf.nn.conv2d(conv3_2, kernel3_3, [1, 1, 1, 1], padding="SAME", name="CONV3_3")
        print("conv3_3", conv3_3)
        bias3_3 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[256], name="BIAS3_3"))
        conv3_3 = tf.nn.bias_add(conv3_3, bias3_3)
        # 加入批标准化
        fc_mean, fc_var = tf.nn.moments(conv3_3, axes=[0, 1, 2])
        scale = tf.Variable(tf.ones([256]))
        shift = tf.Variable(tf.zeros([256]))
        epsilon = 0.001
        ema = tf.train.ExponentialMovingAverage(decay=0.5)
        if train:
            mean,var = mean_var_with_update(ema, fc_mean, fc_var)
        else:
            mean = ema.average(fc_mean)
            var = ema.average(fc_var)
        pre_activation_with_BN = tf.nn.batch_normalization(conv3_3, mean, var, shift, scale, epsilon)
        conv3_3 = tf.nn.relu(pre_activation_with_BN)

    maxpool3 = tf.nn.max_pool(conv3_3, [1,2,2,1],[1,2,2,1],padding="SAME",name="maxpool3")
    print("maxpool3", maxpool3)

    with tf.variable_scope("conv4_1"):
        kernel4_1 = tf.Variable(tf.truncated_normal([3,3,256,512], 0.0, 1.0, dtype=tf.float32))
        conv4_1 = tf.nn.conv2d(maxpool3, kernel4_1, [1,1,1,1], padding="SAME", name="CONV4_1")
        print("conv4_1", conv4_1)
        bias4_1 = tf.Variable(tf.constant(0.0, dtype=tf.float32,shape=[512], name="BIAS4_1"))
        conv4_1 = tf.nn.bias_add(conv4_1, bias4_1)
        # 加入批标准化
        fc_mean, fc_var = tf.nn.moments(conv4_1, axes=[0, 1, 2])
        scale = tf.Variable(tf.ones([512]))
        shift = tf.Variable(tf.zeros([512]))
        epsilon = 0.001
        ema = tf.train.ExponentialMovingAverage(decay=0.5)
        if train:
            mean,var = mean_var_with_update(ema, fc_mean, fc_var)
        else:
            mean = ema.average(fc_mean)
            var = ema.average(fc_var)
        pre_activation_with_BN = tf.nn.batch_normalization(conv4_1, mean, var, shift, scale, epsilon)
        conv4_1 = tf.nn.relu(pre_activation_with_BN)

    with tf.variable_scope("conv4_2"):
        kernel4_2 = tf.Variable(tf.truncated_normal([3,3,512,512],mean=0.0, stddev=1.0,dtype=tf.float32))
        conv4_2 = tf.nn.conv2d(conv4_1, kernel4_2, [1,1,1,1], padding="SAME", name="CONV4_2")
        print("conv4_2", conv4_2)
        bias4_2 = tf.Variable(tf.constant(0.0, dtype=tf.float32,shape=[512],name="BIAS4_2"))
        conv4_2 = tf.nn.bias_add(conv4_2,bias4_2)
        # 加入批标准化
        fc_mean, fc_var = tf.nn.moments(conv4_2, axes=[0, 1, 2])
        scale = tf.Variable(tf.ones([512]))
        shift = tf.Variable(tf.zeros([512]))
        epsilon = 0.001
        ema = tf.train.ExponentialMovingAverage(decay=0.5)
        if train:
            mean,var = mean_var_with_update(ema, fc_mean, fc_var)
        else:
            mean = ema.average(fc_mean)
            var = ema.average(fc_var)
        pre_activation_with_BN = tf.nn.batch_normalization(conv4_2, mean, var, shift, scale, epsilon)
        conv4_2 = tf.nn.relu(pre_activation_with_BN)

    with tf.variable_scope("conv4_3"):
        kernel4_3 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], mean=0.0, stddev=1.0, dtype=tf.float32))
        conv4_3 = tf.nn.conv2d(conv4_2, kernel4_3, [1, 1, 1, 1], padding="SAME", name="CONV4_3")
        print("conv4_3", conv4_3)
        bias4_3 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[512], name="BIAS4_3"))
        conv4_3 = tf.nn.bias_add(conv4_3, bias4_3)
        # 加入批标准化
        fc_mean, fc_var = tf.nn.moments(conv4_3, axes=[0, 1, 2])
        scale = tf.Variable(tf.ones([512]))
        shift = tf.Variable(tf.zeros([512]))
        epsilon = 0.001
        ema = tf.train.ExponentialMovingAverage(decay=0.5)
        if train:
            mean,var = mean_var_with_update(ema, fc_mean, fc_var)
        else:
            mean = ema.average(fc_mean)
            var = ema.average(fc_var)
        pre_activation_with_BN = tf.nn.batch_normalization(conv4_3, mean, var, shift, scale, epsilon)
        conv4_3 = tf.nn.relu(pre_activation_with_BN)

    maxpool4 = tf.nn.max_pool(conv4_3, [1,2,2,1],[1,2,2,1],padding="SAME",name="maxpool4")
    print("maxpool4", maxpool4)

    with tf.variable_scope("conv5_1"):
        kernel5_1 = tf.Variable(tf.truncated_normal([3,3,512,512], 0.0, 1.0, dtype=tf.float32))
        conv5_1 = tf.nn.conv2d(maxpool4, kernel5_1, [1,1,1,1], padding="SAME", name="CONV5_1")
        print("conv5_1", conv5_1)
        bias5_1 = tf.Variable(tf.constant(0.0, dtype=tf.float32,shape=[512], name="BIAS5_1"))
        conv5_1 = tf.nn.bias_add(conv5_1, bias5_1)
        # 加入批标准化
        fc_mean, fc_var = tf.nn.moments(conv5_1, axes=[0, 1, 2])
        scale = tf.Variable(tf.ones([512]))
        shift = tf.Variable(tf.zeros([512]))
        epsilon = 0.001
        ema = tf.train.ExponentialMovingAverage(decay=0.5)
        if train:
            mean,var = mean_var_with_update(ema, fc_mean, fc_var)
        else:
            mean = ema.average(fc_mean)
            var = ema.average(fc_var)
        pre_activation_with_BN = tf.nn.batch_normalization(conv5_1, mean, var, shift, scale, epsilon)
        conv5_1 = tf.nn.relu(pre_activation_with_BN)

    with tf.variable_scope("conv5_2"):
        kernel5_2 = tf.Variable(tf.truncated_normal([3,3,512,512],mean=0.0, stddev=1.0,dtype=tf.float32))
        conv5_2 = tf.nn.conv2d(conv5_1, kernel5_2, [1,1,1,1], padding="SAME", name="CONV5_2")
        print("conv5_2", conv5_2)
        bias5_2 = tf.Variable(tf.constant(0.0, dtype=tf.float32,shape=[512],name="BIAS5_2"))
        conv5_2 = tf.nn.bias_add(conv5_2,bias5_2)
        # 加入批标准化
        fc_mean, fc_var = tf.nn.moments(conv5_2, axes=[0, 1, 2])
        scale = tf.Variable(tf.ones([512]))
        shift = tf.Variable(tf.zeros([512]))
        epsilon = 0.001
        ema = tf.train.ExponentialMovingAverage(decay=0.5)
        if train:
            mean,var = mean_var_with_update(ema, fc_mean, fc_var)
        else:
            mean = ema.average(fc_mean)
            var = ema.average(fc_var)
        pre_activation_with_BN = tf.nn.batch_normalization(conv5_2, mean, var, shift, scale, epsilon)
        conv5_2 = tf.nn.relu(pre_activation_with_BN)

    with tf.variable_scope("conv5_3"):
        kernel5_3 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], mean=0.0, stddev=1.0, dtype=tf.float32))
        conv5_3 = tf.nn.conv2d(conv5_2, kernel5_3, [1, 1, 1, 1], padding="SAME", name="CONV5_3")
        print("conv5_3", conv5_3)
        bias5_3 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[512], name="BIAS5_3"))
        conv5_3 = tf.nn.bias_add(conv5_3, bias5_3)
        # 加入批标准化
        fc_mean, fc_var = tf.nn.moments(conv5_3, axes=[0, 1, 2])
        scale = tf.Variable(tf.ones([512]))
        shift = tf.Variable(tf.zeros([512]))
        epsilon = 0.001
        ema = tf.train.ExponentialMovingAverage(decay=0.5)
        if train:
            mean,var = mean_var_with_update(ema, fc_mean, fc_var)
        else:
            mean = ema.average(fc_mean)
            var = ema.average(fc_var)
        pre_activation_with_BN = tf.nn.batch_normalization(conv5_3, mean, var, shift, scale, epsilon)
        conv5_3 = tf.nn.relu(pre_activation_with_BN)

    maxpool5 = tf.nn.max_pool(conv5_3, [1,2,2,1],[1,2,2,1],padding="SAME",name="maxpool5")
    print("maxpool5", maxpool5)

    shape = maxpool5.get_shape()

    length = shape[1].value * shape[2].value * shape[3].value

    reshape = tf.reshape(maxpool5, [-1, length], name="reshape")

    with tf.variable_scope("fc6"):
        fc6_weight = tf.Variable(
            tf.truncated_normal([length, 2048], mean=0.0, stddev=1.0, dtype=tf.float32, name="fc6_Weight"))
        fc6_bias = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[2048], name="fc6_bias"))
        fc6 = tf.matmul(reshape, fc6_weight)
        fc6 = tf.nn.bias_add(fc6, fc6_bias)
        fc6 = tf.nn.relu(fc6)

    if train: fc6_drop = tf.nn.dropout(fc6, 0.5, name="fc6_drop")

    with tf.variable_scope("fc7"):
        fc7_weight = tf.Variable(
            tf.truncated_normal([2048, 256], mean=0.0, stddev=1.0, dtype=tf.float32, name="fc7_Weight"))
        fc7_bias = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[256], name="fc7_bias"))
        if train:
            fc7 = tf.matmul(fc6_drop, fc7_weight)
            fc7 = tf.nn.bias_add(fc7, fc7_bias)
        else:
            fc7 = tf.matmul(fc6,fc7_weight)
        fc7 = tf.nn.relu(fc7)

    if train: fc7_drop = tf.nn.dropout(fc7, 0.5, name="fc7_drop")

    with tf.variable_scope("fc8"):
        fc8_weight = tf.Variable(
            tf.truncated_normal([256, 2], mean=0.0, stddev=1.0, dtype=tf.float32, name="fc8_Weight"))
        fc8_bias = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[2], name="fc8_bias"))

        if train:
            fc8 = tf.matmul(fc7_drop, fc8_weight)
            fc8 = tf.nn.bias_add(fc8, fc8_bias)
        else:
            fc8 = tf.matmul(fc7,fc8_weight)
        #fc8 = tf.nn.relu(fc8)

    return fc8

