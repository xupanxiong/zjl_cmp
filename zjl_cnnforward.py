'''
author:xupanxiong@qq.com
cnnforward-file, build the network
'''

import tensorflow as tf

NUM_CHANNELS = 3
OUTPUT_NODE = 2

def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b


def conv2d(x,w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def forward(x, train, regularizer):
    #构建网络
    with tf.variable_scope("conv1_1"):
        kernel1_1 = tf.Variable(tf.truncated_normal([3, 3, 3, 64], mean=0.0, stddev=1.0, dtype=tf.float32))
        conv1_1 = tf.nn.conv2d(x, kernel1_1, [1,1,1,1], padding="SAME", name="CONV1_1")
        print("conv1_1",conv1_1)
        bias1_1 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[64], name="BIAS1_1"))
        conv1_1 = tf.nn.bias_add(conv1_1, bias1_1)
        conv1_1 = tf.nn.relu(conv1_1)

    with tf.variable_scope("conv1_2"):
        kernel1_2 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], mean=0.0, stddev=1.0, dtype=tf.float32))
        conv1_2 = tf.nn.conv2d(conv1_1, kernel1_2, [1,1,1,1], padding="SAME", name="CONV1_2")
        print("conv1_2", conv1_2)
        bias1_2 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[64], name="BIAS1_2"))
        conv1_2 = tf.nn.bias_add(conv1_2, bias1_2)
        conv1_2 = tf.nn.relu(conv1_2)

    maxpool1 = tf.nn.max_pool(conv1_2, [1,2,2,1],[1,2,2,1],padding="SAME",name="maxpool1")
    print("maxpool1",maxpool1)

    with tf.variable_scope("conv2_1"):
        kernel2_1 = tf.Variable(tf.truncated_normal([3, 3, 64, 128], mean=0.0, stddev=1.0, dtype=tf.float32))
        conv2_1 = tf.nn.conv2d(maxpool1, kernel2_1, [1,1,1,1], padding="SAME", name="CONV2_1")
        print("conv2_1", conv2_1)
        bias2_1 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[128], name="BIAS2_1"))
        conv2_1 = tf.nn.bias_add(conv2_1, bias2_1)
        conv2_1 = tf.nn.relu(conv2_1)

    with tf.variable_scope("conv2_2"):
        kernel2_2 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], mean=0.0, stddev=1.0, dtype=tf.float32))
        conv2_2 = tf.nn.conv2d(conv2_1, kernel2_2, [1,1,1,1], padding="SAME", name="CONV2_2")
        print("conv2_2", conv2_2)
        bias2_2 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[128], name="BIAS2_2"))
        conv2_2 = tf.nn.bias_add(conv2_2, bias2_2)
        conv2_2 = tf.nn.relu(conv2_2)

    maxpool2 = tf.nn.max_pool(conv2_2, [1,2,2,1],[1,2,2,1],padding="SAME",name="maxpool2")
    print("maxpool2", maxpool2)

    with tf.variable_scope("conv3_1"):
        kernel3_1 = tf.Variable(tf.truncated_normal([3,3,128,256], 0.0, 1.0, dtype=tf.float32))
        conv3_1 = tf.nn.conv2d(maxpool2, kernel3_1, [1,1,1,1], padding="SAME", name="CONV3_1")
        print("conv3_1", conv3_1)
        bias3_1 = tf.Variable(tf.constant(0.0, dtype=tf.float32,shape=[256], name="BIAS3_1"))
        conv3_1 = tf.nn.bias_add(conv3_1, bias3_1)
        conv3_1 = tf.nn.relu(conv3_1)

    with tf.variable_scope("conv3_2"):
        kernel3_2 = tf.Variable(tf.truncated_normal([3,3,256,256],mean=0.0, stddev=1.0,dtype=tf.float32))
        conv3_2 = tf.nn.conv2d(conv3_1, kernel3_2, [1,1,1,1], padding="SAME", name="CONV3_2")
        print("conv3_2", conv3_2)
        bias3_2 = tf.Variable(tf.constant(0.0, dtype=tf.float32,shape=[256],name="BIAS3_2"))
        conv3_2 = tf.nn.bias_add(conv3_2,bias3_2)
        conv3_2 = tf.nn.relu(conv3_2)

    with tf.variable_scope("conv3_3"):
        kernel3_3 = tf.Variable(tf.truncated_normal([3, 3, 256, 256], mean=0.0, stddev=1.0, dtype=tf.float32))
        conv3_3 = tf.nn.conv2d(conv3_2, kernel3_3, [1, 1, 1, 1], padding="SAME", name="CONV3_3")
        print("conv3_3", conv3_3)
        bias3_3 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[256], name="BIAS3_3"))
        conv3_3 = tf.nn.bias_add(conv3_3, bias3_3)
        conv3_3 = tf.nn.relu(conv3_3)

    maxpool3 = tf.nn.max_pool(conv3_3, [1,2,2,1],[1,2,2,1],padding="SAME",name="maxpool3")
    print("maxpool3", maxpool3)

    with tf.variable_scope("conv4_1"):
        kernel4_1 = tf.Variable(tf.truncated_normal([3,3,256,512], 0.0, 1.0, dtype=tf.float32))
        conv4_1 = tf.nn.conv2d(maxpool3, kernel4_1, [1,1,1,1], padding="SAME", name="CONV4_1")
        print("conv4_1", conv4_1)
        bias4_1 = tf.Variable(tf.constant(0.0, dtype=tf.float32,shape=[512], name="BIAS4_1"))
        conv4_1 = tf.nn.bias_add(conv4_1, bias4_1)
        conv4_1 = tf.nn.relu(conv4_1)

    with tf.variable_scope("conv4_2"):
        kernel4_2 = tf.Variable(tf.truncated_normal([3,3,512,512],mean=0.0, stddev=1.0,dtype=tf.float32))
        conv4_2 = tf.nn.conv2d(conv4_1, kernel4_2, [1,1,1,1], padding="SAME", name="CONV4_2")
        print("conv4_2", conv4_2)
        bias4_2 = tf.Variable(tf.constant(0.0, dtype=tf.float32,shape=[512],name="BIAS4_2"))
        conv4_2 = tf.nn.bias_add(conv4_2,bias4_2)
        conv4_2 = tf.nn.relu(conv4_2)

    with tf.variable_scope("conv4_3"):
        kernel4_3 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], mean=0.0, stddev=1.0, dtype=tf.float32))
        conv4_3 = tf.nn.conv2d(conv4_2, kernel4_3, [1, 1, 1, 1], padding="SAME", name="CONV4_3")
        print("conv4_3", conv4_3)
        bias4_3 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[512], name="BIAS4_3"))
        conv4_3 = tf.nn.bias_add(conv4_3, bias4_3)
        conv4_3 = tf.nn.relu(conv4_3)

    maxpool4 = tf.nn.max_pool(conv4_3, [1,2,2,1],[1,2,2,1],padding="SAME",name="maxpool4")
    print("maxpool4", maxpool4)

    with tf.variable_scope("conv5_1"):
        kernel5_1 = tf.Variable(tf.truncated_normal([3,3,512,512], 0.0, 1.0, dtype=tf.float32))
        conv5_1 = tf.nn.conv2d(maxpool4, kernel5_1, [1,1,1,1], padding="SAME", name="CONV5_1")
        print("conv5_1", conv5_1)
        bias5_1 = tf.Variable(tf.constant(0.0, dtype=tf.float32,shape=[512], name="BIAS5_1"))
        conv5_1 = tf.nn.bias_add(conv5_1, bias5_1)
        conv5_1 = tf.nn.relu(conv5_1)

    with tf.variable_scope("conv5_2"):
        kernel5_2 = tf.Variable(tf.truncated_normal([3,3,512,512],mean=0.0, stddev=1.0,dtype=tf.float32))
        conv5_2 = tf.nn.conv2d(conv5_1, kernel5_2, [1,1,1,1], padding="SAME", name="CONV5_2")
        print("conv5_2", conv5_2)
        bias5_2 = tf.Variable(tf.constant(0.0, dtype=tf.float32,shape=[512],name="BIAS5_2"))
        conv5_2 = tf.nn.bias_add(conv5_2,bias5_2)
        conv5_2 = tf.nn.relu(conv5_2)

    with tf.variable_scope("conv5_3"):
        kernel5_3 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], mean=0.0, stddev=1.0, dtype=tf.float32))
        conv5_3 = tf.nn.conv2d(conv5_2, kernel5_3, [1, 1, 1, 1], padding="SAME", name="CONV5_3")
        print("conv5_3", conv5_3)
        bias5_3 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[512], name="BIAS5_3"))
        conv5_3 = tf.nn.bias_add(conv5_3, bias5_3)
        conv5_3 = tf.nn.relu(conv5_3)

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
            tf.truncated_normal([256, OUTPUT_NODE], mean=0.0, stddev=1.0, dtype=tf.float32, name="fc8_Weight"))
        fc8_bias = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[OUTPUT_NODE], name="fc8_bias"))

        if train:
            fc8 = tf.matmul(fc7_drop, fc8_weight)
            fc8 = tf.nn.bias_add(fc8, fc8_bias)
        else:
            fc8 = tf.matmul(fc7,fc8_weight)
        #fc8 = tf.nn.relu(fc8)
    return fc8

