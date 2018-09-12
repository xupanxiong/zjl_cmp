# coding:utf-8
import tensorflow as tf
import zjl_cnnforward
import zjl_cnnforwardbn
import os
import numpy as np
import zjl_TFRecord
import zjl_config as zjlconf

IMAGE_HIGH = zjlconf.IMAGE_HIGH
IMAGE_WIDTH = zjlconf.IMAGE_WIDTH

BATCH_SIZE = 32
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 500000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "zjl_model"
train_num_examples = 72010*9

def backward():
    x = tf.placeholder(tf.float32, [
        BATCH_SIZE,
        IMAGE_HIGH,   #图像行数
        IMAGE_WIDTH,  #图像列数
        zjl_cnnforward.NUM_CHANNELS],
        name= "input_img")
    y_ = tf.placeholder(tf.float32, [None, zjl_cnnforward.OUTPUT_NODE],name='real_label')
    y = zjl_cnnforward.forward(x, True, REGULARIZER)
    #y = zjl_cnnforwardbn.forward(x, True, REGULARIZER)
    global_step = tf.Variable(0, trainable=False)

    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    #loss = cem + tf.add_n(tf.get_collection('losses'))
    loss = cem

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        train_num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    img_batch, label_batch = zjl_TFRecord.get_tfrecord(BATCH_SIZE,isTrain=True)  # 3

    with tf.Session() as sess:
        sess.run(tf.group(tf.global_variables_initializer(),tf.local_variables_initializer()))

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        coord = tf.train.Coordinator()  # 4
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # 5


        for i in range(STEPS):
            xs, ys = sess.run([img_batch, label_batch])
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

        coord.request_stop()  # 7
        coord.join(threads)  # 8

def main():
    backward()


if __name__ == '__main__':
    main()



