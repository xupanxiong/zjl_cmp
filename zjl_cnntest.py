# coding:utf-8
import time
import tensorflow as tf
import numpy as np
import zjl_cnnforward
import zjl_cnnbackward
import zjl_TFRecord
import zjl_config as zjlconf

IMAGE_WIDTH = zjlconf.IMAGE_WIDTH
IMAGE_HIGH = zjlconf.IMAGE_HIGH

TEST_INTERVAL_SECS = 50
TEST_NUM = 100  # 1


def test():

    with tf.Graph().as_default() as g:
        with tf.device('/cpu:0'):
            x = tf.placeholder(tf.float32, [
                TEST_NUM,
                IMAGE_HIGH,
                IMAGE_WIDTH,
                zjl_cnnforward.NUM_CHANNELS])
            y_ = tf.placeholder(tf.float32, [None, zjl_cnnforward.OUTPUT_NODE])
            y = zjl_cnnforward.forward(x, False, None)

            ema = tf.train.ExponentialMovingAverage(zjl_cnnbackward.MOVING_AVERAGE_DECAY)
            ema_restore = ema.variables_to_restore()
            saver = tf.train.Saver(ema_restore)

            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            img_batch, label_batch = zjl_TFRecord.get_tfrecord(TEST_NUM, isTrain=False)  # 2

        while True:
            with tf.Session() as sess:
                with tf.device('/cpu:0'):
                    ckpt = tf.train.get_checkpoint_state(zjl_cnnbackward.MODEL_SAVE_PATH)
                    if ckpt and ckpt.model_checkpoint_path:
                        saver.restore(sess, ckpt.model_checkpoint_path)
                        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                        coord = tf.train.Coordinator()  # 3
                        threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # 4

                        xs, ys = sess.run([img_batch, label_batch])  # 5

                        accuracy_score = sess.run(accuracy, feed_dict={x: xs, y_: ys})

                        print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_score))

                        coord.request_stop()  # 6
                        coord.join(threads)  # 7

                    else:
                        print('No checkpoint file found')
                        return
            time.sleep(TEST_INTERVAL_SECS)


def main():
    with tf.device('/cpu:0'):
        test()  # 8


if __name__ == '__main__':
    main()
