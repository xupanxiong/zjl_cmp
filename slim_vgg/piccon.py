'''
author:xupanxiong@qq.com
print and check the tfrecord
'''

import cv2
import numpy as np
import tensorflow as tf
import zjl_TFRecord_index as zjltf
import pandas as pd
import zjl_config as zjlconf
from scipy.spatial.distance import pdist

BATCH_SIZE = 2
img_batch, label_batch = zjltf.get_tfrecord(BATCH_SIZE,isTrain=False)  # 3
pd_attr = pd.read_table(zjlconf.official_label_list_file, header=None, names=['label', 'name'])

with tf.Session() as sess:
    coord = tf.train.Coordinator()  # 4
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # 5

    for i in range(2):
        xs, ys = sess.run([img_batch, label_batch])
        for j in range(BATCH_SIZE):
            cv2.imshow("image", xs[j])  # 显示图片，后面会讲解
            cv2.waitKey(0)  # 等待按键
            cv2.destroyWindow("image")
            print(ys[j])
            print(type(ys[j]))

    coord.request_stop()  # 7s
    coord.join(threads)  # 8


