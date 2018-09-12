'''
author:xupanxiong@qq.com
print and check the tfrecord
'''

import cv2
import numpy as np
import tensorflow as tf
import zjl_TFRecord
import pandas as pd
import zjl_config as zjlconf
from scipy.spatial.distance import pdist

BATCH_SIZE = 5

def distcos(y1,y2):
    x3_norm = tf.sqrt(tf.reduce_sum(tf.square(ys1), axis=1))
    x4_norm = tf.sqrt(tf.reduce_sum(tf.square(ys2), axis=1))
    # 内积
    x3_x4 = tf.reduce_sum(tf.multiply(ys1, ys2), axis=1)
    cosin = x3_x4 / (x3_norm * x4_norm)
    return cosin

img_batch, label_batch = zjl_TFRecord.get_tfrecord(BATCH_SIZE,isTrain=False)  # 3
pd_attr = pd.read_table(zjlconf.official_label_list_file, header=None, names=['label', 'name'])

with tf.Session() as sess:
    coord = tf.train.Coordinator()  # 4
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # 5

    for i in range(3):
        xs1, ys1 = sess.run([img_batch, label_batch])
        xs2, ys2 = sess.run([img_batch, label_batch])
        dist = distcos(ys1,ys2)
        loss = 1.0 - tf.abs(dist)
        vdist,vloss = sess.run([dist,loss])
        print(vdist,vloss)

        # for i in range(BATCH_SIZE):
        #     dist2 = pdist(np.vstack([ys1[i],ys2[i]]),'cosine')
        #     print(dist2)
        #print(ys1,ys2)
        #print(type(ys1),ys1.shape)
        # for i in range(BATCH_SIZE):
        #     #id = np.argmax(ys[i])
        #     #l,n = zjl_TFRecord.index2ln(id)
        #     #print(zjl_TFRecord.get300wordvetor(n))
        #     # print(xs1[i].shape)
        #     # cv2.imshow("image", xs1[i])  # 显示图片，后面会讲解
        #     # cv2.waitKey(0)  # 等待按键
        #     # cv2.destroyWindow("image")
        #     print(ys1[i])
        #     print(type(ys1[i]))

    coord.request_stop()  # 7
    coord.join(threads)  # 8


