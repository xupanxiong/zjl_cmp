'''
author:xupanxiong@qq.com
print and check the tfrecord
'''

import cv2
import numpy as np
import zjl_TFRecord
import tensorflow as tf
import zjl_cnnforward
import zjl_TFRecord
import random

BATCH_SIZE = 2

resize_width = 64
resize_height = 128

img1 = cv2.imread(r'./DatasetA_train_20180813/train/000c0d617f5b67d116dee15c40d1d47d.jpeg')
img2 = cv2.imread(r'./DatasetA_train_20180813/train/000c272e96d861aa54431b4965136310.jpeg')

# img = np.concatenate((img1, img2), axis=0)
# img = cv2.resize(img, (resize_width,resize_height), interpolation=cv2.INTER_CUBIC)
#
# cv2.imshow("image", img) # 显示图片，后面会讲解
# cv2.waitKey(0) #等待按键
# cv2.destroyWindow("image")

img1 = zjl_TFRecord.rotate_bound(img1,90*random.randint(1,3))
img = np.concatenate((img1, img2), axis=0)
img = cv2.resize(img, (resize_width,resize_height), interpolation=cv2.INTER_CUBIC)

cv2.imshow("image", img) # 显示图片，后面会讲解
cv2.waitKey(0) #等待按键
cv2.destroyWindow("image")

img_batch, label_batch = zjl_TFRecord.get_tfrecord(BATCH_SIZE,isTrain=False)  # 3

with tf.Session() as sess:
    coord = tf.train.Coordinator()  # 4
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # 5

    for i in range(2):
        xs, ys = sess.run([img_batch, label_batch])
        for i in range(len(ys)):
            print(ys[i])
            cv2.imshow("image", xs[i])  # 显示图片，后面会讲解
            cv2.waitKey(0)  # 等待按键
            cv2.destroyWindow("image")

    coord.request_stop()  # 7
    coord.join(threads)  # 8