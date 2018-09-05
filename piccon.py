'''
author:xupanxiong@qq.com
print and check the tfrecord
'''

import cv2
import numpy as np
import zjl_TFRecord
import tensorflow as tf
import zjl_cnnforward
import random

BATCH_SIZE = 2

resize_width = 128
resize_height = 256

def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

img1 = cv2.imread(r'E:\2018CMP\zhijiang\DatasetA_train_20180813\train\000c0d617f5b67d116dee15c40d1d47d.jpeg')
img2 = cv2.imread(r'E:\2018CMP\zhijiang\DatasetA_train_20180813\train\000c272e96d861aa54431b4965136310.jpeg')

# img = np.concatenate((img1, img2), axis=0)
# img = cv2.resize(img, (resize_width,resize_height), interpolation=cv2.INTER_CUBIC)
#
# cv2.imshow("image", img) # 显示图片，后面会讲解
# cv2.waitKey(0) #等待按键
# cv2.destroyWindow("image")

img1 =rotate_bound(img1,90*random.randint(1,3))
img = np.concatenate((img1, img2), axis=0)
img = cv2.resize(img, (resize_width,resize_height), interpolation=cv2.INTER_CUBIC)

cv2.imshow("image", img) # 显示图片，后面会讲解
cv2.waitKey(0) #等待按键
cv2.destroyWindow("image")

# img_batch, label_batch = zjl_TFRecord.get_tfrecord(BATCH_SIZE,isTrain=False)  # 3
#
# with tf.Session() as sess:
#     coord = tf.train.Coordinator()  # 4
#     threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # 5
#
#     for i in range(2):
#         xs, ys = sess.run([img_batch, label_batch])
#         reshaped_xs = np.reshape(xs, (
#             BATCH_SIZE,
#             zjl_cnnforward.IMAGE_WIDTH,
#             zjl_cnnforward.IMAGE_HIGH,
#             zjl_cnnforward.NUM_CHANNELS))
#         for i in range(len(ys)):
#             print("pic i :\n", xs[i], ys[i])
#
#     coord.request_stop()  # 7
#     coord.join(threads)  # 8