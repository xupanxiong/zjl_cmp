'''
author:xupanxiong@qq.com
make TFRecord-file
'''

# coding:utf-8

import tensorflow as tf
import numpy as np
import os
import cv2
import zjl_config as zjlconf
import random

IMAGE_WIDTH = zjlconf.IMAGE_WIDTH
IMAGE_HIGH = zjlconf.IMAGE_HIGH
len_labels = 2


def write_tfRecord(tfRecordName, image_path, label_file):
    writer = tf.python_io.TFRecordWriter(tfRecordName)
    num_pic = 0
    f = open(label_file, 'r')
    contents = f.readlines()
    f.close()
    for content in contents:
        value = content.split()
        img1 = cv2.imread(os.path.join(image_path,value[0]))
        img2 = cv2.imread(os.path.join(image_path,value[1]))
        # if value[0] ==value[1]:
        #     img1 = rotate_bound(img1, 90 * random.randint(1, 3))
        img = np.concatenate((img1, img2), axis=0)
        #img = cv2.resize(img, (resize_width,resize_height), interpolation=cv2.INTER_CUBIC)
        img_raw = img.tobytes()
        labels = [0] * len_labels
        idx = value[2]
        labels[int(idx)] = 1

        example = tf.train.Example(features=tf.train.Features(feature={
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=labels))
        }))
        writer.write(example.SerializeToString())
        num_pic += 1
        print("the number of picture:", num_pic)
    writer.close()
    print("write tfrecord successful")


def generate_tfRecord():
    isExists = os.path.exists(zjlconf.my_tfrecord_path)
    if not isExists:
        os.makedirs(zjlconf.my_tfrecord_path)
        print('The directory was created successfully')
    else:
        print('directory already exists')

    for labelfile in os.listdir(zjlconf.my_label_train_shuffle_path):
        tfRecord_name  = os.path.join(zjlconf.my_tfrecord_path,
                                      labelfile.replace('.txt','.tfrecords'))
        write_tfRecord(tfRecord_name, zjlconf.official_image_train_path,
                       os.path.join(zjlconf.my_label_train_shuffle_path,labelfile))
    #write_tfRecord(tfRecord_valid, image_valid_path, label_valid_path)


def read_tfRecord(tfRecord_path):
    filename_queue = tf.train.string_input_producer(tfRecord_path, shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([len_labels], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string)
                                       })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [IMAGE_HIGH, IMAGE_WIDTH, 3])  # reshape为64*128的3通道图片
    img = tf.cast(img, tf.float32) * (1. / 255)
    label = tf.cast(features['label'], tf.float32)
    return img, label


def get_tfrecord(bathchsize, isTrain=True):
    filelist = []
    for file in os.listdir(zjlconf.my_tfrecord_path):
        filelist.append(os.path.join(zjlconf.my_tfrecord_path, file))
    if isTrain:
        tfRecord_path = filelist[1:]
    else:
        tfRecord_path = [filelist[0]]
    img, label = read_tfRecord(tfRecord_path)
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=bathchsize,
                                                    num_threads=4,
                                                    capacity=bathchsize*64,
                                                    min_after_dequeue=bathchsize*32
                                                    )

    return img_batch, label_batch

def rotate_bound(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(image, M, (nW, nH))

def main():
    generate_tfRecord()


if __name__ == '__main__':
    main()
