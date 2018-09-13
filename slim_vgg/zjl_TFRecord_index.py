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
import pandas as pd


IMAGE_WIDTH = zjlconf.IMAGE_WIDTH
IMAGE_HIGH = zjlconf.IMAGE_HIGH
len_labels = 230


def write_tfRecord(tfRecordName, image_path, label_file):
    writer = tf.python_io.TFRecordWriter(tfRecordName)
    num_pic = 0
    f = open(label_file, 'r')
    contents = f.readlines()
    f.close()
    for content in contents:
        value = content.split('\t')
        img = cv2.imread(os.path.join(image_path,value[0]))
        img_raw = img.tobytes()
        labels = [0] * len_labels
        idx = c2index(val=value[1],col_name='label')
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
    isExists = os.path.exists(zjlconf.my_tfrecord_path1)
    if not isExists:
        os.makedirs(zjlconf.my_tfrecord_path1)
        print('The directory was created successfully')
    else:
        print('directory already exists')

    for labelfile in os.listdir(zjlconf.my_label_train_path):
        tfRecord_name  = os.path.join(zjlconf.my_tfrecord_path1,
                                      labelfile.replace('.txt','.tfrecords'))
        write_tfRecord(tfRecord_name, zjlconf.official_image_train_path,
                       os.path.join(zjlconf.my_label_train_path,labelfile))


def read_tfRecord(tfRecord_path):
    filename_queue = tf.train.string_input_producer([tfRecord_path], shuffle=True)
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
    if isTrain:
        tfRecord_file = os.path.join(zjlconf.my_tfrecord_path1, 'mytrain.tfrecords')
    else:
        tfRecord_file = os.path.join(zjlconf.my_tfrecord_path1, 'mytest.tfrecords')
    img, label = read_tfRecord(tfRecord_file)
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=bathchsize,
                                                    num_threads=4,
                                                    capacity=bathchsize*64,
                                                    min_after_dequeue=bathchsize*32
                                                    )

    return img_batch, label_batch

def c2index(val='ZJL1',col_name='label'):
    pd_attr = pd.read_table(zjlconf.official_label_list_file,header=None,names=['label','name'])
    return list(pd_attr[col_name]).index(val.split('\n')[0])

def index2ln(id):
    pd_attr = pd.read_table(zjlconf.official_label_list_file,header=None,names=['label','name'])
    return pd_attr.loc[id]['label'],pd_attr.loc[id]['name']

def get300wordvetor(name='dog'):
    df_wordvetor = pd.read_table(zjlconf.official_word2vec_train_file,header=None,sep=" ",index_col=0)
    return df_wordvetor.loc[name]

def label2name(label='ZJL01'):
    df_name = pd.read_table(zjlconf.official_label_list_file, header=None, sep="\t", names=['label', 'name'],index_col='label')
    return df_name.loc[label]['name']

def name2wordvetor(name='dog'):
    df_wv = pd.read_table(zjlconf.official_word2vec_train_file, header=None, sep=" ", index_col=0)
    return df_wv.loc[name]

def label2wordvetor(label='ZJL01'):
    name = label2name(label.split()[0])
    return name2wordvetor(name)

def main():
    generate_tfRecord()


if __name__ == '__main__':
    main()
