'''
author:xupanxiong@qq.com
make-label-file
'''

import pandas as pd
import zjl_config as zjlconf
import itertools
import os
import codecs

def shufflelabel():
    isExists = os.path.exists(zjlconf.my_label_train_path)
    if not isExists:
        os.makedirs(zjlconf.my_label_train_path)
        print('The directory was created successfully')
    else:
        print('directory already exists')
    official_label = pd.read_table(zjlconf.official_label_train_file,names = ['filename','label'],header = None)
    df_train = official_label.sample(frac=0.95)
    df_train.to_csv(zjlconf.my_label_train,header=None,index=None,sep='\t')
    df_valid = official_label.sample(frac=0.05)
    df_valid.to_csv(zjlconf.my_label_valid, header=None, index=None, sep='\t')


if __name__ == '__main__':

    shufflelabel()