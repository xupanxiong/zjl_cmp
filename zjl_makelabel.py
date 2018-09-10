'''
author:xupanxiong@qq.com
make-label-file
'''

import pandas as pd
import zjl_config as zjlconf
import itertools
import os
import codecs


def traintxtgroupbylabel():
    official_labe = pd.read_table(zjlconf.official_label_train_file,names = ['filename','label'],header = None)
    isExists = os.path.exists(zjlconf.my_label_train_groupby_path)
    if not isExists:
        os.makedirs(zjlconf.my_label_train_groupby_path)
        print('The directory was created successfully')
    else:
        print('directory already exists')
    pd_label_group = official_labe.groupby('label')
    labelnames = []
    for name,group in pd_label_group:
        group.to_csv(os.path.join(zjlconf.my_label_train_groupby_path,name+'.txt'),index = None,header=None, columns=['filename'])
        labelnames.append(name+'.txt\n')
    with codecs.open(zjlconf.my_label_groubpylist_file,'w',encoding='utf-8') as zf:
        for label in labelnames:
            zf.write(label)

    # pd_jpg = pd.DataFrame()
    # pd_jpg['jpg1'] = os.listdir(zjlconf.official_image_train_path)
    # pd_jpg['jpg2'] = pd_jpg['jpg1']
    # pd_jpg['label'] = 2
    # pd_jpg.to_csv(zjlconf.official_image_filelist,header=None,index=None,sep='\t')

def get20files(filename,times):
    with codecs.open(os.path.join(zjlconf.my_label_train_groupby_path,filename),'r',encoding='utf-8') as lf:
        fcont = lf.read()
    filelist = fcont.split()
    if times < 0:
        print('times must >=0')
        return None
    bi = 20 * times
    ei = 20 * (times + 1)
    if ei <= len(filelist):
        return filelist[bi:ei]
    else:
        return filelist[-20:]


def getbatch190x20(times):
    with codecs.open(zjlconf.my_label_groubpylist_file, 'r', encoding='utf-8') as mllf:
        lfcont = mllf.read()
    groupedfiles = lfcont.split()
    labellist = []
    pd_batch = pd.DataFrame()
    for groupfile in groupedfiles:
        fileslist = get20files(groupfile, times)
        col = groupfile.split('.txt')[0]
        pd_batch[col] = fileslist
    return pd_batch


def comb2files(fileslist,same=False):
    comlist = list(itertools.combinations(fileslist,2))
    if same:
        return zip(comlist,[1]*len(comlist))
    else:
        return zip(comlist,[0]*len(comlist))

def makelabel(times=1):
    pd_tomake = getbatch190x20(times=times)
    cols = pd_tomake.columns.values.tolist()
    labellist = []
    savefile = os.path.join(zjlconf.my_label_train_path,'mytrain_'+str(times)+'.txt')
    for col in cols:
        labellist.extend(list(comb2files(list(pd_tomake[col]),same=True)))
    for row in range(pd_tomake.shape[0]):
        labellist.extend(list(comb2files(list(pd_tomake.iloc[row]), same=False)))
    with codecs.open(savefile,'w',encoding='utf-8') as testf:
        for labeli in labellist:
            linesent = labeli[0][0]+'\t'+labeli[0][1]+'\t'+str(labeli[1])+'\n'
            testf.write(linesent)

def makelabel2(times=1):
    pd_tomake = getbatch190x20(times=times)
    cols = pd_tomake.columns.values.tolist()
    labellist = []
    savefile = os.path.join(zjlconf.my_label_train_path, 'mytrain2_' + str(times) + '.txt')
    for col in cols:
        labellist.extend(list(comb2files(list(pd_tomake[col]), same=True)))
    for row in range(2):
        labellist.extend(list(comb2files(list(pd_tomake.iloc[row]), same=False)))
    with codecs.open(savefile, 'w', encoding='utf-8') as testf:
        for labeli in labellist:
            linesent = labeli[0][0] + '\t' + labeli[0][1] + '\t' + str(labeli[1]) + '\n'
            testf.write(linesent)

def createpath():
    isExists = os.path.exists(zjlconf.my_label_train_groupby_path)
    if not isExists:
        os.makedirs(zjlconf.my_label_train_groupby_path)
        print('The directory was created successfully')
    else:
        print('directory already exists')

    isExists = os.path.exists(zjlconf.my_label_train_path)
    if not isExists:
        os.makedirs(zjlconf.my_label_train_path)
        print('The directory was created successfully')
    else:
        print('directory already exists')

    isExists = os.path.exists(zjlconf.my_label_train_shuffle_path)
    if not isExists:
        os.makedirs(zjlconf.my_label_train_shuffle_path)
        print('The directory was created successfully')
    else:
        print('directory already exists')

def labelshuffl():
    #pd_jpg = pd.read_table(zjlconf.official_image_filelist,names = ['jpg1','jpg2','label'],header = None)
    for file in os.listdir(zjlconf.my_label_train_path):
        pd_traini = pd.read_table(os.path.join(zjlconf.my_label_train_path,file),names = ['jpg1','jpg2','label'],header = None)
        #pd_traini = pd_traini.append(pd_jpg.sample(frac=0.9))
        pd_traini = pd_traini.sample(frac=1)
        pd_traini.to_csv(os.path.join(zjlconf.my_label_train_shuffle_path,file),sep='\t',header=None,index=None)

if __name__ == '__main__':
    createpath()
    traintxtgroupbylabel()
    for i in range(10):
        makelabel2(i)
    labelshuffl()