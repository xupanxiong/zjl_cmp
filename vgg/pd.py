import pandas as pd
import zjl_config as zjlconf

def c2index(val='ZJL1',col_name='label'):
    pd_attr = pd.read_table(zjlconf.official_label_list_file,header=None,names=['label','name'])
    print(pd_attr)
    return list(pd_attr[col_name]).index(val)

#print(c2index('ZJL10','label'))
#print(c2index('dog','name'))