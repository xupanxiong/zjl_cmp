import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist
import zjl_config as zjlconf

pd_web = pd.read_table(zjlconf.official_word2vec_train_file,header=None,sep=" ",index_col=0)
pd_name = pd.read_table(zjlconf.official_label_list_file,header=None,sep="\t",names=['label','name'])


col_index = list(pd_name['name'])
pd_dis = pd.DataFrame(columns=col_index,index=col_index)

for i_name in col_index:  #i行
    for c_name in col_index:  #j列
        X=np.vstack([pd_web.loc[i_name],pd_web.loc[c_name]])
        pd_dis[i_name][c_name]=1-pdist(X,'cosine')[0]

pd_dis.to_csv(zjlconf.official_word_cosdist_file,sep='\t')