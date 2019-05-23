from functions import *

import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import gc
import scipy.sparse as sparse
import re
import itertools
from tqdm import tqdm
import time
from sklearn.decomposition import TruncatedSVD


path = "./"

with open(path + "data/Cooking_Clean_train.pkl", 'rb') as f:
        df_train = pickle.load(f)
with open(path + "data/Cooking_Clean_test.pkl", 'rb') as f:
        df_test = pickle.load(f)
with open(path + "data/Cooking_Clean_valid.pkl", 'rb') as f:
        df_valid = pickle.load(f)

labels = list(df_train)
# We only consider the labels in the train data set, indeed we can't predict a label which is not in the train set
labels.remove("text")
labels = ["text"] + labels

def fill_in_datasets(labs, df):
    out = pd.DataFrame()
    for lab in labs :
        if lab in list(df):   
            out[lab] = df[lab]
        else:
            out[lab] = (df.shape[0])*[0]
    return(out)
        

        
df_train = fill_in_datasets(labels, df_train)
df_test = fill_in_datasets(labels, df_test)
df_valid = fill_in_datasets(labels, df_valid)


#Train
pred_train = np.load("results/cooking_train.npy")
pred_train = trad_output_v(pred_train)


gross_loss(np.array(df_train[labels[1:]]), pred_train)
TPR(np.array(df_train[labels[1:]]), pred_train)
count_correct_comments(np.array(df_train[labels[1:]]), pred_train)
#Test
preds_test = np.load("results/cooking_test.npy")
preds_test = trad_output_v(preds_test)
gross_loss(np.array(df_test[labels[1:]]), preds_test)
TPR(np.array(df_test[labels[1:]]), preds_test)
count_correct_comments(np.array(df_test[labels[1:]]), preds_test)
#Validation
preds_valid = np.load("results/cooking_valid.npy")
preds_valid = trad_output_v(preds_valid)
gross_loss(np.array(df_valid[labels[1:]]), preds_valid)
TPR(np.array(df_valid[labels[1:]]), preds_valid)
count_correct_comments(np.array(df_valid[labels[1:]]), preds_valid)
