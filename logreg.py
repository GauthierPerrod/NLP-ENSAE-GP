# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
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
from functions import *


### We import the preprocessed data sets:

try: 
    path = "./"

    with open(path + "data/Cooking_Clean_train.pkl", 'rb') as f:
        df_train = pickle.load(f)
    with open(path + "data/Cooking_Clean_test.pkl", 'rb') as f:
        df_test = pickle.load(f)
    with open(path + "data/Cooking_Clean_valid.pkl", 'rb') as f:
        df_valid = pickle.load(f)
except:
    print('Please write the good path and check preprocessed data are there')

### We extract the labels:

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

labels = labels[1:]

### We transform the text data (the subjects) in real vectors with the Tf-Idf method :

import nltk.corpus 
nltk.download('stopwords')
from nltk.corpus import stopwords
stpW = stopwords.words("english")

transform_com = TfidfVectorizer(analyzer = 'char', ngram_range=(1,4), max_features=50000,stop_words=stpW ,min_df=2).fit(df_train.text)


%time comments_train = transform_com.transform(df_train.text)
%time comments_test = transform_com.transform(df_test.text)
%time comments_valid = transform_com.transform(df_valid.text)



#########################################################
#              Logistic Regression
#########################################################
    
from sklearn.multiclass import OneVsRestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

### We use a One-vs-the-rest strategy ang a Logistic Regression as classifier

## We optimise the hyper-parameters by cross-validation:

#GridSearch

grid = list(itertools.product(['l1', 'l2'], [0.01, 0.1, 1, 10, 100]))

list_TPR = []

for param in tqdm(grid):
    try:
        est = LogisticRegression(penalty = param[0], C = param[1])
        model = OneVsRestClassifier(est, n_jobs=-1)
        model.fit(comments_train, np.array(df_train[labels[1:]]))
        preds_test = model.predict_proba(comments_test)
        preds_test = trad_output_v(preds_test)
        list_TPR.append(TPR(np.array(df_test[labels[1:]]), preds_test)[0])
    except:
        print('Wrong value of parameters : ' + str(param))

maximum_indices = np.where(np.array(list_TPR)==max(list_TPR))
print('Best Parameters : ' + str(grid[maximum_indices[0][0]]) + ', TPR on test set = ' + str(max(list_TPR)))

## Best Model:

best_params = grid[maximum_indices[0][0]]     

est = LogisticRegression(penalty = best_params[0], C = best_params[1])
model = OneVsRestClassifier(est, n_jobs = -1)

# Fit
model.fit(comments_train, np.array(df_train[labels[1:]]))

#Train
pred_train = model.predict_proba(comments_train)
pred_train = trad_output_v(pred_train)
gross_loss(np.array(df_train[labels[1:]]), pred_train)
TPR(np.array(df_train[labels[1:]]), pred_train)
count_correct_comments(np.array(df_train[labels[1:]]), pred_train)
#Test
preds_test = model.predict_proba(comments_test)
preds_test = trad_output_v(preds_test)
gross_loss(np.array(df_test[labels[1:]]), preds_test)
TPR(np.array(df_test[labels[1:]]), preds_test)
count_correct_comments(np.array(df_test[labels[1:]]), preds_test)
#Validation
preds_valid = model.predict_proba(comments_valid)
preds_valid = trad_output_v(preds_valid)
gross_loss(np.array(df_valid[labels[1:]]), preds_valid)
TPR(np.array(df_valid[labels[1:]]), preds_valid)
count_correct_comments(np.array(df_valid[labels[1:]]), preds_valid)



#####################
#####################

