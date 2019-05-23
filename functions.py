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


#########################################################
#              Useful Functions
#########################################################

def trad_output(x):
    
    if x<0.5:
        return(0)
    else:
        return(1)
        
trad_output_v = np.vectorize(trad_output)

def gross_loss(arr_true, arr_pred):
    
    errors = np.sum(np.abs(arr_true - arr_pred))
    tot = np.shape(arr_true)[0]*np.shape(arr_true)[1]
    print('Gross Loss : ' + str(errors/tot))
    return(errors/tot)

def find_1(arr):

    list_positions = []
    for i in range(np.shape(arr)[0]):
        for j in range(np.shape(arr)[1]):
            if (arr[i,j] == 1):
                list_positions.append([i,j])
    return(list_positions)



def TPR(arr_true, arr_pred):
    
    TP = 0.0
    FN = 0.0
    TN = 0.0
    FP = 0.0
    
    for i in range(np.shape(arr_true)[0]):
        for j in range(np.shape(arr_true)[1]):
            if (arr_true[i,j] == 1):
                if (arr_pred[i,j] == 0):
                    FN += 1
                else:
                    TP += 1
            else:
               if (arr_pred[i,j] == 0):
                   TN += 1
               else:
                   FP += 1
    TPR = TP/(TP+FN)
    FPR = FP/(FP+TN)
    
    print('True Positive rate = ' + str(TPR))
    print('False Positive rate = ' + str(FPR))
    return TPR, FPR
    #return TP, FN, FP, TN
    
def count_correct_comments(arr_true, arr_pred):
    
    nb_bad = 0
    nb_at_least_one = 0
    nb_w_label = 0
    L = np.shape(arr_true)[0]
    M = np.shape(arr_true)[1]
    for i in range(L):
        try:
            np.testing.assert_array_equal(arr_true[i,:], arr_pred[i,:])
        except:
            count = 0
            count_label = 0
            nb_bad += 1
            j = 0
            while (j < M) and (count==0):
                if arr_pred[i,j]==1:
                    count_label = 1
                    if arr_true[i,j]==1:
                        count = 1
                j += 1
            nb_at_least_one += count
            nb_w_label += count_label   
                
    correct_labels = (L-nb_bad)/L
    at_least_one = (L-nb_bad+nb_at_least_one)/L
    wo_label = (nb_bad - nb_w_label)/L
    print('Correct labels rate = ' + str(correct_labels))
    print('at least one correct label rate = ' + str(at_least_one))
    print('without any label rate = ' + str(wo_label))
    return()