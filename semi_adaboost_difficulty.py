import numpy as np
import numpy.matlib

from uci_utils import *
from collections import Counter

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import warnings

from skmultiflow.data.random_rbf_generator import RandomRBFGenerator
from skmultiflow.data.random_tree_generator import RandomTreeGenerator
from operator import itemgetter
from scipy.signal import find_peaks
from scipy.fft import rfft, rfftfreq
from scipy.fft import fft, fftfreq
from sklearn import tree
import heapq
import timeit
import statistics
import math
import random
from itertools import groupby


def split_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.90, random_state=0)
    return X_train, X_test, y_train, y_test

def fit_ensemble(X_train, X_test, y_train, n_estima,tree_dep,global_rand_seed):
    
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=tree_dep), n_estimators=n_estima, algorithm = 'SAMME',random_state=global_rand_seed)
    
    #start = timeit.default_timer()
    clf.fit(X_train, y_train)
    #y_pred = clf.predict(X_test)
    #stop = timeit.default_timer()
    
    y_all_pred = clf.staged_predict(X_test)
    
    #print(X_test[0])
    y_all_pred = list(y_all_pred)
    #print(y_all_pred)
    #print(len(y_all_pred[0]))
    #all_score = clf.staged_score(X_test, y_test)
    #accuracy = accuracy_score(y_test, y_pred)
    all_estimator = clf.estimators_
    #estimator_weights = clf.estimator_weights_
    #estimator_errors = clf.estimator_errors_
    #print("Ensemble",accuracy, "Time:", stop - start)
    return y_all_pred,all_estimator

def get_first_consis_label(y_pred, consistency):
    consistent_ind = 0
    for j in range(0,len(y_pred)-consistency+1):
        cnt = 0
        found = False
        while (y_pred[j+cnt] == y_pred[j]):
            cnt += 1
            if cnt == consistency:
                consistent_ind = j
                found = True
                break
        if found is True:
            break
    if found is False:
        consistent_ind = None
    return consistent_ind

def get_most_consis_label(y_pred):
    consis_cnt = [(k, sum(1 for i in g)) for k,g in groupby(y_pred)]
    sorted_consis_cnt = sorted(consis_cnt, key=lambda x: x[1])
    most_consis_label = sorted_consis_cnt[-1][0]
    return most_consis_label


def rolling_avg(x, N):
    start = []
    end = []
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    for i in range(1, N):
        temp_sum = cumsum[i]/i
        start.append(temp_sum)

    max_window = list(np.convolve(x, np.ones(N)/N, mode='valid'))
    
    end_window = x[-N+1:]
    end_window.reverse()
    end_cumsum = np.cumsum(np.insert(end_window, 0, 0))

    for i in range(1,N):
        temp_sum = end_cumsum[i]/i
        end.append(temp_sum)
    end.reverse()

    combined = start+max_window+end
    #print(combined)
    return combined

def get_most_consis_scores(y_all_pred, consistency, window_size):
    
    consistency =  int(consistency * len(y_all_pred))
    if consistency == 0:
        consistency = 1
        
    #get staged prediction of each instacnes
    y_all_pred = np.transpose((np.array(y_all_pred)))
    

    #print("Consistency:", consistency)
    #print(y_all_pred[2])
    #print(get_most_consis_label(y_all_pred[2]))
    
    #Initilized Score Dicts
    clf_count={}
    fluc_degree = {}
    oscillation_count = {}
    oscillation_auc = {}
    most_consis_label_lst = []
    label_fluc = {}
    
    #print(len(y_all_pred)) 
    for x in range(len(y_all_pred)):
        clf_count[x] = None
        oscillation_count[x] = None
        fluc_degree[x] = None
        oscillation_auc[x] = None
        label_fluc[x] = None
    

    for i in range(len(y_all_pred)):
        
        #for each instance get clf cnt
        #first_consis_label = get_first_consis_label(y_all_pred[i], consistency)
        #clf_count[i] = first_consis_label
        
        #get fluc degree of all inst
        fluc_degree[i] = len(Counter(y_all_pred[i]))
        
        #get most consis label of current inst
        most_consis_label = get_most_consis_label(y_all_pred[i])
        most_consis_label_lst.append(most_consis_label)
        
        #construct Fluc Graph based on the most consis label
        y = []
        #print(y_all_pred[0])
        for sample in y_all_pred[i]:
            if sample != most_consis_label:
                y.append(1)

            else:
                y.append(0)
                
        label_fluc[i] = y
        #get Osci cnt based on most consis label
        osci_count = (np.diff(y)!=0).sum()
        oscillation_count[i] = osci_count
        
        #get rolling average of Fluc Graph
        #print(y)
        full_average = rolling_avg(y,window_size)
        oscillation_auc[i] = full_average
        
        
        for j in range(0,len(y)-consistency+1):
            cnt = 0
            found = False
            while (y[j+cnt] == y[j]) and y[j] == 0:
                cnt += 1
                if cnt == consistency:
                    clf_count[i] = j
                    found = True
                    break
            if found is True:
                break
        
    
    return clf_count, fluc_degree, oscillation_count, oscillation_auc, most_consis_label_lst, label_fluc


def get_most_prob_scores(y_all_pred, consistency, window_size):
    most_prob_label = y_all_pred[-1]
    

    consistency =  int(consistency * len(y_all_pred))
    if consistency == 0:
        consistency = 1

    
    clf_count={}
    oscillation_count = {}
    oscillation_auc = {}
    label_fluc = {}
    for x in range(len(most_prob_label)):
        clf_count[x] = None
        oscillation_count[x] = None
        oscillation_auc[x] = None
        label_fluc[x] = None


    y_all_pred = np.transpose((np.array(y_all_pred)))
    for i in range(len(y_all_pred)):
        right_label = most_prob_label[i]
        
        y = []
        for sample in y_all_pred[i]:
            if sample != right_label:
                y.append(1)
            else:
                y.append(0)
                
        label_fluc[i] = y
        osci_count = (np.diff(y)!=0).sum()
        oscillation_count[i] = osci_count

        full_average = rolling_avg(y,window_size)
        oscillation_auc[i] = full_average


        
        for j in range(0,len(y)-consistency+1):
            cnt = 0
            found = False
            while (y[j+cnt] == y[j]) and y[j] == 0:
                cnt += 1
                if cnt == consistency:
                    clf_count[i] = j
                    found = True
                    break
            if found is True:
                break
            
    
    return clf_count, oscillation_count, oscillation_auc, most_prob_label, label_fluc


































