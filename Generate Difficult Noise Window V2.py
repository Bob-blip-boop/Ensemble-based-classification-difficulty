import numpy as np
import numpy.matlib
from uci_utils import *
from collections import Counter
import random
import matplotlib.pyplot as plt
import sys 

import pandas as pd
import seaborn as sns


from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import cohen_kappa_score

from skclean.detectors import InstanceHardness
from skclean.detectors import KDN
from skclean.detectors import ForestKDN
from skclean.detectors import RandomForestDetector
from skclean.detectors import PartitioningDetector
from skclean.detectors import MCS

from efficient_adaboost_difficulty_V3 import fit_ensemble 
from efficient_adaboost_difficulty_V3 import get_base_clf
from dNN import get_dNN

from skmultiflow.data.random_rbf_generator import RandomRBFGenerator
from skmultiflow.data.random_tree_generator import RandomTreeGenerator
from operator import itemgetter
from itertools import groupby
from scipy.signal import find_peaks

import timeit

from warnings import simplefilter
simplefilter("ignore", category=ConvergenceWarning)
simplefilter("ignore", category=RuntimeWarning)
simplefilter("ignore", category=UserWarning)

def to_str_rnd(n):
    n = np.format_float_positional(n)
    return str(n)

def standardize (lst):
    return [((i-np.mean(lst))/np.std(lst)) for i in lst]

def normalize (x):
    normalized = []
    if max(x) == min(x):
        return x
    else:
        for i in x:
            normalized_i = (i-min(x))/(max(x)-min(x))
            normalized.append(normalized_i)
        return normalized

def decisiontree(X_train, X_test, y_train, y_test, global_rand_seed):
    model = DecisionTreeClassifier(random_state = global_rand_seed)
    
    start = timeit.default_timer()
    model.fit(X_train,y_train)
    predicted_label = model.predict(X_test)
    stop = timeit.default_timer()
    
    time = stop - start
    y_pred = model.predict_proba(X_test)
    probability_estimates = []
    for i in y_pred:
        probability_estimates.append(max(i))

    #accuracy = accuracy_score(y_test, predicted_label)
    #print("dt:",accuracy, "Time:",time)

    return predicted_label, probability_estimates

def decisiontree_acc(X_train, X_test, y_train, y_test, global_rand_seed, metric):
    model = DecisionTreeClassifier(random_state = global_rand_seed)
    #start = timeit.default_timer()
    model.fit(X_train,y_train)
    predicted_label = model.predict(X_test)
    #stop = timeit.default_timer()
    #time = stop - start

    if metric == "kappa":
        kappa = cohen_kappa_score(y_test, predicted_label)
        return kappa
        
    else:
        accuracy = accuracy_score(y_test, predicted_label)
        #print("dt:",accuracy)
        return accuracy

@ignore_warnings(category=ConvergenceWarning)
def mlp(X_train, X_test, y_train, y_test, global_rand_seed):
    #model = MLPClassifier(max_iter=500,solver='sgd', momentum = 0.2 )
    model = MLPClassifier(max_iter=200,random_state = global_rand_seed)
    
    start = timeit.default_timer()
    model.fit(X_train,y_train)
    predicted_label = model.predict(X_test)
    stop = timeit.default_timer()
    
    time = stop - start
    probability_estimates = []
    y_pred = model.predict_proba(X_test)
    for i in y_pred:
        probability_estimates.append(max(i))
    #accuracy = accuracy_score(y_test, predicted_label)
    #print("mlp:",accuracy, "Time:",time)
    return predicted_label, probability_estimates

@ignore_warnings(category=ConvergenceWarning)
def mlp_acc(X_train, X_test, y_train, y_test, global_rand_seed):
    model = MLPClassifier(max_iter=200,random_state = global_rand_seed)
    model.fit(X_train,y_train)
    predicted_label = model.predict(X_test)
    accuracy = accuracy_score(y_test, predicted_label)
    #print("mlp:",accuracy)
    return accuracy
    
def KNN(X_train, X_test, y_train, y_test):
    model = KNeighborsClassifier(n_neighbors=5)
    
    start = timeit.default_timer()
    model.fit(X_train,y_train)
    y_pred = model.predict_proba(X_test)
    stop = timeit.default_timer()
    
    time = stop - start
    probability_estimates = []
    for i in y_pred:
        probability_estimates.append(max(i))
    predicted_label = model.predict(X_test)
    #accuracy = accuracy_score(y_test, predicted_label)
    #print("KNN:",accuracy, "Time:",time)
    return predicted_label, probability_estimates

def KNN_acc(X_train, X_test, y_train, y_test):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train,y_train)
    predicted_label = model.predict(X_test)
    accuracy = accuracy_score(y_test, predicted_label)
    #print("KNN:",accuracy)
    return accuracy

def NB(X_train, X_test, y_train, y_test):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    model = MultinomialNB()
    
    start = timeit.default_timer()
    model.fit(X_train,y_train)
    y_pred = model.predict_proba(X_test)
    stop = timeit.default_timer()
    
    time = stop - start
    probability_estimates = []
    for i in y_pred:
        probability_estimates.append(max(i))
    predicted_label = model.predict(X_test)
    #accuracy = accuracy_score(y_test, predicted_label)
    #print("NB:",accuracy, "Time:",time)
    return predicted_label, probability_estimates

def NB_acc(X_train, X_test, y_train, y_test):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    model = MultinomialNB()
    
    model.fit(X_train,y_train)
    predicted_label = model.predict(X_test)
    accuracy = accuracy_score(y_test, predicted_label)
    #print("NB:",accuracy)
    return accuracy

def RF(X_train, X_test, y_train, y_test, global_rand_seed):
    model = RandomForestClassifier(random_state = global_rand_seed)
    
    start = timeit.default_timer()
    model.fit(X_train,y_train)
    y_pred = model.predict_proba(X_test)
    stop = timeit.default_timer()
    
    time = stop - start
    probability_estimates = []
    for i in y_pred:
        probability_estimates.append(max(i))
    predicted_label = model.predict(X_test)
    #accuracy = accuracy_score(y_test, predicted_label)
    #print("RF:",accuracy, "Time:",time)
    return predicted_label, probability_estimates

def RF_acc(X_train, X_test, y_train, y_test, global_rand_seed):
    model = RandomForestClassifier(random_state = global_rand_seed)
    model.fit(X_train,y_train)
    predicted_label = model.predict(X_test)
    accuracy = accuracy_score(y_test, predicted_label)
    #print("RF:",accuracy)
    return accuracy




def split_data(X,y, global_rand_seed):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.60, random_state=global_rand_seed)
    return X_train, X_test, y_train, y_test

def get_scores(X_train, X_test, y_train, y_test, global_rand_seed):
    
    frequency = []
    dt_predicted_label, dt_score = decisiontree(X_train, X_test, y_train, y_test, global_rand_seed)
    
    mlp_predicted_label, mlp_score = mlp(X_train, X_test, y_train, y_test, global_rand_seed)
            
    knn_predicted_label, knn_score = KNN(X_train, X_test, y_train, y_test)
            
    nb_predicted_label, nb_score = NB(X_train, X_test, y_train, y_test)
            
    rf_predicted_label, rf_score = RF(X_train, X_test, y_train, y_test, global_rand_seed)
    

    for i in range(len(y_test)):
        count = 0
        
        if y_test[i] != nb_predicted_label[i]:
            count += 1
        if y_test[i] != rf_predicted_label[i]:
            count += 1
        if y_test[i] != dt_predicted_label[i]:
            count += 1    
        if y_test[i] != mlp_predicted_label[i]:
            count += 1
        if y_test[i] != knn_predicted_label[i]:
            count += 1

        count = count/5
        frequency.append(count)
        

    
    clf_scores = [dt_score, mlp_score, knn_score, nb_score, rf_score]
    clf_scores = [np.array(x) for x in clf_scores]
    clf_scores = [np.mean(k) for k in zip(*clf_scores)]
    
    return frequency,clf_scores

def get_osc_cnt(ind_y_all_pred,true_lbl):
    err_count = {}
    osc_count = {}
    boost_count = {}
    for i in range(len(ind_y_all_pred)):
        temp = []
        right_label = true_lbl[i]
        e_count = 0
        for sample in ind_y_all_pred[i]:
            if sample != right_label:
                e_count += 1
                temp.append(1)
            else:
                temp.append(0)

        o_count = (np.diff(temp)!=0).sum()
        err_count[i] = e_count
        osc_count[i] = o_count
        
        if 0 in temp:
            boost_count[i] = temp.index(0)
        else:
            boost_count[i] = len(temp)
    return err_count,osc_count, boost_count


def get_osci_score_small(X_train, X_test, y_train, y_test,n_estimators, seed):
    
    window_size = n_estimators
    tree_dep = 1
    consistency = 0
    
    osci_score_list = []
    clf_cnt_score_list = []

    
    start = timeit.default_timer()
    y_all_pred,all_score,all_estimator,estimator_weights,estimator_errors = fit_ensemble(X_train, X_test, y_train, y_test, n_estimators,tree_dep, seed)
    stop = timeit.default_timer()
    adatime = stop - start
    
    true_y_all_pred = list(y_all_pred)
    
    clf_count, oscillation_count, oscillation_auc, label_fluc = get_base_clf(true_y_all_pred ,y_test, consistency, window_size)
    
    
    for i in range(len(X_test)):
        #Fluc Score
        osci_score = (0.5-abs((sum(oscillation_auc[i])/len(oscillation_auc[i]))-0.5))*2
        osci_score_list.append(osci_score)

        
        #Clf Cnt
        if clf_count[i] != None:
            base_clf = clf_count[i]/n_estimators
            clf_cnt_score_list.append(base_clf)
        else:
            clf_cnt_score_list.append(1)

    #Individual Error
    ind_y_all_pred = []
    for estimator in all_estimator:
        ypred = estimator.predict(X_test)
        ind_y_all_pred.append(list(ypred))
    
    ind_y_all_pred = np.transpose((np.array(ind_y_all_pred)))
    err_count, osci_count, boost_count = get_osc_cnt(ind_y_all_pred,y_test)
    err_count_lst = []


    for i in err_count.values():
        i = i/n_estimators
        err_count_lst.append(i)
        
    
    return osci_score_list, clf_cnt_score_list, err_count_lst, adatime

def lst_avg(lst):
    all_lst = [np.array(x) for x in lst]
    all_lst_avg = [round(np.mean(k),5) for k in zip(*all_lst)]
    all_lst_std = [round(np.std(k),5) for k in zip(*all_lst)]
    return all_lst_avg, all_lst_std

def linked_lst_avg(linked_lst):
    b = [[row[i] for row in linked_lst] for i in range(len(linked_lst[0]))]  
    all_lst_avg = []
    all_lst_std = []
    for i in b:
        avg, std = lst_avg(i)
        all_lst_avg.append(avg)
        all_lst_std.append(std)
    
    return all_lst_avg, all_lst_std



def get_clasify_data(samples, classes, features, sep):
    X,y = make_classification(n_samples = samples, n_features=features, n_redundant=0, n_informative=features, n_classes = classes,
                             n_clusters_per_class=1, class_sep = sep, random_state = 0)
    return X,y

def get_imb_clasify_data(samples, classes, features, sep):
    major_perc = 0.75
    minor_perc = (1-major_perc)/(classes-1)
    class_weight = [major_perc]
    for c in range(classes-1):
        class_weight.append(minor_perc)
        
    X,y = make_classification(n_samples = samples, n_features=features, n_redundant=0, n_informative=features, n_classes = classes,
                             n_clusters_per_class=1, class_sep = sep, weights = class_weight, random_state = 0)
    
    
    return X,y

def get_rbf(samples, classes, features):
    stream = RandomRBFGenerator(model_random_state=99, sample_random_state=50, n_classes=classes,n_features=features, n_centroids=50)
    data = stream.next_sample(samples)
    X = data[0]
    y = data[1]
    y = [int(i) for i in y]
    labels = list(set(y))
    if max(y) > len(labels)-1:
        y = [labels.index(i) for i in y]
    y = np.array(y)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    return X,y

def get_imb_rbf(samples, classes, features, c):
    stream = RandomRBFGenerator(model_random_state=99, sample_random_state=50, n_classes=classes,n_features=features, n_centroids=c)
    
    
    #majority class is 50% of the dataset, and the remainder is split into the remaining classes.
    data = stream.next_sample(int(samples/5))
    X = list(data[0])
    y = list(data[1])
    #print(len(y))

    label_cnt = Counter(y).most_common(1)
    #print(Counter(y))
    most_common_label = label_cnt[0]  #in the form of (label,count)
    #print(most_common_label)
    
    major_perc = 0.75
    major_cnt = int(major_perc*samples) - most_common_label[1]
    #print(major_cnt)
    total_minor_cnt = samples - int(major_perc*samples) - (len(y) - most_common_label[1])
    #print(total_minor_cnt)
    
    while major_cnt != 0:
        d = stream.next_sample()
        label = d[1][0]
        if label == most_common_label[0]:
            X.append(d[0][0])
            y.append(label)
            major_cnt -= 1
        elif total_minor_cnt != 0:
            X.append(d[0][0])
            y.append(label)
            total_minor_cnt -= 1
            
    label_cnt = Counter(y)
    #print(label_cnt)
    #print(len(y))
    X = np.array(X)
    y = [int(i) for i in y]
    labels = list(set(y))
    if max(y) > len(labels)-1:
        y = [labels.index(i) for i in y]
    y = np.array(y)
    return X,y

def get_rt(samples, classes, features):
    stream = RandomTreeGenerator(tree_random_state=8873, sample_random_state=69, n_classes=classes,
                                 n_cat_features=0, n_num_features=features, n_categories_per_cat_feature=5, max_tree_depth=6,
                                 min_leaf_depth=3, fraction_leaves_per_level=0.15)
    data = stream.next_sample(samples)
    X = data[0]
    y = data[1]
    y = [int(i) for i in y]
    
    labels = list(set(y))
    if max(y) > len(labels)-1:
        y = [labels.index(i) for i in y]
    
    y = np.array(y)
    return X,y

def get_imb_rt(samples, classes, features):
    stream = RandomTreeGenerator(tree_random_state=8873, sample_random_state=69, n_classes=classes,
                                 n_cat_features=0, n_num_features=features, n_categories_per_cat_feature=5, max_tree_depth=7,
                                 min_leaf_depth=3, fraction_leaves_per_level=0.15)
    
    #majority class is 75% of the dataset, and the remainder is split into the remaining classes.
    data = stream.next_sample(int(samples/5))
    X = list(data[0])
    y = list(data[1])
    #print(len(y))

    label_cnt = Counter(y).most_common(1)
    #print(Counter(y))
    most_common_label = label_cnt[0]  #in the form of (label,count)
    #print(most_common_label)
    
    major_perc = 0.75
    major_cnt = int(major_perc*samples) - most_common_label[1]
    #print(major_cnt)
    total_minor_cnt = samples - int(major_perc*samples) - (len(y) - most_common_label[1])
    #print(total_minor_cnt)
    
    while major_cnt != 0:
        d = stream.next_sample()
        label = d[1][0]
        if label == most_common_label[0]:
            X.append(d[0][0])
            y.append(label)
            major_cnt -= 1
        elif total_minor_cnt != 0:
            X.append(d[0][0])
            y.append(label)
            total_minor_cnt -= 1
            
    label_cnt = Counter(y)
    #print(label_cnt)
    #print(len(y))
    X = np.array(X)
    y = [int(i) for i in y]
    labels = list(set(y))
    if max(y) > len(labels)-1:
        y = [labels.index(i) for i in y]
    y = np.array(y)
    return X,y


def similarity_func(u, v):
    a = set(u)
    b = set(v)
    return (len(a.intersection(b))/len(u))*100

def remove_samples(score,noise_cnt,X_train,y_train):
    #score = normalize(score)
    instances_to_remove = noise_cnt
    #print(score)
    
    #sort instances smallest to largest based on their scores
    sorted_score_idx = np.argsort(score)
    
    #print(sorted_score_idx)
    #remove top k difficulty instances
    instances_kept = sorted_score_idx[:-instances_to_remove]
    #print(instances_kept)
    instances_removed = sorted_score_idx[-instances_to_remove:]
    
    new_X_train = []
    new_y_train= []
    for i in instances_kept:
        new_X_train.append(X_train[i])
        new_y_train.append(y_train[i])
        
    #print(len(new_y_train))
    #print("Samples Removed:", instances_removed)
    return new_X_train,new_y_train, instances_removed


def add_dif_noise(instances_to_noise_idx,X_train,y_train, noise_type):
    
    random.seed(0)
    class_cnt = len(Counter(y_train))

    #change the instances at the given interval into noise
    instances_made_dif = instances_to_noise_idx

    
    #Add Label Noise
    if noise_type == "label": 
        new_y_train = y_train.copy()
        
        for i in instances_made_dif:
            orig_label = y_train[i]
            rand_label = random.randint(0, class_cnt-1)
            
            while rand_label == orig_label:
                rand_label = random.randint(0, class_cnt-1)
            
            new_y_train[i] = rand_label
            
        """
        for i in instances_made_dif:
            print("Original", y_train[i])
            print("Noisy",new_y_train[i])
        print("")
        for i in instances_kept:
            print("Original", y_train[i])
            print("Noisy",new_y_train[i])
        """
        return X_train,new_y_train, instances_made_dif
    #Add Attribute Noise        
    else:
        new_X_train = []
        possilbe_attributes = np.transpose(X_train)
        #selected_noisy_attr = len(possilbe_attributes)

        for attr in range(len(possilbe_attributes)):
            noisy_attr = possilbe_attributes[attr].copy()
            #print(noisy_attr)
            max_value = max(possilbe_attributes[attr])
            min_value = min(possilbe_attributes[attr])
            if min_value == max_value:
                max_value = min_value + 1
                
            for i in instances_made_dif:
                noisy_attr[i] = random.uniform(max_value, max_value + 2*(max_value-min_value))        
                while noisy_attr[i] == max_value:
                    noisy_attr[i] = random.uniform(max_value, max_value + 2*(max_value-min_value))
                                   
            new_X_train.append(noisy_attr)
            
        new_X_train = np.array(new_X_train)
        new_X_train = np.transpose(np.array(new_X_train))

        """
        for i in instances_made_dif:
            print("Original", X_train[i])
            print("Noisy",new_X_train[i])
        print("")
        for i in instances_kept:
            print("Original", X_train[i])
            print("Noisy",new_X_train[i])
        """
        return new_X_train,y_train, instances_made_dif
    

def cv_difficulty_scores(X,y,fold, seed, n_estimators):
    f = []
    c = []
    ie = []
    adatime = []
    skf = KFold(n_splits=fold, shuffle=True, random_state=seed)
    score_idx = []
    for train_index, test_index in skf.split(X, y):
        x_train_fold, x_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        score_idx.extend(test_index)
        
        temp_f, temp_c, temp_ie, temp_adatime = get_osci_score_small(x_train_fold, x_test_fold, y_train_fold, y_test_fold, n_estimators, seed)
        f.extend(temp_f)
        c.extend(temp_c)
        ie.extend(temp_ie)
        adatime.append(temp_adatime)
    
    f = [x for _, x in sorted(zip(score_idx, f))]
    c = [x for _, x in sorted(zip(score_idx, c))]
    ie = [x for _, x in sorted(zip(score_idx, ie))]
    return f, c, ie, adatime

def cv_instance_hardness(X,y,fold, seed):
    freq= []
    clf_score = []
    skf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=seed)
    for train_index, test_index in skf.split(X, y):
        x_train_fold, x_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        
        temp_freq, temp_clf_score = get_scores(x_train_fold, x_test_fold, y_train_fold, y_test_fold, seed)
        freq.extend(temp_freq)
        clf_score.extend(temp_clf_score)

    return freq,clf_score
    
        

dataset_names = []
all_samples_removed = []

for data in range(52,60):
#for data in range(1,2):
    metric = "kappa"
    if data == 1:
        X, y = UCI_Dataset_Loader.iris() #k = 50 d= 1
        name = "Iris"
    elif data == 2:
        X, y = UCI_Dataset_Loader.ecoli()
        name = "E.coli"
    elif data == 3:
        X, y = UCI_Dataset_Loader.car()
        name = "Car"
    elif data == 4:
        X, y = UCI_Dataset_Loader.spam()
        name = "Spam"
    elif data == 5:
        X, y = UCI_Dataset_Loader.yeast()
        name = "Yeast"
    elif data == 6:
        X, y = UCI_Dataset_Loader.abalone()
        name = "Abalone"
        
    elif data == 7:
        continue
        X, y = UCI_Dataset_Loader.nursery()
        name = "Nursery"
    elif data == 8:
        continue
        X, y = UCI_Dataset_Loader.adult()
        name = "Adult"
    elif data == 9:
        continue
        X, y = UCI_Dataset_Loader.magic()
        name = "Magic"
        
    elif data == 10:
        X, y = UCI_Dataset_Loader.segment()
        name = "Segment"
    elif data == 11:
        X, y = UCI_Dataset_Loader.heart_statlog()
        name = "Heart"
    elif data == 12:
        X, y = UCI_Dataset_Loader.balance_scale()
        name = "Balance"
    elif data == 13:
        X, y = UCI_Dataset_Loader.glass()
        name = "Glass"
    elif data == 14:
        X, y = UCI_Dataset_Loader.wine()
        name = "Wine"
    elif data == 15:
        X, y = UCI_Dataset_Loader.dermatology()
        name = "Derma"
    elif data == 16:
        X, y = UCI_Dataset_Loader.haberman()
        name = "Haberman"
    elif data == 17:
        X, y = UCI_Dataset_Loader.ionosphere()
        name = "Ionosphere"
    elif data == 18:
        X, y = UCI_Dataset_Loader.seismic()
        name = "Seismic"
    elif data == 19:
        X, y = UCI_Dataset_Loader.teaching_assistant()
        name = "Teaching"        
    elif data == 20:
        X, y = UCI_Dataset_Loader.website_phishing()
        name = "Phishing"
    elif data == 21:
        X, y = UCI_Dataset_Loader.wholesale_customers()
        name = "Wholesale"
    elif data == 22:
        X, y = UCI_Dataset_Loader.mushroom()
        name = "Mushroom"        
    elif data == 23:
        X, y = UCI_Dataset_Loader.thyroid()
        name = "Thyroid"        
    elif data == 24:
        X, y = UCI_Dataset_Loader.drybean()
        name = "Drybean"   
    elif data == 25:
        X, y = UCI_Dataset_Loader.codon()
        name = "Codon"
    
    elif data == 30:
        X,y = get_rt(5000, 50, 10)
        name = "RT1"
        
    elif data == 31:
        X,y = get_rt(5000, 100, 10)
        name = "RT2"
    
    elif data == 32:
        X,y = get_rbf(5000, 50, 10)
        name = "RBF1"
        
    elif data == 33:
        X,y = get_rbf(5000, 100, 10)
        name = "RBF2"
       
    
    elif data == 34:
        X,y = get_imb_rt(5000, 50, 10)
        name = "RT1 Imb"
        
    elif data == 35:
        X,y = get_imb_rt(5000, 100, 10)
        name = "RT2 Imb"
    
    elif data == 36:
        X,y = get_imb_rbf(5000, 50, 10, 50)
        name = "RBF1 Imb"
    
    elif data == 37:
        X,y = get_imb_rbf(5000, 100, 10, 50)
        name = "RBF2 Imb"
        
    elif data == 51:
        X, y = UCI_Dataset_Loader.balance_scale()
        name = "Balance"
    elif data == 52:
        X, y = UCI_Dataset_Loader.thyroid()
        name = "Thyroid"   
    elif data == 53:
        X, y = UCI_Dataset_Loader.drybean()
        name = "Drybean"   
    
    elif data == 54:
        X,y = get_rt(5000, 100, 10)
        name = "RT2"
    elif data == 55:
        X,y = get_imb_rt(5000, 100, 10)
        name = "RT2 Imb"
    
    elif data == 56:
        X,y = get_rbf(5000, 100, 10)
        name = "RBF2"
    elif data == 57:
        X,y = get_imb_rbf(5000, 100, 10, 50)
        name = "RBF2 Imb"
    
    else:
        continue
    
    
    #noise generation methods
    measure_names = ["R", "dNN", "C", "F", "IE"]
    
    #filtering methods
    filter_names = ["Freq", "Par", "kDN", "RF"]
    
    seeds = [0,1,2,3,4,5,6,7,8,9]
    seeds = [0,1,2,3,4]
    #seeds = [0,1]
    n_estimators = 300
    fold = 5
    k = 5
    #threshold= [0.05, 0.10, 0.15, 0.20, 0.25]
    threshold= [0.1, 0.2 ,0.33, 0.5]
    #threshold = [0.2]
    #t = 0.2
    for t in threshold:
            
        interval_counts = int(1/t)
        print("")
        print("Dataset:", name)
    
        #noise_type = "feature"
        noise_type = "label"
        
        repeated_orig_dt_acc = []
        
        repeated_dt_noisy_performances = []
        repeated_dt_best_performances = []
        
        repeated_detector_performances_gain_all = []
        repeated_detector_performances_all = []
        repeated_detector_prec_all = []
        repeated_detector_recall_all = []
        repeated_detector_f1_all = []
        
        x_axis = []
    
        for global_rand_seed in seeds:
            print("Repeat:", global_rand_seed)
            
            #labels = Counter(y)
            #print(len(X), len(f))
            dnn = get_dNN(X,y)
            _, __, dnn, _ = split_data(X, dnn, global_rand_seed)
            
            #split into dataset and scores into train and test
            orig_X_train, X_test, orig_y_train, y_test = split_data(X,y, global_rand_seed)
    
            
            #Calculate instance difficulty scores
            f, c, ie, adatime = cv_difficulty_scores(orig_X_train,orig_y_train,fold, global_rand_seed, n_estimators)
            
            #get random scores (between 0-1)
            random.seed(global_rand_seed)
            rand_scores = []
            for i in range(len(f)):
                rand_scores.append(random.uniform(0, 1))
    
            
            #For each difficulty scores, select the top K instacnes as the critical instances to be into noise
            difficulty_scores = [rand_scores, dnn, c, f, ie]
            
            #For each difficulty score: sort the score smallest to largest and return index
            sorted_difficulty_scores_idx = [np.argsort(score) for score in difficulty_scores]
            
            #For sorted score idx: split into 10 even intervals while keeping order
            sorted_difficulty_scores_intervals = [np.array_split(score, interval_counts) for score in sorted_difficulty_scores_idx]
            
            dt_acc_orig= decisiontree_acc(orig_X_train, X_test, orig_y_train, y_test, global_rand_seed, metric)
            repeated_orig_dt_acc.append(dt_acc_orig)
            #print(dt_acc_orig)
            
            detector_performances_gain_all_intervals = []
            detector_performances_all_intervals = []
            detector_prec_all_intervals = []
            detector_recall_all_intervals = []
            detector_f1_all_intervals = []
            dt_noisy_performances_all_intervals = []
            dt_best_performances_all_intervals = []
    
            for window in range(0,interval_counts):
                if global_rand_seed == 0:
                    interval = round(t + window*t, 2)
                    x_axis.append(interval)
                if window != interval_counts-1:
                    print(window, end = '')
                else:
                    print(window)
                
                #The performance of classifier for all noisy instance generation methods
                detector_performances_gain_all = []
                detector_performances_all = []
                detector_prec_all = []
                detector_recall_all = []
                detector_f1_all = []
                dt_noisy_performances = []
                dt_best_performances = []
        
                
                for s in range(len(difficulty_scores)):
                    score = sorted_difficulty_scores_intervals[s]
                    #_, __, score_train, score_test = split_data(X, score, global_rand_seed)
                    instances_to_noise_idx = score[window]
                    noisy_X_train, noisy_y_train, instances_modified = add_dif_noise(instances_to_noise_idx,orig_X_train,orig_y_train, noise_type)
        
                    
                    #get performance of classfier with noisy training set
                    dt_acc_noisy = decisiontree_acc(noisy_X_train, X_test, noisy_y_train, y_test, global_rand_seed, metric)
                    dt_noisy_performances.append(dt_acc_noisy)
                    
                    #Get Index of instances that are modified
                    noisy_idx = [0]*len(orig_y_train)
                    for i in instances_modified:
                            noisy_idx[i] = 1
                    noise_cnt = len(instances_modified)
                    
                    
                    #get noise detector scores                
                    #freq, clf_score = cv_instance_hardness(noisy_X_train, noisy_y_train, 5, global_rand_seed)
                    #clf_score = [1 - i for i in clf_score]
                    #print("freq:",freq)
                    freq = InstanceHardness(random_state = global_rand_seed).detect(noisy_X_train, noisy_y_train)
                    freq = [1-i for i in freq]
                    kdn = KDN().detect(noisy_X_train, noisy_y_train)
                    #print("kdn:",kdn)
                    kdn = [1-i for i in kdn]
                    #mcs = MCS().detect(noisy_X_train, noisy_y_train)
                    rf = RandomForestDetector(random_state = global_rand_seed).detect(noisy_X_train, noisy_y_train)
                    rf = [1-i for i in rf]
                    #ForestKdn = ForestKDN(random_state = global_rand_seed).detect(noisy_X_train, noisy_y_train)
                    partition = PartitioningDetector(random_state = global_rand_seed).detect(noisy_X_train, noisy_y_train)
                    partition = [1-i for i in partition]
                    
                    #best filtering method
                    true_clean_X_train, true_clean_y_train, instances_removed = remove_samples(noisy_idx,noise_cnt,noisy_X_train,noisy_y_train)
                    dt_acc_best = decisiontree_acc(true_clean_X_train, X_test, true_clean_y_train, y_test, global_rand_seed, metric)
                    dt_best_performances.append(dt_acc_best)
                    #print(instances_removed, instances_modified)
                    
                    detector_scores = [freq, partition, kdn, rf]
                    detector_prec = []
                    detector_recall = []
                    detector_f1 = []
                    detector_performances = []
                    detector_performances_gain = []
                    
                    for detector in detector_scores:
                        #Filter top k instances with highest scores
                        sudo_clean_X_train, sudo_clean_y_train, instances_removed = remove_samples(detector,noise_cnt,noisy_X_train,noisy_y_train)
                        
                        #Get index of instacnes that are removed
                        predicted_noisy_idx = [0]*len(orig_y_train)
                        for i in instances_removed:
                            predicted_noisy_idx[i] = 1
                        
                        prec = precision_score(noisy_idx,predicted_noisy_idx)
                        recall = recall_score(noisy_idx,predicted_noisy_idx)
                        f1 = f1_score(noisy_idx,predicted_noisy_idx)
                        detector_prec.append(prec)
                        detector_recall.append(recall)
                        detector_f1.append(f1)
                        
                        #get performance of classfier with filtered training set
                        dt_acc_clean = decisiontree_acc(sudo_clean_X_train, X_test, sudo_clean_y_train, y_test, global_rand_seed, metric)
                        detector_performances.append(dt_acc_clean)
                        
                        #get performance gain
                        base_dif = dt_acc_best - dt_acc_noisy
                        #print(s, base_dif)
                        if base_dif == 0:
                            #print(dt_acc_best,dt_acc_noisy)
                            base_dif = 0.01
                        dt_acc_gain = (dt_acc_clean - dt_acc_noisy)/base_dif
                        detector_performances_gain.append(dt_acc_gain)
                    
                    
                    detector_performances_gain_all.append(detector_performances_gain)
                    detector_performances_all.append(detector_performances)
                    detector_prec_all.append(detector_prec)
                    detector_recall_all.append(detector_recall)
                    detector_f1_all.append(detector_f1)
                
                detector_performances_gain_all_intervals.append(detector_performances_gain_all)    
                detector_performances_all_intervals.append(detector_performances_all)
                detector_prec_all_intervals.append(detector_prec_all)
                detector_recall_all_intervals.append(detector_recall_all)
                detector_f1_all_intervals.append(detector_f1_all)
                
                dt_noisy_performances_all_intervals.append(dt_noisy_performances)
                dt_best_performances_all_intervals.append(dt_best_performances)
                
    
            repeated_dt_noisy_performances.append(dt_noisy_performances_all_intervals)
            repeated_dt_best_performances.append(dt_best_performances_all_intervals)    
            
            repeated_detector_performances_gain_all.append(detector_performances_gain_all_intervals)
            repeated_detector_performances_all.append(detector_performances_all_intervals)
            repeated_detector_prec_all.append(detector_prec_all_intervals)
            repeated_detector_recall_all.append(detector_recall_all_intervals)
            repeated_detector_f1_all.append(detector_f1_all_intervals)
            
        
        
    
        #get mean
        orig_dt_acc_avg = np.mean(repeated_orig_dt_acc)
        dt_noisy_performances_avg = np.mean(repeated_dt_noisy_performances,0)
        dt_best_performances_avg = np.mean(repeated_dt_best_performances,0)
        
        detector_performances_gain_all_avg = np.mean(repeated_detector_performances_gain_all,0)
        detector_performances_all_avg  = np.mean(repeated_detector_performances_all,0)
        detector_prec_all_avg = np.mean(repeated_detector_prec_all,0)
        detector_recall_all_avg = np.mean(repeated_detector_recall_all,0)
        detector_f1_all_avg = np.mean(repeated_detector_f1_all,0)
        
        #get std
        orig_dt_acc_std = np.std(repeated_orig_dt_acc)
        dt_noisy_performances_std = np.std(repeated_dt_noisy_performances,0)
        dt_best_performances_std = np.std(repeated_dt_best_performances,0)
        
        detector_performances_gain_all_std = np.std(repeated_detector_performances_gain_all,0)
        detector_performances_all_std  = np.std(repeated_detector_performances_all,0)
        detector_prec_all_std = np.std(repeated_detector_prec_all,0)
        detector_recall_all_std = np.std(repeated_detector_recall_all,0)
        detector_f1_all_std = np.std(repeated_detector_f1_all,0)
        
        threshold_name = str(int(t*100))
        
        stdoutOrigin=sys.stdout 
        sys.stdout = open("print_log " + str(t) + ".txt", "a")
        print("Dataset:", name, "Threshold", threshold_name)
        print("Orig DT Acc", np.format_float_positional(orig_dt_acc_avg), np.format_float_positional(orig_dt_acc_std))
        
        for interval_size in range(len(detector_performances_all_avg)):
            interval_noisy_performances_avg = dt_noisy_performances_avg[interval_size]
            interval_noisy_performances_std = dt_noisy_performances_std[interval_size]
            
            interval_best_performances_avg = dt_best_performances_avg[interval_size]
            interval_best_performances_std = dt_best_performances_std[interval_size]
            
            interval_performance_gain_avg = detector_performances_gain_all_avg[interval_size]
            interval_performance_gain_std = detector_performances_gain_all_std[interval_size]
            
            interval_performance_avg = detector_performances_all_avg[interval_size]
            interval_performance_std = detector_performances_all_std[interval_size]
            
            interval_prec_avg = detector_prec_all_avg[interval_size]
            interval_prec_std = detector_prec_all_std[interval_size]
            
            interval_recall_avg = detector_recall_all_avg[interval_size]
            interval_recall_std = detector_recall_all_std[interval_size]
            
            interval_f1_avg = detector_f1_all_avg[interval_size]
            interval_f1_std = detector_f1_all_std[interval_size]
        
            print("Interval:", interval_size)
            for generation_method in range(len(interval_performance_avg)):
                f_name = measure_names[generation_method]
                print(f_name+"_best", to_str_rnd(interval_best_performances_avg[generation_method]), to_str_rnd(interval_best_performances_std[generation_method]))
                print(f_name+"_noisy", to_str_rnd(interval_noisy_performances_avg[generation_method]), to_str_rnd(interval_noisy_performances_std[generation_method]))
                
                print(f_name+"_performance_gain_avg", ' '.join(map(to_str_rnd,interval_performance_gain_avg[generation_method])))
                print(f_name+"_performance_gain_std", ' '.join(map(to_str_rnd,interval_performance_gain_std[generation_method])))
                
                print(f_name+"_performance_avg", ' '.join(map(to_str_rnd,interval_performance_avg[generation_method])))
                print(f_name+"_performance_std", ' '.join(map(to_str_rnd,interval_performance_std[generation_method])))
                
                print(f_name+"_prec_avg", ' '.join(map(to_str_rnd,interval_prec_avg[generation_method])))
                print(f_name+"_prec_std", ' '.join(map(to_str_rnd,interval_prec_std[generation_method])))
                
                print(f_name+"_recall_avg", ' '.join(map(to_str_rnd,interval_recall_avg[generation_method])))
                print(f_name+"_recall_std", ' '.join(map(to_str_rnd,interval_recall_std[generation_method])))
                
                print(f_name+"_f1_avg", ' '.join(map(to_str_rnd,interval_f1_avg[generation_method])))
                print(f_name+"_f1_std", ' '.join(map(to_str_rnd,interval_f1_std[generation_method])))
            print("")
            
        sys.stdout.close()
        sys.stdout=stdoutOrigin
        
        #x_axis_label = [.1,.2,.3,.4,.5,.6,.7,.8,.9,1.]
        x_axis_label = x_axis
        acc_avg = np.array([orig_dt_acc_avg] * len(x_axis_label))
        acc_std = np.array([orig_dt_acc_std] * len(x_axis_label))
        
        dt_noisy_performances_avg = np.transpose(dt_noisy_performances_avg)
        dt_noisy_performances_std = np.transpose(dt_noisy_performances_std)
        
        plt.figure(figsize=(5,3))
        plt.fill_between(x_axis_label, acc_avg - acc_std,  acc_avg + acc_std, color = 'k', alpha = 0.2)
        plt.plot(x_axis_label, acc_avg, color = 'k', linestyle='dashed' , linewidth=2, alpha = 0.5)
        for generation_method in range(len(dt_noisy_performances_avg)):
            f_name = measure_names[generation_method]
            if f_name == "R" or f_name =="dNN":
                l = '-.'
            else:
                l = '-'
            y_avg = dt_noisy_performances_avg[generation_method]
            y_std = dt_noisy_performances_std[generation_method]
            plt.errorbar(x_axis,y_avg, yerr=y_std, marker = 's', capsize=6, label = f_name, linestyle = l)
            
            #print(f_name+"_best", to_str_rnd(interval_best_performances_avg[generation_method]), to_str_rnd(interval_best_performances_std[generation_method]))
            #print(f_name+"_noisy", to_str_rnd(interval_noisy_performances_avg[generation_method]), to_str_rnd(interval_noisy_performances_std[generation_method])) 
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel('Difficulty Level', fontsize=20)
        plt.ylabel('Kappa', fontsize=20)
        #plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0, fontsize=16)
        plt.title("Dataset: " + name + "  Noise Level: " + str(t), fontsize=20)
        plt.savefig('D:/UNi/Thesis/noise generation/plots/' + name + " " + str(t) +'.png', bbox_inches="tight")
        plt.show()
        
        detector_performances_gain_all_avg = np.transpose(detector_performances_gain_all_avg,(2,1,0))
        detector_performances_gain_all_std = np.transpose(detector_performances_gain_all_std,(2,1,0))
        
        fig, axs = plt.subplots(1,len(filter_names), figsize=(12, 2.5), sharex=True, sharey=True)
        axs = axs.ravel()
        sub_plt_cnt = 0
        for filter_method in range(len(detector_performances_gain_all_avg)):
            ax = axs[sub_plt_cnt]
            #ax.set_ylim([-1.5, 1.5])
            f_name = filter_names[filter_method]
            performance_gain_all_method_avg = detector_performances_gain_all_avg[filter_method]
            performance_gain_all_method_std = detector_performances_gain_all_std[filter_method]
            
            for generation_method in range(len(performance_gain_all_method_avg)):
                d_name = measure_names[generation_method]
                if d_name == "R" or d_name =="dNN":
                    l = '-.'
                else:
                    l = '-'
                y_avg = performance_gain_all_method_avg[generation_method]
                y_std = performance_gain_all_method_std[generation_method]
                ax.errorbar(x_axis,y_avg, yerr=y_std, marker = 's', capsize=6, label = d_name, linestyle = l)
            #ax.set_xticks(x_axis)
            ax.tick_params(axis='both', which='major', labelsize=18)
            ax.set_title(f_name, fontsize = 20)
                
            sub_plt_cnt += 1
            
        fig.suptitle("Dataset: " + name  +"  Noise Level: "+ str(t), fontsize = 20, y = 1.1)  
        plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0, fontsize = 18)
        fig.text(0.5, -0.1, 'Difficulty Level', ha='center', fontsize=20)
        fig.text(0.05, 0.5, 'Kappa Gain', va='center', rotation='vertical', fontsize=20)
        plt.savefig('D:/UNi/Thesis/noise generation/plots/' + name + " Filter " + str(t) +'.png', bbox_inches="tight")
        plt.show()
        
        
            
        
        
            
            
            
        































