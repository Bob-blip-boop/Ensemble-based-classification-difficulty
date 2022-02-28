import numpy as np
import numpy.matlib
from uci_utils import *
from collections import Counter
import random
import matplotlib.pyplot as plt
import sys 

import pandas as pd
import seaborn as sns
from sklearn.utils import shuffle

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
    n = np.format_float_positional(n, precision=3)
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
    
def kdn_score(X, X_test, y, y_test, k):
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='kd_tree').fit(X)
    _, indices = nbrs.kneighbors(X_test)
    neighbors = indices[:, 1:]
    diff_class = np.matlib.repmat(y_test, k, 1).transpose() != y[neighbors]
    score = np.sum(diff_class, axis=1) / k
    return score

def decisiontree(X_train, X_test, y_train, y_test, global_rand_seed):
    model = DecisionTreeClassifier(random_state = global_rand_seed)
    model.fit(X_train,y_train)
    predicted_label = model.predict(X_test)
    
    probability_estimates = []
    y_pred = model.predict_proba(X_test)
    for i in y_pred:
        probability_estimates.append(max(i))
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

def cv_filter_decisiontree_acc(X,y, seeds, metric, noise_idx):
    X = np.array(X)
    y = np.array(y)
    repeated_best_acc = []
    for global_rand_seed in seeds:
        kf = KFold(n_splits=5, shuffle=True, random_state=global_rand_seed)
        dt_acc_best = []
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            noise_idx_split = [noise_idx[i] for i in train_index]
            #print(noise_idx_split)
            clean_x = []
            clean_y = []
            for i in range(len(noise_idx_split)):
                if noise_idx_split[i] == 0:
                    clean_x.append(X_train[i])
                    clean_y.append(y_train[i])
                    
            #print(len(clean_x))      
                
            model = DecisionTreeClassifier(random_state = global_rand_seed)
            #start = timeit.default_timer()
            model.fit(clean_x,clean_y)
            predicted_label = model.predict(X_test)
            #stop = timeit.default_timer()
            #time = stop - start
    
            if metric == "kappa":
                kappa = cohen_kappa_score(y_test, predicted_label)
                dt_acc_best.append(kappa)
                
            else:
                accuracy = accuracy_score(y_test, predicted_label)
                dt_acc_best.append(accuracy)
        best_acc = np.mean(dt_acc_best)
        repeated_best_acc.append(best_acc)
    acc_avg = np.mean(repeated_best_acc)
    acc_std = np.std(repeated_best_acc)
    return acc_avg,acc_std

    
    

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
    final_pred = true_y_all_pred[-1]
    kappa = cohen_kappa_score(y_test, final_pred)
    
    
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
        
    
    return osci_score_list, clf_cnt_score_list, err_count_lst, adatime, kappa

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


def remove_samples(score,noise_cnt,X_train,y_train):
    #score = normalize(score)
    instances_to_remove = noise_cnt
    #print(score)
    
    score = normalize(score)
    
    #sort instances smallest to largest based on their scores  
    random_order = np.random.randint(len(score), size = len(score))
    sorted_score_idx = np.lexsort((random_order,score))    
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
    
def add_noise(y, noise_perc):
    random.seed(0)
    class_cnt = len(Counter(y))
    noise_cnt = int(noise_perc*len(y))
    randomlist = random.sample(range(0, len(y)), noise_cnt)
    
    instances_modified = randomlist

    new_y = y.copy()
    for i in randomlist:
        rand_label = random.randint(0, class_cnt-1)
        while new_y[i] == rand_label:
            rand_label = random.randint(0, class_cnt-1)
        new_y[i] = rand_label
    
    return new_y, instances_modified

def add_attr_noise(X, noise_perc, attribute_perc):
    new_x = []
    random.seed(0)
    noise_cnt = int(noise_perc*len(y))
    randomlist = random.sample(range(0, len(X)), noise_cnt)
    instances_modified = randomlist
    #print(randomlist)
    
    possilbe_attributes = np.transpose(X)
    #print(len(possilbe_attributes))
    attribute_perc = int(attribute_perc*len(possilbe_attributes))
    selected_noisy_attr = random.sample(list(range(0,len(possilbe_attributes))),attribute_perc)
    #print(attribute_perc)
    #print(selected_noisy_attr)
    #print(selected_noisy_attr)
    count = 0
    for attr in range(len(possilbe_attributes)):
        if attr in selected_noisy_attr:
            count+= 1
            noisy_attr = possilbe_attributes[attr].copy()
            #unique_attributes = list(set(possilbe_attributes[attr]))
            #print(unique_attributes)
            max_value = max(possilbe_attributes[attr])
            min_value = min(possilbe_attributes[attr])
            if min_value == max_value:
                max_value = min_value + 1
            
            for i in randomlist: 
                noisy_attr[i] = random.uniform(max_value, max_value + 2*(max_value-min_value))
                #print(max_value, max_value + 2*(max_value-min_value))
                while noisy_attr[i] == max_value:
                    noisy_attr[i] = random.uniform(max_value, max_value + 2*(max_value-min_value))   
            new_x.append(noisy_attr)
        else:
            new_x.append(possilbe_attributes[attr])

    new_x = np.transpose(new_x)
    
    """
    for i in randomlist:
        print("Original", X[i])
        print("Noisy",new_x[i])
    """
    return new_x, instances_modified

def cv_difficulty_scores(X,y,fold, seed, n_estimators):
    f = []
    c = []
    ie = []
    adatime = []
    kappa_all = []
    skf = KFold(n_splits=fold, shuffle=True, random_state=seed)
    score_idx = []
    for train_index, test_index in skf.split(X, y):
        x_train_fold, x_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        score_idx.extend(test_index)
        
        temp_f, temp_c, temp_ie, temp_adatime, kappa = get_osci_score_small(x_train_fold, x_test_fold, y_train_fold, y_test_fold, n_estimators, seed)
        f.extend(temp_f)
        c.extend(temp_c)
        ie.extend(temp_ie)
        adatime.append(temp_adatime)
        kappa_all.append(kappa)
    

    f = [x for _, x in sorted(zip(score_idx, f))]
    c = [x for _, x in sorted(zip(score_idx, c))]
    ie = [x for _, x in sorted(zip(score_idx, ie))]
    kappa_avg = np.mean(kappa_all)
    return f, c, ie, adatime, kappa_avg

def cv_instance_hardness(X,y,fold, seed):
    freq= []
    clf_score = []
    skf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=seed)
    for train_index, test_index in skf.split(X, y):
        x_train_fold, x_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        
        temp_freq, temp_clf_score = get_scores(x_train_fold, x_test_fold, y_train_fold, y_test_fold, seed)
        temp_clf_score = [(1 - i) for i in temp_clf_score]
        freq.extend(temp_freq)
        clf_score.extend(temp_clf_score)

    return freq,clf_score
    
        
dataset_names = []
all_samples_removed = []

for data in range(57,60):
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
    
    
    
    
    seeds = [0,1,2,3,4,5,6,7,8,9]
    n_estimators = 300
    fold = 5
    threshold= [0.05, 0.10, 0.20, 0.40]
    #threshold= [0.10, 0.20, 0.40]
    #threshold = [0.05]

    noise_lvl = [0.05, 0.10, 0.20, 0.40]
    noise_type = "feature"
    #noise_type = "label"
    
    #clean_X = X.copy()
    #clean_y = y.copy()
    
    
    fig, axs = plt.subplots(1,len(noise_lvl), figsize=(18, 3), sharex=True, sharey=True)
    axs = axs.ravel()
    sub_plt_cnt = 0
    
    for nl in noise_lvl:   
        if noise_type == "label":
            y, instances_modified = add_noise(y, noise_perc = nl)
        elif noise_type == "feature":
            X, instances_modified = add_attr_noise(X, noise_perc = nl, attribute_perc = 1)
        
        #Get index of instacnes that are modified
        noisy_idx = [0]*len(X)
        for i in instances_modified:
            noisy_idx[i] = 1
        
        
        dt_best_acc_avg,dt_best_acc_std = cv_filter_decisiontree_acc(X,y, seeds, metric, noisy_idx)
        
 
        print("")
        print("Dataset:", name, "Noise Level:", nl)
        #filtering methods
        filter_names = ["R", "Freq", "kDN", "RF", "C", "F", "IE", "Best"]
        repeated_orig_dt_acc_allt = []
        repeated_ada_score_allt = []
        repeated_filterd_dt_acc_allt = []
        repeated_filter_acc_allt = []
        for t in threshold:
            print("Threshold:", t)
            repeated_orig_dt_acc = []
            repeated_filtered_dt_acc = []
            repeated_filter_acc = []
            repeated_ada_score = []
        
            for global_rand_seed in seeds:
                if global_rand_seed != seeds[-1]:
                    print(global_rand_seed, end = '')
                else:
                    print(global_rand_seed)
                    
          
                #Calculate kDN scores
                kdn = KDN().detect(X, y)
                kdn = [1-i for i in kdn]
                
                #Calculate freq and clf scores
                #freq, clf_score = cv_instance_hardness(X, y, fold, global_rand_seed)
                freq = InstanceHardness(random_state = global_rand_seed).detect(X, y)
                freq = [1-i for i in freq]
                
                rf = RandomForestDetector(random_state = global_rand_seed).detect(X, y)
                rf = [1-i for i in rf]
        
                #Calculate instance difficulty scores
                f, c, ie, adatime, ada_score = cv_difficulty_scores(X, y, fold, global_rand_seed, n_estimators)
                repeated_ada_score.append(ada_score)
                
                #get random scores (between 0-1)
                np.random.seed(global_rand_seed)
                rand_scores = np.random.uniform(0, 1, len(y))
                
                best = noisy_idx
                filter_scores = [rand_scores, freq, kdn, rf, c, f, ie, best]
                    
                idx = list(range(len(y)))                        
                #split into dataset and scores into train and test
                kf = KFold(n_splits=5, shuffle=True, random_state=global_rand_seed)
                dt_acc_orig = []
                filter_performance = [[] for _ in range(len(filter_scores))]
                filter_acc = [[] for _ in range(len(filter_scores))]
                
                for train_index, test_index in kf.split(X, y):
                    orig_X_train, X_test = X[train_index], X[test_index]
                    orig_y_train, y_test = y[train_index], y[test_index]
                    
                    noisy_idx_split = [noisy_idx[i] for i in train_index]
                    
                               
                    if t == 0.05:
                        dt_acc_orig_temp = decisiontree_acc(orig_X_train, X_test, orig_y_train, y_test, global_rand_seed, metric)
                        dt_acc_orig.append(dt_acc_orig_temp)
                    
                    filter_cnt = int(t*len(orig_y_train))
                    filter_performance_temp = []
                    filter_acc_temp = []
                    #For each difficulty scores, select the top K instacnes as instances to be removed
                    for f_score in range(len(filter_scores)):                
                        detector = [filter_scores[f_score][i] for i in train_index]
                        sudo_clean_X_train, sudo_clean_y_train, instances_removed = remove_samples(detector,filter_cnt,orig_X_train,orig_y_train)
                        
                        #Get index of instacnes that are removed
                        correctly_removed_cnt = 0
                        for i in instances_removed:
                            if noisy_idx_split[i] == 1:
                                correctly_removed_cnt+=1
                        detector_acc= correctly_removed_cnt/filter_cnt
                        filter_acc_temp.append(detector_acc)
                                
                        
                        #get performance of classfier with filtered training set
                        dt_acc_clean_temp = decisiontree_acc(sudo_clean_X_train, X_test, sudo_clean_y_train, y_test, global_rand_seed, metric)
                        """
                        if f_score == 1 or f_score == 4:                        
                            print(filter_names[f_score])
                            print(dt_acc_clean_temp)
                            print(Counter(list(sudo_clean_y_train)))
                            print("")
                        """
                        filter_performance_temp.append(dt_acc_clean_temp)
                        
                    for temp_f in range(len(filter_scores)):
                        filter_performance[temp_f].append(filter_performance_temp[temp_f])
                        filter_acc[temp_f].append(filter_acc_temp[temp_f])
                    
                #print(filter_performance)
                filter_performance = [np.mean(i) for i in filter_performance]
                #print(filter_performance)
                filter_acc = [np.mean(i) for i in filter_acc]
                dt_acc_orig = np.mean(dt_acc_orig)
                repeated_orig_dt_acc.append(dt_acc_orig)  
                
                repeated_ada_score.append(ada_score)
                repeated_filtered_dt_acc.append(filter_performance)
                repeated_filter_acc.append(filter_acc)
            
            if t == 0.05:
                repeated_orig_dt_acc_allt.append(repeated_orig_dt_acc)
                repeated_ada_score_allt.append(repeated_ada_score)
            repeated_filterd_dt_acc_allt.append(repeated_filtered_dt_acc)
            repeated_filter_acc_allt.append(repeated_filter_acc)
        
        
        orig_avg = np.mean(repeated_orig_dt_acc_allt)
        orig_std = np.std(repeated_orig_dt_acc_allt)
        
        ada_avg = np.mean(repeated_ada_score_allt)
        ada_std = np.std(repeated_ada_score_allt)
        
        filterd_avg = np.transpose(np.mean(repeated_filterd_dt_acc_allt,1))
        filterd_std = np.transpose(np.std(repeated_filterd_dt_acc_allt,1))
        
        filter_acc_avg = np.transpose(np.mean(repeated_filter_acc_allt,1))
        filter_acc_std = np.transpose(np.std(repeated_filter_acc_allt,1))
        
        stdoutOrigin=sys.stdout 
        sys.stdout = open( noise_type + " Noise Filtering" + ".txt", "a")
        print("Dataset:", name, "Noise Type:", noise_type, "Noise Level:", nl)
        print("Ada Acc",  to_str_rnd(ada_avg) + "/" + to_str_rnd(ada_std))
        print("Orig DT Acc",  to_str_rnd(orig_avg) + "/" + to_str_rnd(orig_std))
        print("Best DT Acc",  to_str_rnd(dt_best_acc_avg) + "/" + to_str_rnd(dt_best_acc_std))
        print("Post-Filter Performance")
        for f in range(len(filter_names)):
            print(filter_names[f],  ' '.join( [to_str_rnd(filterd_avg[f][t]) + "/" + to_str_rnd(filterd_std[f][t]) for t in range(len(threshold))]))
        print("Filter Acc")
        for f in range(len(filter_names)):
            print(filter_names[f],  ' '.join( [to_str_rnd(filter_acc_avg[f][t]) + "/" + to_str_rnd(filter_acc_std[f][t]) for t in range(len(threshold))]))
        print("")
        sys.stdout.close()
        sys.stdout=stdoutOrigin

        x_axis = [int(i*100) for i in threshold]
        x_axis[0:0] = [0]
        ax = axs[sub_plt_cnt]
              
        for filtering_method in range(len(filter_names)):
            f_name = filter_names[filtering_method]
            if f_name == "Best":
                continue
            baselines = ["R", "kDN", "Freq", "RF"]
            if f_name in baselines:
                l = '-.'
            else:
                l = '-'

            y_avg = list(filterd_avg[filtering_method])
            y_avg[0:0] = [orig_avg]
            y_std = list(filterd_std[filtering_method])
            y_std[0:0] = [0]
            
            #orig_avg = [orig_avg]
            #orig_std = [orig_std]
            
            y1 = np.tile(orig_avg,len(x_axis)) - np.array(orig_std)
            y2 = np.tile(orig_avg,len(x_axis)) + np.array(orig_std)
            ax.plot(x_axis, np.tile(orig_avg,len(x_axis)), '--', alpha = 0.1, c = "black")
            ax.fill_between(x_axis, y1, y2, color = "grey", alpha = 0.1)
            
            #y1 = np.tile(dt_best_acc_avg,len(x_axis)) - np.array(dt_best_acc_std)
            #y2 = np.tile(dt_best_acc_avg,len(x_axis)) + np.array(dt_best_acc_std)
            
            #ax.plot(x_axis, np.tile(dt_best_acc_avg,len(x_axis)), ':', alpha = 0.2, c = "black")
            #ax.fill_between(x_axis, y1, y2, color = "grey", alpha = 0.1)
            
            ax.errorbar(x_axis,y_avg, yerr=y_std, marker = 's', capsize=6, label = f_name, linestyle = l, alpha = 0.9)
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.set_title("Noise Level: " + str(nl), fontsize = 16)
        sub_plt_cnt += 1
            
    fig.suptitle("Dataset: " + name , fontsize = 18, y = 1.1)
    plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0, fontsize = 16)
    fig.text(0.5, -0.05, '% Instances Removed', ha='center', fontsize=18)
    fig.text(0.075, 0.5, 'Kappa', va='center', rotation='vertical', fontsize=18)
    plt.savefig('D:/UNi/Thesis/Filtering/Plots V2/' + name +'.png', bbox_inches="tight")
    plt.show()
        
        
        
    
        
        

        


    
        
    
        
            
            
            
        































