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
        
    dt_score = [i * -1 for i in dt_score]
    mlp_score = [i * -1 for i in mlp_score]
    knn_score = [i * -1 for i in knn_score]
    nb_score = [i * -1 for i in nb_score]
    rf_score = [i * -1 for i in rf_score] 
    
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

def remove_samples(score,threshold,X_train,y_train):
    #score = normalize(score)
    instances_to_remove = int(threshold*len(y_train))
    #print(score)
    
    #sort instances based on their scores
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

def add_dif_noise(score,threshold,X_train,y_train, window, noise_type):
    
    random.seed(0)
    class_cnt = len(Counter(y_train))
    instances_difficult = int(threshold*len(y_train))
    #print(instances_difficult)
    
    #normalize scores against maximum (between 0-1)
    score = normalize(score)
    abs_score = [abs(i-window/100) for i in score]
    
    #sort instances based on their abs_score, scores closer to window will be smaller
    sorted_score_idx = np.argsort(abs_score)
    #print(sorted_score_idx)
    
    #change the top k difficult instances's labels
    instances_kept = sorted_score_idx[instances_difficult:]
    instances_made_dif = sorted_score_idx[:instances_difficult]
    #print(instances_kept)
    #print(instances_made_dif)
    
    
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
    for train_index, test_index in skf.split(X, y):
        x_train_fold, x_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        
        temp_f, temp_c, temp_ie, temp_adatime = get_osci_score_small(x_train_fold, x_test_fold, y_train_fold, y_test_fold, n_estimators, seed)
        f.extend(temp_f)
        c.extend(temp_c)
        ie.extend(temp_ie)
        adatime.append(temp_adatime)
    return f, c, ie, adatime
    
        

dataset_names = []
all_samples_removed = []

for data in range(6,7):
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
    
    allT_dt_difference_avg = []
    allT_dt_difference_std = []


    
    #noise generation methods
    measure_names = ["rand", "dNN" ,"clf_cnt", "f", "ind_err"]
    
    #filtering methods
    filter_names = ["kdn", "inst_hard", "RF"]
    
    #seeds = [0,1,2,3,4,5,6,7,8,9]
    seeds = [0]
    n_estimators = 300
    fold = 5
    k = 5
    #threshold= [0.05, 0.10, 0.15, 0.20, 0.25]
    threshold= [0.05, 0.10, 0.20, 0.40]
    #threshold = [0.05]
    t = 0.4
    print("")
    print("Dataset:", name)
    interval_size = .1
    #noise_type = "feature"
    noise_type = "label"
    
    fig, axs = plt.subplots(len(seeds),5, figsize=(12, 2), sharex=True, sharey=True)
    #fig.subplots_adjust(top=0.75)
    #fig.tight_layout()
    #fig.subplots_adjust(top=0.95)
    axs = axs.ravel()
    sub_plt_cnt = 0
    frame1 = plt.gca()
    frame1.axes.yaxis.set_ticklabels([])
        
    
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
        difficulty_scores = [normalize(score) for score in difficulty_scores]
        difficulty_names = ["R", "dNN", "C", "F", "IE"]
        colors = plt.rcParams["axes.prop_cycle"]()
        
        DF_var = pd.DataFrame.from_dict({"R":difficulty_scores[0],"dNN":difficulty_scores[1],"C":difficulty_scores[2],"F":difficulty_scores[3], "IE": difficulty_scores[4]})



        for i in range(len(difficulty_scores)):
            score = difficulty_scores[i]
            score = normalize(score)
            d_name = difficulty_names[i]
            ax = axs[sub_plt_cnt]
            c = next(colors)["color"]
            #sns.kdeplot(score, ax = axs[sub_plt_cnt], color=c, cut=0)
            hist_graph = ax.hist(score, alpha=0.5, bins=10, density=True, color=c)
            ax.set_title(d_name, fontsize = 18)
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.set_xticks([0.0,0.5,1.0])

            sub_plt_cnt += 1
    fig.suptitle("Dataset: " + name, fontsize = 20, y = 1.15)  
    plt.show()
    fig.savefig('D:/UNi/Thesis/noise generation/plots/' + name +'.png', bbox_inches="tight")
        

        
 
            
            
               
        
    
        
        
        
    































