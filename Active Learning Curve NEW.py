from sklearn.tree import DecisionTreeClassifier
from uci_utils import *
from modAL.models import ActiveLearner
from get_difficulty_scores import get_osci_score
from get_difficulty_scores import get_clf_scores
from skmultiflow.data.random_rbf_generator import RandomRBFGenerator
from skmultiflow.data.random_tree_generator import RandomTreeGenerator
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from operator import itemgetter
from collections import Counter
import numpy as np
import os
from sklearn.metrics import auc
from sklearn.datasets import make_classification

from sklearn.model_selection import train_test_split
from modAL.expected_error import expected_error_reduction
from modAL.uncertainty import uncertainty_sampling
from sklearn.metrics import cohen_kappa_score
import sys 
import timeit


def get_initial_split(X,y, seed, initial_train_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1-initial_train_size), random_state=seed)
    return X_train, X_test, y_train, y_test

def split_data(X,y, seed):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=seed)
    return X_train, X_test, y_train, y_test

def get_sudo_label(X_train, X_test, y_train,global_rand_seed):
    model = DecisionTreeClassifier(random_state = global_rand_seed)
    model.fit(X_train,y_train)
    predicted_label = model.predict(X_test)
    return predicted_label

def get_inst(acc_lst, x_axis):
    final_acc = acc_lst[-1]
    inst = 0
    for i in range(len(acc_lst)):
        if acc_lst[i] >= final_acc:
            inst = x_axis[i]
            break
    inst = inst/x_axis[-1] 
    return inst
    

def lst_avg(lst):
    all_lst = [np.array(x) for x in lst]
    all_lst_avg = [round(np.mean(k),5) for k in zip(*all_lst)]
    all_lst_std = [round(np.std(k),5) for k in zip(*all_lst)]
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
    y = np.array(y)
    return X,y

def get_rt(samples, classes, features):
    stream = RandomTreeGenerator(tree_random_state=8873, sample_random_state=69, n_classes=classes,
                                 n_cat_features=0, n_num_features=features, n_categories_per_cat_feature=5, max_tree_depth=6,
                                 min_leaf_depth=3, fraction_leaves_per_level=0.15)
    data = stream.next_sample(samples)
    X = data[0]
    y = data[1]
    return X,y

def get_imb_rt(samples, classes, features):
    stream = RandomTreeGenerator(tree_random_state=8873, sample_random_state=69, n_classes=classes,
                                 n_cat_features=0, n_num_features=features, n_categories_per_cat_feature=5, max_tree_depth=7,
                                 min_leaf_depth=3, fraction_leaves_per_level=0.15)
    
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
    y = np.array(y)
    return X,y

def make_imb(X,y):
    label_cnt = Counter(y).most_common(1)
    most_common_label = label_cnt[0] 
    major_cnt = most_common_label[1]
    
    #75% as major class
    minor_cnt = int(major_cnt/3)
    
    imb_X = []
    imb_y = []
    
    major_idx = []
    remainder_idx = []
    for i in range(len(y)):
        if y[i] == most_common_label[0]:
            major_idx.append(i)
        else:
            remainder_idx.append(i)
            
    np.random.seed(0)
    minor_idx = np.random.choice(remainder_idx, minor_cnt, replace=False)
    
    for i in major_idx:
        imb_X.append(X[i])
        imb_y.append(y[i])
    
    for i in minor_idx:
        imb_X.append(X[i])
        imb_y.append(y[i])
    
    X = np.array(imb_X)
    y = np.array(imb_y)
    label_cnt = Counter(y)
    return X,y


def generate_difficult(X,y,size,classes,nth_neighbour):
    #print(X,y)
    nbrs = NearestNeighbors(n_neighbors= int(len(X)/(classes*6)), algorithm='kd_tree').fit(X)
    #nbrs = NearestNeighbors(n_neighbors= len(X), algorithm='kd_tree').fit(X)
    dist, indices = nbrs.kneighbors(X)
    merged = np.dstack((dist,indices))
    #print(dist)
    #print(indices)
    #print(merged)
    
    """
    [0] = index of nearest neighbour
    [1] = label of nearest neighbour
    [2] = distance 
    [3] = index of cur sample
    [4] = label of cur sample
    """
    
    neighbors_index = []
    for i in range(len(merged)):
        temp = []
        for j in merged[i]:
            j = list(j)
            temp.append([int(j[1]) ,y[int(j[1])] ,j[0] ,i ,y[i]])
        neighbors_index.append(temp)
    #print(neighbors_index)
    
    closest_other_neigh = []
    for i in neighbors_index:
        count = 0
        for j in range(1,len(i)):
            if i[j][1] != i[0][1]:
                count += 1
                if count == nth_neighbour:
                    closest_other_neigh.append(i[j])
                    break
    #print(closest_other_neigh)
    closest_other_neigh = sorted(closest_other_neigh, key = itemgetter(2))
    #print(closest_other_neigh)
    
    #print(len(closest_other_neigh))
    no_dup_neigh = []
    for i in range(len(closest_other_neigh)):
        temp = closest_other_neigh[i].copy()
        temp[0] = closest_other_neigh[i][3]
        temp[3] = closest_other_neigh[i][0]
        temp[1] = closest_other_neigh[i][4]
        temp[4] = closest_other_neigh[i][1]
        if temp not in no_dup_neigh:
            no_dup_neigh.append(closest_other_neigh[i])
            
    #print(len(no_dup_neigh))        
    diff_sample_cnt = int((size/2)*len(X))
    if diff_sample_cnt == 0:
        diff_sample_cnt = 1
    #print("Difficult Sample Count", diff_sample_cnt*2)
    for i in no_dup_neigh[:diff_sample_cnt]:
        near_index = i[0]
        near_label = i[1]
        cur_index = i[3]
        cur_label = i[4]
        y[cur_index] = near_label
        y[near_index] =cur_label
    
    #print(X,y)
    return X,y



def random_sampling(classifier, X_pool, global_rand_seed, n_instances):
    np.random.seed(global_rand_seed)
    n_samples = len(X_pool)
    query_idx = np.random.choice(range(n_samples),n_instances, replace=False)
    return query_idx


def difficulty_sampling(X_train, y_train, n_instances, diff_inst, X_pool, global_rand_seed, mode, n_estimators, dif_measure):
    
    #Random Inst Selection
    np.random.seed(global_rand_seed)
    n_samples = len(X_pool)
    rand_inst = n_instances - diff_inst
    if rand_inst != 0:
        #rand_ind = np.random.choice(range(n_samples),rand_inst, replace=False)
        #uncertainty sampling as base
        rand_ind = uncertainty_sampling(base_learner, X_pool, n_instances = rand_inst)
    else:
        rand_ind = []

    all_inst_ind = [i for i in range(len(X_pool))]
    possible_diff_inst = np.setdiff1d(all_inst_ind, rand_ind)
    
    consistency = 0
    osci_score, clf_cnt_score, osci_cnt, true_y_all_pred,ind_err_score, ind_osci_score, \
        ind_y_all_pred,oscillation_auc,osci_posi,fft_max,peak_pos, fft_sum, auc_sum, adatime = \
            get_osci_score(X_train, X_pool, y_train,consistency, global_rand_seed, n_estimators, mode)
            
    if dif_measure == "C cnt":
        score = clf_cnt_score
    elif dif_measure == "F":
        score = osci_score
    elif dif_measure == "F cnt":
        score = osci_cnt
    elif dif_measure == "F max":
        score = osci_posi
    elif dif_measure == "F peak":
        score = peak_pos
    elif dif_measure == "A max":
        score = fft_max
    elif dif_measure == "A sum":
        score = fft_sum
    elif dif_measure == "F ind":
        score = ind_osci_score
    else:
        score = ind_err_score
    
    #print(len(score), len(possible_diff_inst))
    possible_scores = [(j,score[j]) for j in possible_diff_inst]
    #print(possible_scores)
    sorted_possible_scores = sorted(possible_scores, key=lambda x: x[1])
    sorted_score_ind = [i[0] for i in sorted_possible_scores]
    #print(sorted_score_ind)
    
    if diff_inst!= 0:
        diff_ind = sorted_score_ind[-diff_inst:]
    else:
        diff_ind = []
    combined_ind = list(rand_ind) + list(diff_ind)
    #print(len( list(rand_ind)), len(list(diff_ind)))
    
    #print(combined_ind)
    if dif_measure == "C cnt":
        return combined_ind, adatime
    else:
        return combined_ind

def clf_sampling(X_train, y_train, n_instances, X_pool, global_rand_seed):
    clf_scores ,dt_score, mlp_score, knn_score, nb_score, rf_score = \
            get_clf_scores(X_train, X_pool, y_train, global_rand_seed)
        
    sorted_score_ind = np.argsort(clf_scores)
    diff_ind = sorted_score_ind[-n_instances:]
    return diff_ind

def get_alc(x,class_cnt, pool_cnt, metric):
    
    if metric == "kappa":
        amx = pool_cnt
        arand = 0
    else:
        amx = pool_cnt
        arand = pool_cnt/class_cnt
    
    alc = []
    for i in x:
        alc_i = (i-arand)/(amx-arand)
        alc.append(alc_i)
    return alc

def get_alc_lst(all_lst, class_cnt, pool_cnt, x_axis, metric):
    auc_lst = []
    for repeat in all_lst:
        a = auc(x_axis,repeat)
        auc_lst.append(a)
    norm_auc_lst = get_alc(auc_lst, class_cnt, pool_cnt, metric)
    std = round(np.std(norm_auc_lst),5)
    return std

def get_kappa(l, X, y):
    y_pred = l.predict(X)
    kappa = cohen_kappa_score(y, y_pred)
    return kappa


rand_auc_lst = []
uncer_auc_lst = []
clf_cnt_auc_lst = []
f_auc_lst = []
f_cnt_auc_lst = []
ind_err_auc_lst = []

rand_avg_lst = []
uncer_avg_lst = []
clf_cnt_avg_lst = []
f_avg_lst = []
f_cnt_avg_lst = []
ind_err_avg_lst = []

rand_std_lst = []
uncer_std_lst = []
clf_cnt_std_lst = []
f_std_lst = []
f_cnt_std_lst = []
ind_err_std_lst = []

for data in range(83,92):
#for data in range(1,2):
    metric = "kappa"
    if data == 1:
        X, y = UCI_Dataset_Loader.iris() #k = 50 d= 1
        name = "iris"
    elif data == 2:
        X, y = UCI_Dataset_Loader.ecoli()
        name = "ecoli"
    elif data == 7:
        X, y = UCI_Dataset_Loader.nursery()
        name = "nusery"
        seeds = [0,1,2,3,5,6,7,8,10,11] #for initial_train_size =  0.001 
    
    elif data == 8:
        X, y = UCI_Dataset_Loader.adult()
        name = "adult"
        seeds = [0,1,2,3,5,6,7,8,10,11] #for initial_train_size =  0.001 
    elif data == 9:
        X, y = UCI_Dataset_Loader.magic()
        name = "magic"
        seeds = [0,2,3,4,5,6,7,8,9,11] #for initial_train_size =  0.001 

    
    elif data == 70:
        X,y = get_rt(5000, 50, 10)
        name = "rt5"
        
    elif data == 71:
        X,y = get_rt(5000, 100, 10)
        name = "rt6"
    
    elif data == 72:
        X,y = get_rbf(5000, 50, 10)
        name = "rbf5"
        
    elif data == 73:
        X,y = get_rbf(5000, 100, 10)
        name = "rbf6"
       
    
    elif data == 80:
        X,y = get_imb_rt(5000, 50, 10)
        name = "rt5imb2"
        
    elif data == 81:
        X,y = get_imb_rt(5000, 100, 10)
        name = "rt6imb2"
    
    elif data == 82:
        X,y = get_imb_rbf(5000, 50, 10, 50)
        name = "rbf5imb2"
        
    elif data == 83:
        X,y = get_imb_rbf(5000, 100, 10, 50)
        name = "rbf6imb2"
    
    elif data ==84:
        X, y = UCI_Dataset_Loader.nursery()
        name = "nusery"
    
    elif data == 85:
        X, y = UCI_Dataset_Loader.nursery()
        X, y = make_imb(X, y)
        name = "nusery imb2"
    
    elif data == 86:
        X, y = UCI_Dataset_Loader.mushroom()
        name = "mushroom"
        
    elif data == 87:
        X, y = UCI_Dataset_Loader.thyroid()
        name = "thyroid"
        
    elif data == 88:
        X, y = UCI_Dataset_Loader.drybean()
        name = "drybean"
        
    elif data == 89:
        X, y = UCI_Dataset_Loader.drybean()
        X, y = make_imb(X, y)
        name = "drybean imb2"
    
    

    elif data == 91:
        X, y = UCI_Dataset_Loader.adult()
        name = "adult"
    elif data == 92:
        X, y = UCI_Dataset_Loader.car()
        name = "car"
        
    elif data == 93:
        X, y = UCI_Dataset_Loader.mammography()
        name = "mammography"
  
    elif data == 113:
        X, y = UCI_Dataset_Loader.codon()
        name = "codon"

        
        

    
    else:
        continue
    
    #exp_err_acc_lst = []
    rand_acc_lst = []
    uncer_acc_lst = []
    clf_cnt_acc_lst = []
    f_acc_lst = []
    f_cnt_acc_lst = []
    ind_err_acc_lst = []
    clf_acc_lst = []
    
    rand_inst_lst = []
    uncer_inst_lst = []
    clf_cnt_inst_lst = []
    f_inst_lst = []
    ind_err_inst_lst = []
    
    
    
    random_time_lst = []
    uncer_time_lst = []
    diff_time_lst = []
    ada_time_lst = []

    instances_sampled_base = []
    #Nursery and Adult Seed = [0,1,2,3,5,6,7,8,10,11] 0.001%
    #Magic Seed = [0,2,3,4,5,6,7,8,9,11] 0.001%
    #thyroid seed = [0,1,2,6,7,8,9,10,12,13] 0.01%
    
    batch_size = 0.01              #In Percentage .01 = 1%   
    diff_size = 1                  #In Percentage 1 = 100%
    initial_train_size = 0.01   #In Percentage 0.001 = .1%
    n_estimators = 300
    seeds = [0,1,2,3,4,5,6,7,8,9]
    mode = "consis"
    #mode = "prob"
    
    if name == "thyroid" and initial_train_size == 0.01:
        seeds = [0,1,2,6,7,8,9,10,12,13] 
    
    
    """
    if name != "MNIST":
        class_cnt = len(set(y))
    else:
        class_cnt = len(set(y_test))
    """
    class_cnt = len(set(y))
    
    
    for global_rand_seed in seeds:
        model = DecisionTreeClassifier(random_state = global_rand_seed)
        
        if name != "MNIST":
            #Split dataset into 50% train and 50% Test
            X_train , X_test, y_train, y_test = split_data(X,y, global_rand_seed)
        
        
        #Split Training set into Pool and initial Training Set
        initial_X_train, X_pool, initial_y_train, y_pool = get_initial_split(X_train,y_train,global_rand_seed, initial_train_size)
        
        
        clf_cnt_X_train = initial_X_train.copy()
        clf_cnt_y_train = initial_y_train.copy()
        
        f_X_train = initial_X_train.copy()
        f_y_train = initial_y_train.copy()
        
        f_cnt_X_train = initial_X_train.copy()
        f_cnt_y_train = initial_y_train.copy()
        
        ind_err_X_train = initial_X_train.copy()
        ind_err_y_train = initial_y_train.copy()
        
        
        
        # initializing the active learners
        learner = ActiveLearner(estimator = model,
            X_training = initial_X_train, y_training = initial_y_train)
        rand_learner = ActiveLearner(estimator = model,
            X_training = initial_X_train, y_training = initial_y_train)
        exp_err_learner = ActiveLearner(estimator = model,
            X_training = initial_X_train, y_training = initial_y_train)
        uncer_learner = ActiveLearner(estimator = model,
            X_training = initial_X_train, y_training = initial_y_train)
        clf_cnt_learner = ActiveLearner(estimator = model,
            X_training = initial_X_train, y_training = initial_y_train)
        f_learner = ActiveLearner(estimator = model,
            X_training = initial_X_train, y_training = initial_y_train)
        f_cnt_learner = ActiveLearner(estimator = model,
            X_training = initial_X_train, y_training = initial_y_train)
        ind_err_learner = ActiveLearner(estimator = model,
            X_training = initial_X_train, y_training = initial_y_train)
        #clf_learner = ActiveLearner(estimator = model,
           # X_training = initial_X_train, y_training = initial_y_train)
        
        
        # pool-based sampling
        #clf_X_pool = X_pool.copy()
        #clf_y_pool = y_pool.copy()

        exp_err_X_pool = X_pool.copy()
        exp_err_y_pool = y_pool.copy()
        
        rand_X_pool = X_pool.copy()
        rand_y_pool = y_pool.copy()
        
        uncer_X_pool = X_pool.copy()
        uncer_y_pool = y_pool.copy()
        
        clf_cnt_X_pool = X_pool.copy()
        clf_cnt_y_pool = y_pool.copy()
        
        f_X_pool = X_pool.copy()
        f_y_pool = y_pool.copy()
        
        f_cnt_X_pool = X_pool.copy()
        f_cnt_y_pool = y_pool.copy()
        
        ind_err_X_pool = X_pool.copy()
        ind_err_y_pool = y_pool.copy()
        
        
        if metric == "kappa":
            initial_score = get_kappa(learner, X_test, y_test)
        else:
            #Accuracy with no extra instances
            initial_score = learner.score(X_test, y_test)
        
        #clf_acc = [initial_score]
        exp_err_acc = [initial_score]
        rand_acc = [initial_score]
        uncer_acc = [initial_score]
        clf_cnt_acc = [initial_score]
        f_acc = [initial_score]
        f_cnt_acc = [initial_score]
        ind_err_acc = [initial_score]
        
        
        
        n = int(len(y_pool)*batch_size)
        diff_perc = int(n*diff_size)
        
        if n < 2:
            n = 2
            diff_perc = int(n*diff_size)
            
        if diff_perc != 0 & n == 1:
            diff_perc = 1
        if diff_perc < 1 & diff_perc != 0: 
            diff_perc = 1
        
        if global_rand_seed == 0:
            print("")
            print("Dataset:", name, "Batch Size", n, "Diff Size", diff_perc, "Initial Train", initial_train_size, "Mode", mode, "Base Learners", n_estimators)
            print(metric)
  
        print("Repeat:",global_rand_seed)
        
        x_axis = [0]
        j = -1
        for i in range(n, len(X_pool), n):
            j += 1
            x_axis.append(i)
            if global_rand_seed == 0:
                instances_sampled_base.append(n)
            
            """
            #Expected Error Sampling 
            exp_err_idx = expected_error_reduction(exp_err_learner, exp_err_X_pool, n_instances = n)
            exp_err_learner.teach(
                X = exp_err_X_pool[exp_err_idx],
                y = exp_err_y_pool[exp_err_idx]
            )
            # remove queried instance from pool
            exp_err_X_pool = np.delete(exp_err_X_pool, exp_err_idx, axis=0)
            exp_err_y_pool = np.delete(exp_err_y_pool, exp_err_idx)
            print('Accuracy after Expt Err query no. %d: %f' % (i+1, exp_err_learner.score(X_test, y_test)))
            exp_err_acc.append(exp_err_learner.score(X_test, y_test))
            """
            
            #Random Sampling 
            start = timeit.default_timer()
            rand_idx = random_sampling(rand_learner, rand_X_pool, global_rand_seed, n)
            stop = timeit.default_timer()
            random_time_lst.append(stop - start)
            
            rand_learner.teach(
                X = rand_X_pool[rand_idx],
                y = rand_y_pool[rand_idx]
            )
            # remove queried instance from pool
            rand_X_pool = np.delete(rand_X_pool, rand_idx, axis=0)
            rand_y_pool = np.delete(rand_y_pool, rand_idx)
            #print('Accuracy after Random query no. %d: %f' % (i+1, rand_learner.score(X_test, y_test)))
            if metric == "kappa":
                rand_acc.append(get_kappa(rand_learner, X_test, y_test))
            else:
                rand_acc.append(rand_learner.score(X_test, y_test))

            
            
            #Uncertainty Sampling 
            start = timeit.default_timer()
            uncer_idx = uncertainty_sampling(uncer_learner, uncer_X_pool, n_instances = n)
            stop = timeit.default_timer()
            uncer_time_lst.append(stop - start)
            
            uncer_learner.teach(
                X = uncer_X_pool[uncer_idx],
                y = uncer_y_pool[uncer_idx]
            )
            # remove queried instance from pool
            uncer_X_pool = np.delete(uncer_X_pool, uncer_idx, axis=0)
            uncer_y_pool = np.delete(uncer_y_pool, uncer_idx)
            #print('Accuracy after Uncertainty query no. %d: %f' % (i+1, uncer_learner.score(X_test, y_test)))
            if metric == "kappa":
                uncer_acc.append(get_kappa(uncer_learner, X_test, y_test))
            else:
                uncer_acc.append(uncer_learner.score(X_test, y_test))
            
            
            
            #Clf Cnt Sampling 
            start = timeit.default_timer()
            clf_cnt_indx, adatime = difficulty_sampling(clf_cnt_X_train, clf_cnt_y_train, n, diff_perc, clf_cnt_X_pool, 
                                            global_rand_seed, mode, n_estimators, dif_measure = "C cnt")
            stop = timeit.default_timer()
            diff_time_lst.append(stop - start)
            ada_time_lst.append(adatime)
            clf_cnt_learner.teach(
                X = clf_cnt_X_pool[clf_cnt_indx],
                y = clf_cnt_y_pool[clf_cnt_indx]
            )
            #add selected instance to ensemble traing set
            clf_cnt_X_train = np.append(clf_cnt_X_train, clf_cnt_X_pool[clf_cnt_indx], axis=0)
            clf_cnt_y_train = np.append(clf_cnt_y_train, clf_cnt_y_pool[clf_cnt_indx])
            # remove queried instance from pool
            clf_cnt_X_pool = np.delete(clf_cnt_X_pool, clf_cnt_indx, axis=0)
            clf_cnt_y_pool = np.delete(clf_cnt_y_pool, clf_cnt_indx)
            #print('Accuracy after clf_cnt query no. %d: %f' % (i+1, clf_cnt_learner.score(X_test, y_test)))
            if metric == "kappa":
                clf_cnt_acc.append(get_kappa(clf_cnt_learner, X_test, y_test))
            else:
                clf_cnt_acc.append(clf_cnt_learner.score(X_test, y_test))
            
            
            
            #f Sampling 
            f_indx = difficulty_sampling(f_X_train, f_y_train, n, diff_perc, f_X_pool, 
                                            global_rand_seed, mode, n_estimators, dif_measure = "F")
            f_learner.teach(
                X = f_X_pool[f_indx],
                y = f_y_pool[f_indx]
            )
            #add selected instance to ensemble traing set
            f_X_train = np.append(f_X_train, f_X_pool[f_indx], axis=0)
            f_y_train = np.append(f_y_train, f_y_pool[f_indx])
            # remove queried instance from pool
            f_X_pool = np.delete(f_X_pool, f_indx, axis=0)
            f_y_pool = np.delete(f_y_pool, f_indx)
            #print('Accuracy after f query no. %d: %f' % (i+1, f_learner.score(X_test, y_test)))
            if metric == "kappa":
                f_acc.append(get_kappa(f_learner, X_test, y_test))
            else:
                f_acc.append(f_learner.score(X_test, y_test))
                
            
            """
            #f cnt Sampling 
            f_cnt_indx = difficulty_sampling(f_cnt_X_train, f_cnt_y_train, n, diff_perc, f_cnt_X_pool, 
                                            global_rand_seed, mode, dif_measure = "F cnt")
            f_cnt_learner.teach(
                X = f_cnt_X_pool[f_cnt_indx],
                y = f_cnt_y_pool[f_cnt_indx]
            )
            #add selected instance to ensemble traing set
            f_cnt_X_train = np.append(f_cnt_X_train, f_cnt_X_pool[f_cnt_indx], axis=0)
            f_cnt_y_train = np.append(f_cnt_y_train, f_cnt_y_pool[f_cnt_indx])
            # remove queried instance from pool
            f_cnt_X_pool = np.delete(f_cnt_X_pool, f_cnt_indx, axis=0)
            f_cnt_y_pool = np.delete(f_cnt_y_pool, f_cnt_indx)
            #print('Accuracy after f_cnt query no. %d: %f' % (i+1, f_cnt_learner.score(X_test, y_test)))
            if metric == "kappa":
                f_cnt_acc.append(get_kappa(f_cnt_learner, X_test, y_test))
            else:
                f_cnt_acc.append(f_cnt_learner.score(X_test, y_test))
            """
            
            
            #Ind Err Sampling 
            ind_err_indx = difficulty_sampling(ind_err_X_train, ind_err_y_train, n, diff_perc, ind_err_X_pool, 
                                            global_rand_seed, mode, n_estimators, dif_measure = "Ind Err")
            ind_err_learner.teach(
                X = ind_err_X_pool[ind_err_indx],
                y = ind_err_y_pool[ind_err_indx]
            )
            #add selected instance to ensemble traing set
            ind_err_X_train = np.append(ind_err_X_train, ind_err_X_pool[ind_err_indx], axis=0)
            ind_err_y_train = np.append(ind_err_y_train, ind_err_y_pool[ind_err_indx])
            # remove queried instance from pool
            ind_err_X_pool = np.delete(ind_err_X_pool, ind_err_indx, axis=0)
            ind_err_y_pool = np.delete(ind_err_y_pool, ind_err_indx)
            #print('Accuracy after ind_err query no. %d: %f' % (i+1, ind_err_learner.score(X_test, y_test)))
            if metric == "kappa":
                ind_err_acc.append(get_kappa(ind_err_learner, X_test, y_test))
            else:
                ind_err_acc.append(ind_err_learner.score(X_test, y_test))
            
            
            i += n
            
        
        all_training_x = np.concatenate( (initial_X_train, X_pool), axis = 0 )
        all_training_y = np.concatenate((initial_y_train, y_pool), axis = 0 )
        learner.fit(all_training_x,all_training_y)
        
        
        if metric == "kappa":
            final_score = get_kappa(learner, X_test, y_test)
        else:
            #Accuracy with no extra instances
            final_score = learner.score(X_test, y_test)
        #print("Final Score:", final_score)
        
        
        
        
        rand_acc.append(final_score)
        uncer_acc.append(final_score)
        #exp_err_acc.append(final_score)
        clf_cnt_acc.append(final_score)
        f_acc.append(final_score)
        f_cnt_acc.append(final_score)
        ind_err_acc.append(final_score)
        #clf_acc.append(final_score)
        
        
        rand_acc_lst.append(rand_acc)
        uncer_acc_lst.append(uncer_acc)
        #exp_err_acc_lst.append(exp_err_acc)
        clf_cnt_acc_lst.append(clf_cnt_acc)
        f_acc_lst.append(f_acc)
        f_cnt_acc_lst.append(f_cnt_acc)
        ind_err_acc_lst.append(ind_err_acc)
        #clf_acc_lst.append(clf_acc)
        
        x_axis.append(len(y_pool))
        
        rand_inst = get_inst(rand_acc, x_axis)
        uncer_inst = get_inst(uncer_acc, x_axis)
        clf_cnt_inst = get_inst(clf_cnt_acc, x_axis)
        f_inst = get_inst(f_acc, x_axis)
        ind_err_inst = get_inst(ind_err_acc, x_axis)
        
        rand_inst_lst.append(rand_inst)
        uncer_inst_lst.append(uncer_inst)
        clf_cnt_inst_lst.append(clf_cnt_inst)
        f_inst_lst.append(f_inst)
        ind_err_inst_lst.append(ind_err_inst)

    
    
    #reap_exp_err_acc_lst_avg, reap_exp_err_acc_lst_std = lst_avg(exp_err_acc_lst)
    reap_uncer_acc_lst_avg, reap_uncer_acc_lst_std = lst_avg(uncer_acc_lst)
    reap_rand_acc_lst_avg, reap_rand_acc_lst_std = lst_avg(rand_acc_lst)
    reap_clf_cnt_acc_lst_avg, reap_clf_cnt_acc_lst_std = lst_avg(clf_cnt_acc_lst)
    reap_f_acc_lst_avg, reap_f_acc_lst_std = lst_avg(f_acc_lst)
    reap_f_cnt_acc_lst_avg, reap_f_cnt_acc_lst_std = lst_avg(f_cnt_acc_lst)
    reap_ind_err_acc_lst_avg, reap_ind_err_acc_lst_std = lst_avg(ind_err_acc_lst)
    #reap_clf_acc_lst_avg, reap_clf_acc_lst_std= lst_avg(clf_acc_lst)
    
    random_time_avg = np.mean(random_time_lst)
    uncer_time_avg = np.mean(uncer_time_lst)
    diff_time_avg = np.mean(diff_time_lst)
    ada_time_avg = np.mean(ada_time_lst)
    
    random_time_std = np.std(random_time_lst)
    uncer_time_std = np.std(uncer_time_lst)
    diff_time_std = np.std(diff_time_lst)
    ada_time_std = np.std(ada_time_lst)
    

    
    stdoutOrigin=sys.stdout 
    sys.stdout = open("time_log.txt", "a")
    print("")
    print("Dataset:", name, "Batch Size", n, "Diff Size", diff_perc, "Initial Train", initial_train_size, "Mode", mode, "Base Learners", n_estimators)
    print("")
    
    print("Random avg time per Sampling: ", random_time_avg, "std:", random_time_std)
    print("Uncerainty avg time per Sampling: ", uncer_time_avg, "std:", uncer_time_std)
    print("Difficulty avg time per Sampling: ", diff_time_avg, "std:", diff_time_std)
    print("Adaboost avg time per Sampling: ", ada_time_avg, "std:", ada_time_std)
    print("Number of samplings count:", len(diff_time_lst))
    
    sys.stdout.close()
    sys.stdout=stdoutOrigin
    
    measure_names = ["uncer", "rand", "clf_cnt", "f", "f_cnt", "ind_err"]

    stdoutOrigin=sys.stdout 
    sys.stdout = open("print_log.txt", "a")
    print("")
    print("Dataset:", name, "Batch Size", n, "Diff Size", diff_perc, "Initial Train", initial_train_size, "Mode", mode, "Base Learners", n_estimators)
    print("")
    print(measure_names[0] + "_all =", uncer_acc_lst)
    print(measure_names[0] + "_avg =", reap_uncer_acc_lst_avg)
    print(measure_names[0] + "_std =", reap_uncer_acc_lst_std)
    print("")
    print(measure_names[1] + "_all =", rand_acc_lst)
    print(measure_names[1] + "_avg =", reap_rand_acc_lst_avg)
    print(measure_names[1] + "_std =", reap_rand_acc_lst_std)
    print("")
    print(measure_names[2] + "_all =", clf_cnt_acc_lst)
    print(measure_names[2] + "_avg =", reap_clf_cnt_acc_lst_avg)
    print(measure_names[2] + "_std =", reap_clf_cnt_acc_lst_std)
    print("")
    print(measure_names[3] + "_all =", f_acc_lst)
    print(measure_names[3] + "_avg =", reap_f_acc_lst_avg)
    print(measure_names[3] + "_std =", reap_f_acc_lst_std)
    #print("")
    #print(measure_names[4] + "_all =", f_cnt_acc_lst)
    #print(measure_names[4] + "_avg =", reap_f_cnt_acc_lst_avg)
    #print(measure_names[4] + "_std =", reap_f_cnt_acc_lst_std)
    print("")
    print(measure_names[5] + "_all =", ind_err_acc_lst)
    print(measure_names[5] + "_avg =", reap_ind_err_acc_lst_avg)
    print(measure_names[5] + "_std =", reap_ind_err_acc_lst_std)
    sys.stdout.close()
    sys.stdout=stdoutOrigin
    
    rand_inst_avg = np.mean(rand_inst_lst)
    uncer_inst_avg = np.mean(uncer_inst_lst)
    clf_cnt_inst_avg = np.mean(clf_cnt_inst_lst)
    f_inst_avg = np.mean(f_inst_lst)
    ind_err_inst_avg = np.mean(ind_err_inst_lst)
    
    rand_inst_std = np.std(rand_inst_lst)
    uncer_inst_std = np.std(uncer_inst_lst)
    clf_cnt_inst_std = np.std(clf_cnt_inst_lst)
    f_inst_std = np.std(f_inst_lst)
    ind_err_inst_std = np.std(ind_err_inst_lst)
    
    print("random_INST =", rand_inst_avg, rand_inst_std)
    print("uncer_INST =", uncer_inst_avg, uncer_inst_std)
    print("clf_cnt_INST=", clf_cnt_inst_avg, clf_cnt_inst_std)
    print("f_INST =", f_inst_avg, f_inst_std)
    print("ind_err_INST =", ind_err_inst_avg, ind_err_inst_std)
    
    stdoutOrigin=sys.stdout 
    sys.stdout = open("instance_results.txt", "a")
    print("Dataset:", name, "Batch Size", n, "Diff Size", diff_perc, "Initial Train", initial_train_size, "Mode", mode, "Base Learners", n_estimators)
    print("random_INST =", rand_inst_avg, rand_inst_std)
    print("uncer_INST =", uncer_inst_avg, uncer_inst_std)
    print("clf_cnt_INST=", clf_cnt_inst_avg, clf_cnt_inst_std)
    print("f_INST =", f_inst_avg, f_inst_std)
    print("ind_err_INST =", ind_err_inst_avg, ind_err_inst_std)
    print("")
    sys.stdout.close()
    sys.stdout=stdoutOrigin
    
    
    pool_cnt = len(y_pool)
    #alc std
    rand_std = get_alc_lst(rand_acc_lst, class_cnt, pool_cnt, x_axis, metric)
    uncer_std = get_alc_lst(uncer_acc_lst, class_cnt, pool_cnt, x_axis, metric)
    clf_cnt_std = get_alc_lst(clf_cnt_acc_lst, class_cnt, pool_cnt, x_axis, metric)
    f_std = get_alc_lst(f_acc_lst, class_cnt, pool_cnt, x_axis, metric)
    #f_cnt_std = get_alc_lst(f_cnt_acc_lst, class_cnt, pool_cnt, x_axis, metric)
    ind_err_std = get_alc_lst(ind_err_acc_lst, class_cnt, pool_cnt, x_axis, metric)
    
    
    rand_avg_lst.append(reap_rand_acc_lst_avg)
    uncer_avg_lst.append(reap_uncer_acc_lst_avg)
    clf_cnt_avg_lst.append(reap_clf_cnt_acc_lst_avg)
    f_avg_lst.append(reap_f_acc_lst_avg)
    #f_cnt_avg_lst.append(reap_f_cnt_acc_lst_avg)
    ind_err_avg_lst.append(reap_ind_err_acc_lst_avg)
    
    rand_std_lst.append(reap_rand_acc_lst_std)
    uncer_std_lst.append(reap_uncer_acc_lst_std)
    clf_cnt_std_lst.append(reap_clf_cnt_acc_lst_std)
    f_std_lst.append(reap_f_acc_lst_std)
    f_cnt_std_lst.append(reap_f_cnt_acc_lst_std)
    ind_err_std_lst.append(reap_ind_err_acc_lst_std)
    
    
    plt.plot(x_axis, reap_rand_acc_lst_avg, label = "Random")
    plt.plot(x_axis, reap_uncer_acc_lst_avg, label = "Uncertainty")
    plt.plot(x_axis, reap_clf_cnt_acc_lst_avg, label = "Clf cnt")
    plt.plot(x_axis, reap_f_acc_lst_avg, label = "F")
    #plt.plot(x_axis, reap_f_cnt_acc_lst_avg, label = "F cnt")
    plt.plot(x_axis, reap_ind_err_acc_lst_avg, label = "Ind Err")
    
    plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
    plt.title(name + " " + str(n) + " d" + str(diff_perc) + " t" + str(initial_train_size))
    plt.xlabel('Instance Queried')
    plt.ylabel('Accuracy')
    plt.savefig('D:/UNi/Thesis/Acitve Learning new/Learning Curve Plots/' + name + " " + str(n) + " d" + str(diff_perc) + " t" + str(initial_train_size) + " " + mode + " " +  str(n_estimators) + '.png', bbox_inches="tight")
    plt.show()
    
    
    
    rand_auc = auc(x_axis,reap_rand_acc_lst_avg)
    uncer_auc = auc(x_axis,reap_uncer_acc_lst_avg)
    clf_cnt_auc = auc(x_axis, reap_clf_cnt_acc_lst_avg)
    f_auc = auc(x_axis, reap_f_acc_lst_avg)
    #f_cnt_auc = auc(x_axis, reap_f_cnt_acc_lst_avg)
    ind_err_auc = auc(x_axis, reap_ind_err_acc_lst_avg)
    
    auc_lst = [rand_auc,uncer_auc,clf_cnt_auc,f_auc,ind_err_auc]
    alc_lst = get_alc(auc_lst,class_cnt, pool_cnt, metric)
    
    print("")
    print("random_ALC =", alc_lst[0], rand_std)
    #print("Exp Err ALC =", reap_exp_err_ALC_lst)
    print("uncer_ALC =", alc_lst[1], uncer_std)
    print("clf_cnt_ALC =", alc_lst[2], clf_cnt_std)
    print("f_ALC =", alc_lst[3], f_std)
    #print("f_cnt_ALC =", alc_lst[4], f_cnt_std)
    print("ind_err_ALC =", alc_lst[4], ind_err_std)
    #print("Avg Clf ALC =", reap_clf_ALC_lst)
    
    
    rand_auc_lst.append(alc_lst[0])
    uncer_auc_lst.append(alc_lst[1])
    clf_cnt_auc_lst.append(alc_lst[2])
    f_auc_lst.append(alc_lst[3])
    #f_cnt_auc_lst.append(alc_lst[4])
    ind_err_auc_lst.append(alc_lst[4])
    
    stdoutOrigin=sys.stdout 
    sys.stdout = open("alc_results.txt", "a")
    print("Dataset:", name, "Batch Size", n, "Diff Size", diff_perc, "Initial Train", initial_train_size, "Mode", mode, "Base Learners", n_estimators)
    print("random_ALC =", alc_lst[0], rand_std)
    #print("Exp Err ALC =", reap_exp_err_ALC_lst)
    print("uncer_ALC =", alc_lst[1], uncer_std)
    print("clf_cnt ALC =", alc_lst[2], clf_cnt_std)
    print("f_ALC =", alc_lst[3], f_std)
    #print("f_cnt_ALC =", alc_lst[4], f_cnt_std)
    print("ind_err_ALC =", alc_lst[4], ind_err_std)
    print("")
    sys.stdout.close()
    sys.stdout=stdoutOrigin
    


diff_size = str(int(diff_size*100))

"""
print("")
print("random_"+ diff_size +"_avg = ", rand_avg_lst)
print("random_"+ diff_size +"_std = ", rand_std_lst)
print("")   
print("uncertainty_"+ diff_size +"_avg = ", uncer_avg_lst)
print("uncertainty_"+ diff_size +"_std = ", uncer_std_lst)
print("")   
print("clf_Cnt_"+ diff_size +"_avg = ", clf_cnt_avg_lst)
print("clf_Cnt_"+ diff_size +"_std = ", clf_cnt_std_lst)
print("")   
print("f_"+ diff_size +"_avg = ", f_avg_lst)
print("f_"+ diff_size +"_std = ", f_std_lst)
print("")   
print("f_cnt_"+ diff_size +"_avg = ", f_cnt_avg_lst)
print("f_cnt_"+ diff_size +"_std = ", f_cnt_std_lst)
print("")   
print("ind_err_"+ diff_size +"_avg = ", ind_err_avg_lst)
print("ind_err_"+ diff_size +"_std = ", ind_err_std_lst)
print("")  
"""

print("")
print("random_"+ diff_size +"_ALC = ", rand_auc_lst)
print("uncertainty_"+ diff_size +"_ALC = ", uncer_auc_lst)
print("clf_Cnt_"+ diff_size +"_ALC = ", clf_cnt_auc_lst)
print("f_"+ diff_size +"_ALC = ", f_auc_lst)
#print("f_cnt_"+ diff_size +"_ALC = ", f_cnt_auc_lst)
print("ind_err_"+ diff_size +"_ALC = ", ind_err_auc_lst)
print("")   


stdoutOrigin=sys.stdout 
sys.stdout = open("avg std alc " + mode +".txt", "w")
print("")
print("random_"+ diff_size +"_avg = ", rand_avg_lst)
print("random_"+ diff_size +"_std = ", rand_std_lst)
print("")   
print("uncertainty_"+ diff_size +"_avg = ", uncer_avg_lst)
print("uncertainty_"+ diff_size +"_std = ", uncer_std_lst)
print("")   
print("clf_Cnt_"+ diff_size +"_avg = ", clf_cnt_avg_lst)
print("clf_Cnt_"+ diff_size +"_std = ", clf_cnt_std_lst)
print("")   
print("f_"+ diff_size +"_avg = ", f_avg_lst)
print("f_"+ diff_size +"_std = ", f_std_lst)
#print("")   
#print("f_cnt_"+ diff_size +"_avg = ", f_cnt_avg_lst)
#print("f_cnt_"+ diff_size +"_std = ", f_cnt_std_lst)
print("")   
print("ind_err_"+ diff_size +"_avg = ", ind_err_avg_lst)
print("ind_err_"+ diff_size +"_std = ", ind_err_std_lst)
print("")   



print("")
print("random_"+ diff_size +"_ALC = ", rand_auc_lst)
print("uncertainty_"+ diff_size +"_ALC = ", uncer_auc_lst)
print("clf_Cnt_"+ diff_size +"_ALC = ", clf_cnt_auc_lst)
print("f_"+ diff_size +"_ALC = ", f_auc_lst)
#print("f_cnt_"+ diff_size +"_ALC = ", f_cnt_auc_lst)
print("ind_err_"+ diff_size +"_ALC = ", ind_err_auc_lst)
print("")   
sys.stdout.close()
sys.stdout=stdoutOrigin




























