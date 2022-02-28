from semi_adaboost_difficulty import get_most_consis_scores
from semi_adaboost_difficulty import get_most_prob_scores
from semi_adaboost_difficulty import fit_ensemble

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import MinMaxScaler
from itertools import groupby

import numpy as np
from scipy.signal import find_peaks
from scipy.fft import rfft
import timeit

def decisiontree(X_train, X_test, y_train,global_rand_seed):
    model = DecisionTreeClassifier(max_depth = 10, random_state = global_rand_seed)
    
    model.fit(X_train,y_train)
    predicted_label = model.predict(X_test)

    y_pred = model.predict_proba(X_test)
    probability_estimates = []
    for i in y_pred:
        probability_estimates.append(max(i))
    #total_nodes = model.get_n_leaves()
    #sample_depth = [i/total_nodes for i in leaf_index]
    #accuracy = accuracy_score(y_test, predicted_label)
   # print("dt:",accuracy, "Time:",time)

    return predicted_label, probability_estimates

@ignore_warnings(category=ConvergenceWarning)
def mlp(X_train, X_test, y_train,global_rand_seed):
    #model = MLPClassifier(max_iter=500,solver='sgd', momentum = 0.2 )
    model = MLPClassifier(max_iter=200, random_state = global_rand_seed)
    
    model.fit(X_train,y_train)
    predicted_label = model.predict(X_test)

    probability_estimates = []
    y_pred = model.predict_proba(X_test)
    for i in y_pred:
        probability_estimates.append(max(i))
    #accuracy = accuracy_score(y_test, predicted_label)
    #print("mlp:",accuracy, "Time:",time)
    return predicted_label, probability_estimates

def KNN(X_train, X_test, y_train):
    model = KNeighborsClassifier(n_neighbors=5)

    model.fit(X_train,y_train)
    y_pred = model.predict_proba(X_test)

    probability_estimates = []
    for i in y_pred:
        probability_estimates.append(max(i))
    predicted_label = model.predict(X_test)
    #accuracy = accuracy_score(y_test, predicted_label)
    #print("KNN:",accuracy, "Time:",time)
    return predicted_label, probability_estimates

def NB(X_train, X_test, y_train):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    model = MultinomialNB()
    

    model.fit(X_train,y_train)
    y_pred = model.predict_proba(X_test)

    
    probability_estimates = []
    for i in y_pred:
        probability_estimates.append(max(i))
    predicted_label = model.predict(X_test)
    #accuracy = accuracy_score(y_test, predicted_label)
    #print("NB:",accuracy, "Time:",time)
    return predicted_label, probability_estimates

def RF(X_train, X_test, y_train,global_rand_seed):
    model = RandomForestClassifier(random_state = global_rand_seed)
    
    model.fit(X_train,y_train)
    y_pred = model.predict_proba(X_test)

    probability_estimates = []
    for i in y_pred:
        probability_estimates.append(max(i))
    predicted_label = model.predict(X_test)
    #accuracy = accuracy_score(y_test, predicted_label)
    #print("RF:",accuracy, "Time:",time)
    return predicted_label, probability_estimates

def get_osc_cnt(ind_y_all_pred,true_lbl):
    err_count = {}
    osc_count = {}
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

    return err_count,osc_count

def get_osci_score(X_train, X_test, y_train,consistency, global_rand_seed, n_estimators, mode):
    
    osci_score_list = []
    clf_cnt_score_list = []
    osci_cnt = []
    osci_posi = []
    fft_max = []
    fft_sum = []
    peak_pos = []
    auc_sum = []
    
    tree_dep = 1
    #n_estimators = 300
    
    window_size = n_estimators
    
    start = timeit.default_timer()
    y_all_pred,all_estimator = fit_ensemble(X_train, X_test, y_train, n_estimators,tree_dep,global_rand_seed)
    stop = timeit.default_timer()
    adatime = stop - start
    
    if mode == "consis": #Sudo_true = most_consis_label_lst
        clf_count, fluc_degree, oscillation_count, oscillation_auc, sudo_true_label, label_fluc = get_most_consis_scores(y_all_pred, consistency, window_size)
    else:                #Sudo_true = most_prob_label
        clf_count, oscillation_count, oscillation_auc, sudo_true_label, label_fluc = get_most_prob_scores(y_all_pred, consistency, window_size)
    
    for i in range(len(X_test)):
        
        #AUC (Iterations with incorect label)
        auc_sum.append(sum(oscillation_auc[i]))
        
        #Fluc Cnt
        osci_cnt.append(oscillation_count[i]/n_estimators)
        
        #Fluc Score
        osci_score = (0.5-abs((sum(oscillation_auc[i])/len(oscillation_auc[i]))-0.5))*2
        osci_score_list.append(osci_score)
        
        #Fluc Maxima
        consis_cnt = [(k, sum(1 for i in g)) for k,g in groupby(label_fluc[i])]
        if consis_cnt[0][1] == n_estimators and consis_cnt[0][0] == 0:
            osci_posi.append(0)
        else:
            sorted_consis_cnt = sorted(consis_cnt, key=lambda x: x[1])
            most_consis_label = (None,None)
            n = -1
            while most_consis_label[0] != 1:
                most_consis_label = sorted_consis_cnt[n]
                n -= 1
    
            max_idx = 0
            for j in consis_cnt:
                if j[0]!= 1:
                    max_idx += j[1]
                else:
                    if j[0] == 1 and j[1] == most_consis_label[1]:
                        max_idx += j[1]/2
                        break
                    else:
                        max_idx += j[1]
            osci_posi.append(max_idx)
        
        #Fluc Peaks
        auc_osci = list(oscillation_auc[i])
        peaks, _ = find_peaks(auc_osci, height=0)
        peaks = [i/len(auc_osci) for i in peaks]
        mag = np.linalg.norm(peaks)
        peak_pos.append(mag)
        
        #Max Amp
        yf = rfft(auc_osci)
        yf = np.abs(yf)
        max_amp = max(yf[1:])
        fft_max.append(max_amp)
        
        #Amp Sum
        amp_sum = sum(yf[1:])
        fft_sum.append(amp_sum)
        
        #Clf Cnt
        if clf_count[i] != None:
            base_clf = clf_count[i]/n_estimators
            clf_cnt_score_list.append(base_clf)
        else:
            clf_cnt_score_list.append(1)
    
    ind_y_all_pred = []
    for estimator in all_estimator:
        ypred = estimator.predict(X_test)
        ind_y_all_pred.append(list(ypred))
    
    
    ind_y_all_pred = np.transpose((np.array(ind_y_all_pred)))
    err_count, osci_count = get_osc_cnt(ind_y_all_pred,sudo_true_label)
    
    #Ind Error
    err_count_lst = []
    for i in err_count.values():
        i = i/n_estimators
        err_count_lst.append(i)
    
    #Ind Fluc 
    osci_count_lst = []
    for i in osci_count.values():
        i = i/n_estimators
        osci_count_lst.append(i)      
    
    return osci_score_list, clf_cnt_score_list, osci_cnt, y_all_pred, err_count_lst, osci_count_lst, \
        ind_y_all_pred, oscillation_auc, osci_posi, fft_max, peak_pos, fft_sum, auc_sum, adatime

def get_clf_scores(X_train, X_test, y_train, global_rand_seed):      
    
    dt_predicted_label, dt_score = decisiontree(X_train, X_test, y_train,global_rand_seed)
    
    mlp_predicted_label, mlp_score = mlp(X_train, X_test, y_train,global_rand_seed)
            
    knn_predicted_label, knn_score = KNN(X_train, X_test, y_train)
            
    nb_predicted_label, nb_score = NB(X_train, X_test, y_train)
            
    rf_predicted_label, rf_score = RF(X_train, X_test, y_train,global_rand_seed)
    
    """
    frequency = []
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
    """
    dt_score = [i * -1 for i in dt_score]
    mlp_score = [i * -1 for i in mlp_score]
    knn_score = [i * -1 for i in knn_score]
    nb_score = [i * -1 for i in nb_score]
    rf_score = [i * -1 for i in rf_score] 
    
    clf_scores = [dt_score, mlp_score, knn_score, nb_score, rf_score]
    clf_scores = [np.array(x) for x in clf_scores]
    clf_scores = [np.mean(k) for k in zip(*clf_scores)]
    
    return clf_scores ,dt_score, mlp_score, knn_score, nb_score, rf_score





