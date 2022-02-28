"""
Diffrences from the previous version: 
    1. Made More Efficient
"""
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
import statistics
import timeit


def fit_ensemble(X_train, X_test, y_train, y_test, n_estima,tree_dep,global_rand_seed):
    
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=tree_dep), n_estimators=n_estima, algorithm = 'SAMME',random_state=global_rand_seed)
    
    #start = timeit.default_timer()
    clf.fit(X_train, y_train)
    #stop = timeit.default_timer()
    
    y_all_pred = clf.staged_predict(X_test)
    all_score = clf.staged_score(X_test, y_test)
    all_estimator = clf.estimators_
    estimator_weights = clf.estimator_weights_
    estimator_errors = clf.estimator_errors_
    
    #y_pred = clf.predict(X_test)
    #accuracy = accuracy_score(y_test, y_pred)
    #print("Ensemble",accuracy, "Time:", stop - start)
    return y_all_pred,all_score,all_estimator,estimator_weights, estimator_errors

"""
def get_oscilation_score(oscillation_position,n, size):
    positions = oscillation_position
    y = []
    count = 0
    check = None
    for i in range(positions[0],n):
        if positions[count] == i:
            if count != len(positions)-1:
                count += 1
            if check == None:
                check = 0
            elif check == 0:
                check = 1
            elif check == 1:
                check = 0
        y.append(check)
    full_average = np.convolve(y, np.ones(size), 'valid') / size
    #full_average = np.convolve(y, np.ones(size), 'same') / size
    #full_average = np.convolve(y, np.ones(size), 'full') / size
    #plt.plot(full_average)
    #plt.show()
    return full_average
"""

def rolling_avg(x, N):
    start = []
    end = []
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    #print(cumsum)
    for i in range(1, N):
        temp_sum = cumsum[i]/i
        start.append(temp_sum)
    #print(start)
    #max_window = list((cumsum[N:] - cumsum[:-N]) / float(N))
    max_window = list(np.convolve(x, np.ones(N)/N, mode='valid'))
    #print(max_window)
    #if N %2 ==0:
    #    max_window = max_window[1:-1]

    
    end_window = x[-N+1:]
    end_window.reverse()
    end_cumsum = np.cumsum(np.insert(end_window, 0, 0))
    #print(end_window)
    #print(end_cumsum)
    for i in range(1,N):
        temp_sum = end_cumsum[i]/i
        end.append(temp_sum)
    end.reverse()
    #print(end)
    #mid = int(N/2)
    combined = start+max_window+end
    #combined = start[mid:]+max_window+end[:-mid]
    #print(combined)
    return combined

"""
def ApEn(U, m, r) -> float:

    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [
            len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0)
            for x_i in x
        ]
        return (N - m + 1.0) ** (-1) * sum(np.log(C))

    N = len(U)

    return abs(_phi(m + 1) - _phi(m))

"""

def get_base_clf(y_all_pred, true_label, consistency, window_size):
    clf_count={}
    oscillation_count = {}
    oscillation_auc = {}
    label_fluc = {}

    consistency =  int(consistency * len(y_all_pred))
    if consistency == 0:
        consistency = 1
    

    for x in range(len(true_label)):
        clf_count[x] = None
        oscillation_count[x] = None
        oscillation_auc[x] = None
        label_fluc[x] = None

    y_all_pred = np.transpose((np.array(y_all_pred)))
    for i in range(len(y_all_pred)):
        right_label = true_label[i]
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
    
    return clf_count, oscillation_count, oscillation_auc, label_fluc

