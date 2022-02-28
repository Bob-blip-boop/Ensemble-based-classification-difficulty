from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from uci_utils import *


def nearest_enemy (X: np.ndarray, y: np.ndarray, cls_index: np.ndarray, 
                   i: int, metric: str = "euclidean", n_neighbors=1) :
    " This function computes the distance from a point x_i to their nearest enemy"
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(X)
    X = scaler.transform(X)
    
    X_ = X[np.logical_not(cls_index[y[i]])]
    y_ = y[np.logical_not(cls_index[y[i]])]
    
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
    neigh.fit(X_, y_) 
    dist_enemy, pos_enemy = neigh.kneighbors([X[i, :]])
    dist_enemy = np.reshape(dist_enemy, (n_neighbors,))
    pos_enemy_ = np.reshape(pos_enemy, (n_neighbors,))
    query = X_[pos_enemy_, :]
    
    pos_enemy = np.where(np.all(X==query,axis=1))
    #print(pos_enemy[0])
    #print(pos_enemy.shape)

    #pos_enemy = np.reshape(pos_enemy, (n_neighbors,))

    return dist_enemy, pos_enemy

def nearest_neighboor_same_class (X: np.ndarray, y: np.ndarray, cls_index: np.ndarray,
                                  i: int, metric: str = "euclidean", n_neighbors=1) :
    " This function computes the distance from a point x_i to their nearest neighboor from its own class"
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(X)
    X = scaler.transform(X)
    
    query = X[i, :]
    label_query = y[i]
    #print(label_query)
    X_ = X[cls_index[label_query]]
    y_ = y[cls_index[label_query]]
    
    pos_query = np.where(np.all(X_==query,axis=1))
    X_ = np.delete(X_, pos_query, axis = 0)
    y_ = np.delete(y_, pos_query, axis = 0) 
    
    if len(X_) < 2:
        return 0, 0

    
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
    #print(X_.shape, y_.shape)
    #print(query)
    neigh.fit(X_, y_) 
    dist_neigh, pos_neigh = neigh.kneighbors([X[i, :]])
    dist_neigh = np.reshape(dist_neigh, (n_neighbors,))
    pos_neigh = np.reshape(pos_neigh, (n_neighbors,))
    return dist_neigh, pos_neigh

def intra_extra(X: np.ndarray, y: np.ndarray, cls_index: np.ndarray):
    intra = np.sum([nearest_neighboor_same_class (X, y, cls_index, i)[0] for i in range(np.shape(X)[0])])
    extra = np.sum([nearest_enemy (X, y, cls_index, i)[0] for i in range(np.shape(X)[0])])
    return intra/extra

def get_dNN(X,y):
    classes, class_freqs = np.unique(y, return_counts=True)
    cls_index = [np.equal(y, i) for i in range(classes.shape[0])]
    
    
    knn = []
    for i in range(len(y)):
        intra = nearest_neighboor_same_class (X, y, cls_index, i)[0]
        extra = nearest_enemy (X, y, cls_index, i)[0]
        knn.append((intra/extra)[0])
    return knn


"""
X, y = UCI_Dataset_Loader.iris()
dNN = get_dNN(X,y)
print(dNN)
"""





