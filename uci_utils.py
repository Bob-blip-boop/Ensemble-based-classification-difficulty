import pandas as pd, numpy as np
import warnings
from IPython.display import Markdown, display
from sklearn.preprocessing import LabelEncoder

class UCI_Dataset_Loader():
    
    @classmethod
    def mushroom(cls):
        url = "D:/UNi/Thesis/Active Learning/dataset/agaricus-lepiota.data"
        data = pd.read_csv(url, header=0)
        features = data.iloc[:,1:]
        features = pd.get_dummies(features)
        features = features.to_numpy()
        labels = data.iloc[:,0]
        labels = labels.astype('category').cat.codes
        labels = labels.to_numpy()
        return features, labels
    
    @classmethod
    def codon(cls):
        url = "D:/UNi/Thesis/Active Learning/dataset/codon_usage.csv"
        data = pd.read_csv(url, header=0)
        labelencoder = LabelEncoder()
        data = data.apply(labelencoder.fit_transform)

        features = data.iloc[:,1:]
        #features = pd.get_dummies(features)
        features = features.to_numpy()
        labels = data.iloc[:,0]
        #labels = labels.astype('category').cat.codes
        labels = labels.to_numpy()

        return features, labels
    
    
    @classmethod
    def thyroid(cls):
        url = "D:/UNi/Thesis/Active Learning/dataset/thyroid-disease.data"
        data = pd.read_csv(url, header=None, sep = ' ')
        features = data.iloc[:,:-3]
        features = pd.get_dummies(features)
        features = features.to_numpy()
        labels = data.iloc[:,-3]
        labels = labels.astype('category').cat.codes
        labels = labels.to_numpy()
        return features, labels
    
    @classmethod
    def drybean(cls):
        url = "D:/UNi/Thesis/Active Learning/dataset/DryBeanDataset/Dry_Bean_Dataset.arff"
    
        data=pd.read_csv(url, skiprows=25, header=None, sep = ',')
        features = data.iloc[:,:-1]
        features = pd.get_dummies(features)
        features = features.to_numpy()
        labels = data.iloc[:,-1]
        labels = labels.astype('category').cat.codes
        labels = labels.to_numpy()
        return features, labels
    
    
    @classmethod
    def mnist_train(cls):
        url = "D:/UNi/Thesis/Active Learning/dataset/mnist_train.csv"
        data = pd.read_csv(url, header=None)
        features = data.iloc[:,1:]
        features = features.to_numpy()
        labels = data.iloc[:,0]
        labels = labels.astype('category').cat.codes
        labels = labels.to_numpy()
        return features, labels
    
    @classmethod
    def mnist_test(cls):
        url = "D:/UNi/Thesis/Active Learning/dataset/mnist_test.csv"
        data = pd.read_csv(url, header=None)
        features = data.iloc[:,1:]
        features = features.to_numpy()
        labels = data.iloc[:,0]
        labels = labels.astype('category').cat.codes
        labels = labels.to_numpy()
        return features, labels
    
    @classmethod
    def mammography(cls):
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/mammography.csv"
        data = pd.read_csv(url, header=None)
        features = data.iloc[:,:-1]
        features = features.to_numpy()
        labels = data.iloc[:,-1]
        labels = labels.astype('category').cat.codes
        labels = labels.to_numpy()
        return features, labels

    
    @classmethod
    def online_shopper(cls):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv"
        data = pd.read_csv(url, header=0)
        features = data.iloc[:,:-1]
        features = pd.get_dummies(features)
        features = features.to_numpy()
        labels = data.iloc[:,-1]
        labels = labels.astype('category').cat.codes
        labels = labels.to_numpy()
        return features, labels
    
    @classmethod
    def isolet(cls):
        url = "D:/UNi/Thesis/Data/isolet.data"
        data = pd.read_csv(url, header=None)
        features = data.iloc[:,:-1]
        features = features.to_numpy()
        labels = data.iloc[:,-1]
        labels = labels.astype('category').cat.codes
        labels = labels.to_numpy()
        return features, labels
    
    @classmethod
    def segment(cls):
        url="D:/UNi/Thesis/Data/segmentation.data"
        data=pd.read_csv(url, header=None)
        features = data.iloc[:,1:]
        features = pd.get_dummies(features)
        features = features.to_numpy()
        labels = data.iloc[:,0]
        labels = labels.astype('category').cat.codes
        labels = labels.to_numpy()
        return features, labels
            
    
    @classmethod
    def heart_statlog(cls):
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat"
        data=pd.read_csv(url, header=None, delim_whitespace=True)
        features = data.iloc[:,:-1]
        features = pd.get_dummies(features)
        features = features.to_numpy()
        labels = data.iloc[:,-1]
        labels = labels.astype('category').cat.codes
        labels = labels.to_numpy()
        return features, labels
    
    @classmethod
    def balance_scale(cls):
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data"
        data=pd.read_csv(url, header=None, )
        features = data.iloc[:,1:]
        features = pd.get_dummies(features)
        features = features.to_numpy()
        labels = data.iloc[:,0]
        labels = labels.astype('category').cat.codes
        labels = labels.to_numpy()
        return features, labels
    
    
    @classmethod
    def glass(cls):
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
        data=pd.read_csv(url, header=None, )
        features = data.iloc[:,1:-1]
        features = pd.get_dummies(features)
        features = features.to_numpy()
        labels = data.iloc[:,-1]
        labels = labels.astype('category').cat.codes
        labels = labels.to_numpy()
        return features, labels
    
    @classmethod
    def wine(cls):
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
        data=pd.read_csv(url, header=None, )
        features = data.iloc[:,1:]
        features = pd.get_dummies(features)
        features = features.to_numpy()
        labels = data.iloc[:,0]
        labels = labels.astype('category').cat.codes
        labels = labels.to_numpy()
        return features, labels
    
    @classmethod
    def spam(cls):
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
        data=pd.read_csv(url, header=None, )
        features = data.iloc[:,:-1]
        features = pd.get_dummies(features)
        features = features.to_numpy()
        labels = data.iloc[:,-1]
        labels = labels.astype('category').cat.codes
        labels = labels.to_numpy()
        return features, labels
    
    @classmethod
    def iris(cls):
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
        data=pd.read_csv(url, header=None, )
        features = data.iloc[:,:-1]
        features = pd.get_dummies(features)
        features = features.to_numpy()
        labels = data.iloc[:,-1]
        labels = labels.astype('category').cat.codes
        labels = labels.to_numpy()
        return features, labels
    
    @classmethod
    def magic(cls):
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data"
        data=pd.read_csv(url, header=None, )
        features = data.iloc[:,:-1]
        features = pd.get_dummies(features)
        features = features.to_numpy()
        labels = data.iloc[:,-1]
        labels = labels.astype('category').cat.codes
        labels = labels.to_numpy()
        return features, labels
    
    
    @classmethod
    def abalone(cls):
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
        data=pd.read_csv(url, header=None, )
        features = data.iloc[:,:-1]
        features = pd.get_dummies(features)
        features = features.to_numpy()
        labels = data.iloc[:,-1]
        labels = labels.astype('category').cat.codes
        labels = labels.to_numpy()
        return features, labels
    
    @classmethod
    def yeast(cls):
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data"
        data=pd.read_csv(url, header=None, delim_whitespace=True)
        features = data.iloc[:,1:-1]
        features = pd.get_dummies(features)
        features = features.to_numpy()
        labels = data.iloc[:,-1]
        labels = labels.astype('category').cat.codes
        labels = labels.to_numpy()
        return features, labels
    
    
    @classmethod
    def adult(cls):
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        data=pd.read_csv(url, header=None, )
        features = data.iloc[:,:-1]
        features = pd.get_dummies(features)
        features = features.to_numpy()
        labels = data.iloc[:,-1]
        labels = labels.astype('category').cat.codes
        labels = labels.to_numpy()
        return features, labels

    @classmethod
    def car(cls):
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
        data=pd.read_csv(url, header=None, )
        features = data.iloc[:,:-1]
        features = pd.get_dummies(features)
        features = features.to_numpy()
        labels = data.iloc[:,-1]
        labels = labels.astype('category').cat.codes
        labels = labels.to_numpy()
        return features, labels
    
    @classmethod
    def credit_default(cls):
        try:
            import xlrd
        except:
            raise ImportError("To load this dataset, you need the library 'xlrd'. Try installing: pip install xlrd")
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
        data=pd.read_excel(url, header=1)
        features = data.iloc[:,:-1]
        features = pd.get_dummies(features)
        features = features.to_numpy()
        labels = data.iloc[:,-1]
        labels = labels.astype('category').cat.codes
        labels = labels.to_numpy()
        return features, labels  
    
    @classmethod
    def dermatology(cls):
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/dermatology.data"
        data=pd.read_csv(url, header=None, )
        features = data.iloc[:,1:]
        features = pd.get_dummies(features)
        features = features.to_numpy()
        labels = data.iloc[:,0]
        labels = labels.astype('category').cat.codes
        labels = labels.to_numpy()
        return features, labels
    
    @classmethod
    def diabetic_retinopathy(cls):
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00329/messidor_features.arff"
        data=pd.read_csv(url, skiprows=24, header=None)
        features = data.iloc[:,:-1]
        features = pd.get_dummies(features)
        features = features.to_numpy()
        labels = data.iloc[:,-1]
        labels = labels.astype('category').cat.codes
        labels = labels.to_numpy()
        return features, labels
    
    @classmethod
    def ecoli(cls):
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data"
        data=pd.read_csv(url, header=None, sep='\s+')
        features = data.iloc[:,1:-1]
        features = pd.get_dummies(features)
        features = features.to_numpy()
        labels = data.iloc[:,-1]
        labels = labels.astype('category').cat.codes
        labels = labels.to_numpy()
        return features, labels    
    
    @classmethod
    def eeg_eyes(cls):
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff"
        data=pd.read_csv(url, skiprows=19, header=None, sep=',')
        features = data.iloc[:,:-1]
        features = pd.get_dummies(features)
        features = features.to_numpy()
        labels = data.iloc[:,-1]
        labels = labels.astype('category').cat.codes
        labels = labels.to_numpy()
        return features, labels        
    
    @classmethod
    def haberman(cls):
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data"
        data=pd.read_csv(url, skiprows=0, header=None, sep=',')
        features = data.iloc[:,:-1]
        features = pd.get_dummies(features)
        features = features.to_numpy()
        labels = data.iloc[:,-1]
        labels = labels.astype('category').cat.codes
        labels = labels.to_numpy()
        return features, labels            
    
    @classmethod
    def ionosphere(cls):
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"
        data=pd.read_csv(url, skiprows=0, header=None, sep=',')
        features = data.iloc[:,:-1]
        features = pd.get_dummies(features)
        features = features.to_numpy()
        labels = data.iloc[:,-1]
        labels = labels.astype('category').cat.codes
        labels = labels.to_numpy()
        return features, labels                
                      
    
    @classmethod
    def mice_protein(cls):
        try:
            import xlrd
        except:
            raise ImportError("To load this dataset, you need the library 'xlrd'. Try installing: pip install xlrd")
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00342/Data_Cortex_Nuclear.xls"
        data=pd.read_excel(url, header=0, na_values=['', ' '])
        features = data.iloc[:,1:-4]
        features = features.fillna(value=0)
        features = pd.get_dummies(features)
        features = features.to_numpy()
        labels = data.iloc[:,-1]
        labels = labels.astype('category').cat.codes
        labels = labels.to_numpy()
        return features, labels    
    
    @classmethod
    def nursery(cls):
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data"
        data=pd.read_csv(url, header=None, )
        features = data.iloc[:,:-1]
        features = pd.get_dummies(features)
        features = features.to_numpy()
        labels = data.iloc[:,-1]
        labels = labels.astype('category').cat.codes
        labels = labels.to_numpy()
        return features, labels                            
    
    @classmethod
    def seeds(cls):
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt"
        data=pd.read_csv(url, header=0, sep='\s+')
        features = data.iloc[:,:-1]
        features = pd.get_dummies(features)
        features = features.to_numpy()
        labels = data.iloc[:,-1]
        labels = labels.astype('category').cat.codes
        labels = labels.to_numpy()
        return features, labels          
    
    @classmethod
    def seismic(cls):
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00266/seismic-bumps.arff"
        data=pd.read_csv(url, skiprows=154, header=0, sep=',')
        features = data.iloc[:,:-1]
        features = pd.get_dummies(features)
        features = features.to_numpy()
        labels = data.iloc[:,-1]
        labels = labels.astype('category').cat.codes
        labels = labels.to_numpy()
        return features, labels              
    
    @classmethod
    def soybean(cls):
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/soybean/soybean-small.data"
        data=pd.read_csv(url, skiprows=0, header=0, sep=',')
        features = data.iloc[:,:-1]
        features = pd.get_dummies(features)
        features = features.to_numpy()
        labels = data.iloc[:,-1]
        labels = labels.astype('category').cat.codes
        labels = labels.to_numpy()
        return features, labels                  
    
    @classmethod
    def teaching_assistant(cls):
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/tae/tae.data"
        data=pd.read_csv(url, skiprows=0, header=0, sep=',')
        features = data.iloc[:,:-1]
        features = pd.get_dummies(features)
        features = features.to_numpy()
        labels = data.iloc[:,-1]
        labels = labels.astype('category').cat.codes
        labels = labels.to_numpy()
        return features, labels                      
    
    @classmethod
    def tic_tac_toe(cls):
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data"
        data=pd.read_csv(url, skiprows=0, header=0, sep=',')
        features = data.iloc[:,:-1]
        features = pd.get_dummies(features)
        features = features.to_numpy()
        labels = data.iloc[:,-1]
        labels = labels.astype('category').cat.codes
        labels = labels.to_numpy()
        return features, labels                          
    
    @classmethod
    def website_phishing(cls):
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00327/.old.arff"
        data=pd.read_csv(url, skiprows=36, header=None, sep=',')
        features = data.iloc[:,:-1]
        features = pd.get_dummies(features)
        features = features.to_numpy()
        labels = data.iloc[:,-1]
        labels = labels.astype('category').cat.codes
        labels = labels.to_numpy()
        return features, labels                              
    
    @classmethod
    def wholesale_customers(cls):
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv"
        data=pd.read_csv(url, skiprows=0, header=0, sep=',')
        features = data.iloc[:,2:]
        features = pd.get_dummies(features)
        features = features.to_numpy()
        labels = data.iloc[:,1]
        labels = labels.astype('category').cat.codes
        labels = labels.to_numpy()
        return features, labels     

                           
    
    
