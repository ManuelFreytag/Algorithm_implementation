# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 15:57:09 2017

@author: Manuel
"""

import numpy as np
import random
import math
random.seed(42)
np.random.seed(42)


def castInput(inp):
    ret = []
    for x in inp:
        if(type(x) != "numpy.ndarray"):
            t = np.array(x)
            ret += [t]
            if(t.shape == ()):
                raise TypeError("wrong input")
        else:
            ret += [x]
    return inp
    
def kNN_density_estimation(n,m,k):
    """
    kNN density estimation with euclidian distance
    
    Parameter explanation:
    ___________________________________________________________________________
    n:      Vector with new data that is supposed to be estimated
    m:      Matrix with previous data used to train
    k:      Number of k nearest neighbors
    ___________________________________________________________________________
    """
    #Cast the input if its a normal array or array of arrays
    n,m = castInput([n,m])          
    #calculate the distance matrix (in memory)
    if(type(n[0]).__module__ == np.__name__):
        dist = [[np.sqrt((x-y)**2).sum() for y in m] for x in n]
    else: #must look this ugly to increase the performance
        print(type(n[0]))
        dist = [[math.sqrt((x-y)**2) for y in m] for x in n]
    
    #sort the distance
    dist = np.array(dist)
    dist.sort(axis = 1)
        
    #2) get the nearest neighbor distance
    nnVector = dist[:,k-1]
    
    return 1/(m.shape[0]*2*nnVector)


def kNN_classifier(n,m,l,k):
    """
    kNN classifier using the with euclidian distance.
    
    Parameter explanation:
    ___________________________________________________________________________
    n:      test data
    m:      training data
    l:      label values of the train data
    k:      Number of k nearest neighbors
    ___________________________________________________________________________
    """
    #get all label values
    label_vals = np.unique(l)
    
    #filter the data according to the label values
    #estimate all density values of the input parameters for each label
    est = []
    for x in label_vals:
        #filter the rows according to the labels
        tmp = m[l == x]
        est += [kNN_density_estimation(n,tmp,k)]
        
    #for each instance calculate the most proximite class
    #save the label position in a vector
    pred_array = []
    for i in range(est[0].shape[0]):
        tmp_max, it = 0,0
        for j, x in enumerate(est):
            if(x[i] > tmp_max):
                tmp_max, it = x[i], j
        pred_array += [it]
        
    #return the class vector of the predictions
    return [label_vals[x] for x in pred_array]
    
#Testing and visualizing
if __name__ == "__main__":
    #Visualization of
    print("VISUALIZATION OF THE KNN DENSITY ESTIMATION")
    gx = np.arange(-10,45,0.25)
    n = 2000
    c = np.append(np.random.randn(n//2)*5,np.random.randn(n//2)*8+32)

    %time data = kNN_density_estimation(gx,c,100)   #using 100 nearest neighbors for smoothing
    
    #VISUALIZE IT
    import matplotlib.pyplot as plt
    
    plt.plot(gx, data)
    plt.show()
    
    #Performance testing
    from sklearn.datasets import load_digits
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    
    #benchmarking on the digits dataset (~1800 instances already scaled)
    print("PERFORMANCE TESTING OF THE KNN CLASSIFIER")
    digits= load_digits()
    
    #get test and train sets
    test = random.sample(range(digits.data.shape[0]),len(digits.data)//3)
    train = [x for x in range(digits.data.shape[0]) if x not in test]
    
    gnb = GaussianNB()
    %time y_pred = gnb.fit(digits.data[train], digits.target[train]).predict(digits.data[test])
    print("Naive Bayes: Number of mislabeled points out of a total %d points : %d"
          % (len(test),(digits.target[test] != y_pred).sum()))
    
    RF = RandomForestClassifier() 
    %time y_pred = RF.fit(digits.data[train], digits.target[train]).predict(digits.data[test]) 
    print("Random Forest: Number of mislabeled points out of a total %d points : %d"
          % (len(test),(digits.target[test] != y_pred).sum()))
    
    knn = KNeighborsClassifier(n_neighbors=5) #also euclidean distance
    %time y_pred = knn.fit(digits.data[train], digits.target[train]).predict(digits.data[test]) 
    print("KNN scikit: Number of mislabeled points out of a total %d points : %d"
          % (len(test),(digits.target[test] != y_pred).sum()))
    
    #My own KNN implementation  
    %time prediction = kNN_classifier(digits.data[test], digits.data[train],digits.target[train],3) 
    
    #print accuracy
    print("KNN OWN: Number of mislabeled points out of a total %d points : %d"
      % (len(test),(digits.target[test] != prediction).sum()))  
    
    #-> good classification, bad classification speed