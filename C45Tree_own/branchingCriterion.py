# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 12:11:47 2016

@author: Manuel
"""
import math as m
import pandas as pa
import numpy as np

def giniIndex(x,y):
    #Calculating the gini(S) coefficient of y        
    values = labelCount(x)[0]
    labelC = labelCount(y)[1]
    attLabelFreq = attLabelFrequency(x,y)
    

    giniSplit = 0
    
    for i in values:
        #calulate the giniSplit value
        giniS = 1
        
        for j in attLabelFreq.loc[i,:]:
            giniS += - (j/(attLabelFreq.loc[i,:].sum()))**2
        
        #check for validity of the gini index
        if((giniS > (1-(len(labelC)*((1/len(labelC))**len(labelC))))) | (giniS < 0)):
            raise Warning("The giniS value exceeds the logical given limits", giniS)
        
        giniSplit += (attLabelFreq.loc[i,:].sum()/sum(labelC))*giniS
        
    return giniSplit
        
    
    
    

def gainRatio(x,y, oldInfo):    
    #calculte all val. combinations
    values = pa.Series(x).drop_duplicates()
    counts = []
    
    if(isinstance(x,list) == True):
        for v in values:
            counts = counts + [x.count(v)]
    else:
        for v in values:
            counts = counts + [np.count_nonzero(x == v)]
            
    intrinsic_info = information(counts)
    infoGain = informationGain(x,y,oldInfo)
    
    gainR = (infoGain/intrinsic_info)
    
    return gainR

def informationGain(x, y, oldInfo):
    infoGain = oldInfo - averageInformation(x,y)
    
    return infoGain
    

def averageInformation(x, y):
    '''Calculating the average information gathered by x considering y
    
    This function takes two equaly long iterable objects and calculates the average information.
    Both iterable objects should symbolize a independent attribute (x) and dependent attribute (y)
    
    Parameters
    ----------
    x : values of the independent attribute
    y : values of the dependent attribute
    
    Return
    ---------
    avg_info : double
     
    Required modules:
    ---------
    math, pandas
    
    For more information please refer to: https://en.wikipedia.org/wiki/Entropy_(information_theory)      
    '''

    avg_info = 0
    freq_table = attLabelFrequency(x,y)
    
    #calculate the number of all examples
    nr_ex_all = attLabelFrequency(x,y).sum().sum()
    
    
    for i in freq_table.index:
        freq = freq_table.loc[i,:].values
        #calculate the sum of regarded examples
        avg_info = avg_info + (sum(freq)/nr_ex_all)*information(freq)
    
    return avg_info
    
def labelCount(x):
        
    labelCount = []

    for t in pa.Series(x).drop_duplicates():
        if(isinstance(x,list) == True):
            labelCount = labelCount + [x.count(t)]
        else:
            labelCount = labelCount + [np.count_nonzero(x == t)]
                                
    return [pa.Series(x).drop_duplicates().values, labelCount]


def attLabelFrequency(x,y):
    '''This function counts the frequency of each lable value corresponding to the attribute values
    
    Parameters
    ----------
    x : values of the independent attribute
    y : values of the dependent attribute
    
    Return
    ---------
    freq_table : pandas.DataFrame
    
    Required modules:
    ---------
    pandas
    '''
    
    #Check for exceptions
    if(len(x) != len(y)):
        raise ValueError("The iterable objects have no corresponding size")
    
    x_ser = pa.Series(x).drop_duplicates()
    y_ser = pa.Series(y).drop_duplicates()

    freq_table = pa.DataFrame(index = x_ser.values, columns = y_ser.values)
    
    #fill the dataframe with 0's
    for i in x_ser.values:
        for j in y_ser.values:
            freq_table.loc[i,j] = 0
        
    #For every match, increment in 1
    for i in range(0,len(x)):
        freq_table.loc[x[i],y[i]] = freq_table.loc[x[i],y[i]] + 1

    return freq_table

def information(x):
    '''Calculating the information gathered by the value counts x
    
    Return
    ---------
    info : double
    
    Required modules:
    ---------
    math
    
    For more information please refer to: https://en.wikipedia.org/wiki/Entropy_(information_theory) 
    '''
    
    #transforming x
    sumx= sum(x)
    nx = []
    
    for i in x:
        nx  = nx + [(i/sumx)]
    
    info = entropy(nx)
    return info

def entropy(x):
    '''Calculating the entropy gathered by the class probabilities x
    
    Return
    ---------
    info : double
    
    Required modules:
    ---------
    math
    
    For more information please refer to: https://en.wikipedia.org/wiki/Entropy_(information_theory) 
    '''
    
    e = 0
    for i in x:
        if(i != 0):
            e = e - (i*m.log(i,2))    
    return e