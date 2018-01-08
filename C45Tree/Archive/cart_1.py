# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 15:17:24 2016

@author: Manuel
"""
import math as m
import pandas as pa
import numpy as np

def fit(X, y, gain_thres = 0):
    '''This function fits a decision tree classifier according to the given parameters
    
    Parameters
    ----------
    X : required DataFrame (pandas) of all independent attributes
    y : required list or numpy array of the label values
    gain_thres : 
        
    Required modules
    ----------
    numpy
    pandas
    
    '''
    rules = selectAndPrun(X,y)
    
    
    return rules
    
def selectAndPrun(X,y):
    #1. Calculate the best attribute
    best_entr = len(X)
    b_attr = ""
    
    #transform values to required form
    X_matrix = X.values.swapaxes(0,1)
    
    for x in range(0, len(X_matrix)):
        if(best_entr > averageInformation(X_matrix[x],y)):
            best_entr = averageInformation(X_matrix[x],y)
            b_attr = X.iloc[:,x].name
    
    attr_values = X[b_attr].drop_duplicates().values
    
    #2. build rule
    rules = []
    
    for a in range(0,len(attr_values)):
        rules = rules + ([[b_attr, attr_values[a]]])
    
    #3. Calculate new sets based on the old rules
    
    #No further prunning possible, as all attributes are regarded!
    if(len(X.columns) > 1):
        #Build new sets & labels
        X = X.assign(y = pa.Series(y, index = X.index))
        newSets = [X[X.loc[:,b_attr]==attr_values[0]]]
        
        for i in range(1,len(attr_values)):
            newSets = newSets + [X[X.loc[:,b_attr]==attr_values[i]]]
            
        #Calculate new & corresponding label values
        newYs = [newSets[0].loc[:,'y']]
        newSets[0] = newSets[0].drop('y',1)
        
        for i in range(1,len(newSets)):
            newYs = newYs + [newSets[i].loc[:,'y']]
            newSets[i] = newSets[i].drop('y',1)
            

        
        #Drop all unuseable attributes
        for i in range(0,len(newSets)):
            newSets[i] = newSets[i].drop(b_attr, 1)
            
            #No splitt necessary if for attributes with only one attribute value
            for j in newSets[i]:
                #If it is not the last column, drop the column
                if(len(newSets[i].columns) > 1):
                    if(len(newSets[i].loc[:,j].drop_duplicates())==1):
                        newSets[i] = newSets[i].drop(j,1)
                #Else drop the whole table
                else:
                    if(len(newSets[i].loc[:,j].drop_duplicates())==1):
                        newSets[i] = pa.DataFrame()
                        
            #Drop all sets if the label value is already completely done
            for j in newYs[i]:
                if(len(newYs[i].drop_duplicates())==1):
                    newSets[i] = pa.DataFrame()
                    newYs[i] = pa.Series()
                    
        #build next rules of suptrees recursevly.
        if(newSets[0].values != []):
            temp_rules = [selectAndPrun(newSets[0], newYs[0].values)]
        else:
            temp_rules = [None]
                      
        for i in range(1,len(newSets)):
            if(newSets[i].values != []):
                temp_rules = temp_rules + [selectAndPrun(newSets[i], newYs[i].values)]
            else:
                temp_rules = temp_rules + [None]
        

        #Combine old rules with new ones
        if(temp_rules != []):
            for a in range(0,len(attr_values)):
                if(temp_rules[a] != None):
                    newAttr = [rules[a].copy()] + [temp_rules[a].copy()]
                    rules[a] = newAttr
            
    return rules
            


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


    
    