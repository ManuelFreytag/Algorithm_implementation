# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 15:15:03 2016

@author: Manuel
"""

from C45Tree_own import branchingCriterion
from C45Tree_own import split
import pandas as pa
    
def fit(X,y, branching = "gainRatio", splitCriterion = "infoGain", splitNumeric = "binary", gain_thres = 0):
    '''This function fits a decision tree classifier according to the given parameters
    
    Parameters
    ----------
    X : required DataFrame (pandas) of all independent attributes
    y : required list or numpy array of the label values
    branching: "gainRatio", "infoGain", "giniIndex"
    splitCriterion: "infoGain", "giniIndex"
    splitNumeric: "binary", "multiple"
    gain_thres : 
        
    Required modules
    ----------
    pandas
    
    '''
    #transform values to required form
    
    X_matrix = X.values.swapaxes(0,1).tolist()
    
    
    #1. Calculate the best attribute
    best_prio_score = -1
    b_attr = ""
    
    #calculating the old information value of considering soleley the label vals.
   
    labelCount = branchingCriterion.labelCount(y)[1]                      
    oldInfo = branchingCriterion.information(labelCount)
    arrayTypes = X.dtypes
    
    #selecting the best attribute for the next split, according to the set branching factor    
    for x in range(0, len(X_matrix)):
        #If the next attribute is numeric, split it accordingly!
        if(split.typeCheck(arrayTypes[x]) == "numeric"):
            #no splitt possible, as only one value is left
            if(len(X_matrix[x]) == 1):
                X_matrix[x] = str("="+X_matrix[x])
                
            if(splitNumeric == "binary"):
                X_matrix[x] = split.binarySplit(X_matrix[x], y, splitCriterion)
            else:
                raise NotImplementedError("Multiple split has not yet been implemented")
            
        
        if(branching == "gainRatio"):
            prio_score = branchingCriterion.gainRatio(X_matrix[x], y, oldInfo)
        elif(branching == "infoGain"):
            prio_score = branchingCriterion.informationGain(X_matrix[x],y,oldInfo)
        elif(branching == "giniIndex"):
            prio_score = branchingCriterion.giniIndex(X_matrix[x],y)
        else:
            raise NotImplementedError("The given branching criterion could not be recognized")
        
        if(best_prio_score < prio_score):
            best_prio_score = prio_score
            b_attr = X.iloc[:,x].name
    

    #If the best attribute was a numeric, build new values
    pa.options.mode.chained_assignment = None
    X[b_attr] = X_matrix[X.columns.get_loc(b_attr)]
    pa.options.mode.chained_assignment = "warn"

    #2. build rule
    rules = []
    attr_values = X[b_attr].drop_duplicates().values
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
            
            #No splitt necessary for attributes with only one value
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

                    
        print(newSets)
        #build rules recursevly
        for i in range(0, len(newSets)):
            if(newSets[i].values != []):
                rules[i] = [rules[i].copy()] + [fit(newSets[i], newYs[i].values)]
            else:
                rules[i] = rules[i].copy() + [newYs[i].iloc[0]]

            
    return rules
            
    

