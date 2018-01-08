# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 11:57:24 2016

@author: Manuel
"""

from C45Tree_own import branchingCriterion
import pandas as pa

def binarySplit(x, y, splitting = "infoGain"):
    '''This function transforms all numeric attributes in two values based on
    the maximum information gain'''
        
    #Sort the array & y accordingly
    df = pa.DataFrame()
    df["x"] = x
    df["y"] = y
    df = df.sort_values(by = "x")
    x_new = df["x"]
    y_new = df["y"]
    

    #calculate the initial information
    labelC = branchingCriterion.labelCount(y)[1]                             
    oldInfo = branchingCriterion.information(labelC)
        
    #for each potential split, calculate the information gain & decide on it
    #Use the sorted values to decide on it!
    
    bestSplitValue = 0
    bestSpValue = -1
    lastValue = ""
    
    #This is calculated for every splitpossibility
    for sp in x_new:
        #Check if new splitting possibility
        if(lastValue != sp):
            lastValue = sp
            
            #Build new temp array
            tempArray = []
            for el in x_new:
                if(el <= lastValue):
                    tempArray = tempArray + ["<="+str(lastValue)]
                else:
                    tempArray = tempArray + [">"+str(lastValue)]
            
                        
            #Calculate new average information
            if(splitting == "infoGain"):
                spValue = branchingCriterion.informationGain(tempArray,y_new.values, oldInfo)
            elif(splitting == "giniIndex"):
                spValue = branchingCriterion.giniIndex(tempArray,y_new)
            else:
                raise ValueError("The given splitting criterion can not be identified")
            
            #Check if the new InfoGain exceeds maximum

#            print(spValue)
            if(bestSpValue < spValue):
                bestSpValue = spValue
#                print("new best: ", bestSpValue)
                bestSplitValue = sp
                
    #Build new array accroding to the best split and return it
    retArray = []

    for el in x:
        if(el <= bestSplitValue):
            retArray = retArray + ["<="+str(bestSplitValue)]
        else:
            retArray = retArray + [">"+str(bestSplitValue)]
    
    
    return retArray            
   


def typeCheck(dtype):
    ''' This function is used to classify all attributes as "discrete" or "numeric"
    
    As the only acceptable input for the tree learner is a DataFrame, all values are stored in a np.array
    Therefore, the dtypes of the input all can be modeled according to 
    https://docs.scipy.org/doc/numpy/user/basics.types.html
    
    Parameter
    ---------
    x = dtype'''
        
    discreteTypes = ["bool_", "bool", "object"]
    numericTypes = ["int_", "intc", "intp", "int8", "int16", "int32", "int64", "uint8", "uint16", 
                    "uint32", "uint64", "float_", "float16", "float32", "float64", "complex_", "complex64",
                    "complex128"]
                    
    dynamicDiscrete = ["U","V","S","a","O","b"]
    dynamicNumeric = ["M","m","c","f","u","i"]
                    
    #Checking for discrete and numeric
    
    #Check for numeric full
    try:
        discreteTypes.index(dtype)
        return "discrete"
    except ValueError:
        #check for discrete
        try:
            numericTypes.index(dtype)
            return "numeric"
        except ValueError:
            #check for dynamic discrete
            try:
                dynamicDiscrete.index(dtype[0])
                return "discrete"
            except ValueError:
                #check for dynamic numeric
                try:
                    dynamicNumeric.index(dtype[0])
                    return "numeric"
                except ValueError:
                    print("No classification possible")
                    
        

    
    
    
    
    
