# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 16:08:04 2016

@author: Manuel
"""

from C45Tree_own import split
import pandas as pa

def apply(X, tree):
    results = []
    
    for x in range(0,len(X.index)):
        temp_tree = tree.copy()
        example = X.loc[x,:]
        
        while(True == True):
            #Search for the correct next value
            for i in range(0,len(temp_tree)):
                node = searchNextNode(temp_tree[i])
                
                #Check for numeric attributes
                try:
                    if(X[node[0]].str.isnumeric().loc[0] == True):
                        #Phrase the first part and cast the second part
                        #check the what portion of the string needs to be removed
                        example = checkAndCompose(example, node)
                        
                except AttributeError:
                    if(split.typeCheck(X[node[0]].loc[0].dtype)=="numeric"):
                        example = checkAndCompose(example, node)
                
                if(example.loc[node[0]] == node[1]):
                    #Cut the correct subtree
                    temp_tree = temp_tree[i]
                    break
            
            #Check if we already have a classification solution
            if(isinstance(temp_tree[0], list) == True):
                #No solution, new cut
                temp_tree = temp_tree[1]
            else:
                #Solution, add the result to array
                results = results + [temp_tree[2]]
                break

    return results
    
def checkAndCompose(example, node):
        pa.options.mode.chained_assignment = None
    
        if(node[1][0:2] == "<="):
            if(float(example.loc[node[0]]) <= float(node[1][2:])):
                example.loc[node[0]] = node[1]
#                example.loc.__setitem__((node[0]), node[1])
        if(node[1][0] == ">"):
            if(float(example.loc[node[0]]) > float(node[1][1:])):
                example.loc[node[0]] = node[1]
#                example.loc.__setitem__((node[0]), node[1])

        pa.options.mode.chained_assignment = 'warn'
                
        return example
    
def searchInTree(tree, path):
    temp_path = path.copy()
    temp_tree = tree.copy()
    
    while(isinstance(temp_tree, list) == True):
        #move one dimension in
        if(len(temp_path) > 0):
            temp_tree = temp_tree[temp_path[0]]
            #Remove done path part
            temp_path.pop(0)
        else:
            #If all parts of the path are used, we search for the very first item
            temp_tree = temp_tree[0]
            
    return temp_tree
    
def searchNextNode(tree):
    temp_tree = tree.copy()
    
    while(isinstance(temp_tree[0], list) == True):
        temp_tree = temp_tree[0]
            
    return temp_tree
    