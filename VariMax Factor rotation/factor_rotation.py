# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 00:01:29 2017

@author: Manuel
"""

import math
import numpy as np

m = np.array([[1,2,3],[1,2,3]])


#implementation based on:
#http://public.dhe.ibm.com/software/analytics/spss/support/Stats/Docs/19.0/Client/User_Manuals/English/IBM_SPSS_Statistics_19_Algorithms.pdf
#Results are equal to the verimax(m) function in R

#we are only implementing the varimax algorithm
def calcSV(matrix):
    n = len(matrix)
    m = len(matrix[0])
    
    SV = 0
    for i in range(m):
        SV += (n*(matrix[:,i]**4).sum() - (((matrix[:,i]**2).sum())**2)/n**2)
    return SV

def varimax(V, rotate = True):

    n = len(V)
    #diagonal matrix of communalities
    Hn = np.diag([((V[i]**2).sum())**(-0.5) for i in range(n)])
    H = np.diag([((V[i]**2).sum())**(0.5) for i in range(n)])

    #normalize
    norm_V = np.dot(Hn,V)

    #rotated_V = varimaxIteration(V)
    rotated_V = varimaxIteration(norm_V)
    new_V = np.dot(H,rotated_V)
    
    #reflect vectors with nevative sum
    return new_V

def varimaxIteration(V):
    tmp_V = V.copy()
    n = len(V)
 
    switches = [[x,y] for x in range(len(V[0])) for y in range(x,len(V[0])) if x != y]
    
    for x in switches:
        fj = tmp_V[:,x[0]]
        fk = tmp_V[:,x[1]]
        
        u = fj**2 - fk**2
        v = 2*fj*fk
        
        A = u.sum()
        B = v.sum()
        C = np.absolute((u**2-v**2)).sum()
        D = (2*u*v).sum()
        
        X = D-(2*A*B)/n
        Y = C-(A**2-B**2)/n
        P = 0.25*math.atan(X/Y)
        
        if(np.absolute(math.sin(P)) <= (10**(-15))):
            continue
        
        trans_m = np.array([[math.cos(P), -math.sin(P)],[math.sin(P),math.cos(P)]])
        tmp_V[:,x] = np.dot(tmp_V[:,x],trans_m)
         
    #print((tmp_V**2).sum())
    if(np.absolute((calcSV(tmp_V) - calcSV(V))) <= (10**(-5))):
        return tmp_V
    else:
        print("end")
        print(tmp_V)
        print((tmp_V**2).sum())
        return varimaxIteration(tmp_V)
      
test = varimax(np.array(m1.components_[:,[0,1]]))
print(test)