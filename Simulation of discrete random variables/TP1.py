import numpy as np
from numpy import zeros, linspace
from numpy.random import rand
from math import *
import matplotlib.pyplot as plt
import matplotlib.colors as colors
lambd=2.0
Nmc=1000
a=0.0
delta=0.02
print (delta)

def V_A_Exponentielle(lambd):
    X=-log(rand())/lambd
    return X

def Chaine_Valeurs_Exponentielle(lambd,Nmc):
    X=zeros(Nmc)
    for n in range(0,Nmc):
        X[n]=V_A_Exponentielle(lambd)
    return X

X=Chaine_Valeurs_Exponentielle(lambd, Nmc)

def V_A_Poisson(lambd):
    n=0
    proba=np.exp(-lambd)
    F=proba
    U=rand()
    while U>F:
        proba=proba*lambd/(n+1);
        F=F+proba
        n=n+1
    return n 

 
def Chaine_Valeurs_Poisson(lambd,Nmc):
    X=zeros(Nmc)
    for n in range(0,Nmc):
        X[n]=V_A_Poisson(lambd)
    return X

Y=Chaine_Valeurs_Poisson(lambd,Nmc)

def Densite_MC(X,a,delta,Nmc):
    N_x=100
    x=zeros(N_x)
    P=zeros(N_x)
    for i in range(0,N_x):
        x[i]=a+delta*i
        counter=0.0
        for n in range(1, Nmc):
            if (X[n] >= x[i]) and (X[n]< x[i] +delta): 
                counter=counter+1
        P[i]=counter/Nmc
 
    fig=plt.figure()
    plt.plot(x,P)
    plt.scatter(x,P)
    y=lambd*np.exp(-lambd*x)
    plt.plot(x,P,'r')
 
Densite_MC(X,a,delta,Nmc)

def Repartition_MC(Y,a,delta,Nmc):
    N_x=100
    x=zeros(N_x)
    P=zeros(N_x)
    for i in range(0,N_x):
        x[i]=a+delta*i
        counter=0.0
        for n in range(1, Nmc):
            if (Y[n]>= x[i]) and (Y[n]<x[i] +delta) : 
                counter=counter+1
        P[i]=counter/(Nmc*delta)
 
    fig=plt.figure()
    plt.plot(x,P)
    plt.scatter(x,P)
    y=lambd*np.exp(-lambd*x)
    plt.plot(x,y,'r')
 
def Repartition_MC2(X,a,delta,Nmc):
    N_x=100
    x=zeros(N_x)
    P=zeros(N_x)
    for i in range(0,N_x):
        x[i]=a+delta*i
        counter=0.0
        for n in range(1, Nmc):
            if X[n]<x[i]:
                counter=counter+1
        P[i]=counter/Nmc
        
    fig=plt.figure()
    plt.scatter(x,P)
    y=1-np.exp(-lambd*x)
    plt.plot(x,y,'r')
    
Repartition_MC2(X, a, delta, Nmc)
    