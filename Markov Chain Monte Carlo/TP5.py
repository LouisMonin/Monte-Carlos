import math
import random
from matplotlib import pyplot as plt 
import numpy as np

# programme qui simule une v.a normale (c'est-à dire un echantillon d'observations de taille N)
def NormaleBM(N):
    Nd=int(math.floor(N/2)) # On simule les variables aléatoires par paire
    L=[]
    for i in range(Nd):
        U1=random.random()
        U2=random.random()
        Theta=2*math.pi*U1
        R=math.sqrt(-2*math.log(U2))
        X=R*math.cos(Theta)
        Y=R*math.sin(Theta)
        L.append(X)
        L.append(Y)
    
    # Dans le cas où N est impaire, on simule une derniere va normale
    if N%2==1 :  
        U1=random.random()
        U2=random.random()
        Theta=2*math.pi*U1
        R=math.sqrt(-2*math.log(U2))
        X=R*math.cos(Theta)
        L.append(X)
    
    return L

X=NormaleBM(1000)
print(X)

def IntegraleI(n):
    beta=2
    Norm=NormaleBM(n)
    G=[math.exp(beta*k) for k in Norm]
    S=sum(G)/n
    return(S)

IntegraleI(10000)



def Convergence(N):
    Valeurs=[]
    for i in range(1,N):
        Valeurs.append(IntegraleI(i))
    k=range(1,N)
    plt.plot(k,Valeurs)
    
Convergence(10)

def Convergence(N,p):
    Valeurs=[]
    for i in range(1,N):
        Valeurs.append(IntegraleI(i*p))
    k=np.array(range(1,N))*p
    k=list(k)
    plt.ylim(0,10)
    plt.plot(k,Valeurs)
    
Convergence(100,100)

def IntegraleConv(n):
    beta=2
    Norm=NormaleBM(n)
    Valeurs=[]
    S=0
    for k in range(1,n):
        S=S+math.exp(beta*Norm[k-1])
        Valeurs.append(S/k)
    plt.ylim(0,10)
    plt.plot(Valeurs)
    
IntegraleConv(10000)


# Simulation d'une observation d'une va exponentielle :
def expObs(lam):
    U=random.random()
    X=-math.log(1-U)/lam  # X=F^-1(U)
    return(X)

def exp(N,lam): #lam : paramètre de la loi exponentielle ; N:taille de l'échantillon
    E=[]
    for k in range(N):
        Obs=expObs(lam)
        E.append(Obs)           # on ajoute cette observation au vecteur
    return(E)

def IntegraleII(n):
    beta=2
    Exp=exp(n,beta)
    G=[math.exp(-k**2/2)/(beta*math.sqrt(2*math.pi)) for k in Exp]
    S=sum(G)/n
    return(S)

IntegraleII(1000)

def ConvergenceII(N):
    Valeurs=[]
    for i in range(1,N):
        Valeurs.append(IntegraleII(i))
    k=range(1,N)
    plt.plot(k,Valeurs)
    print(Valeurs)
ConvergenceII(1000)

def ConvergenceIIp(N,p):
    Valeurs=[]
    for i in range(1,N):
        Valeurs.append(IntegraleII(i*p))
    k=np.array(range(1,N))*p
    k=list(k)
    plt.ylim(0,1)
    plt.plot(k,Valeurs)
    
ConvergenceIIp(20,1)

