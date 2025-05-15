import math
import random
import numpy as np
from matplotlib import pyplot as plt 
import statistics as stats

def Laplace():
    U=random.random()
    if U<1/2:
        Z=math.log(2*U)
    else :
        Z=-math.log(2*(1-U))
    return Z

def Normale(N):
    L=[]
    k=math.sqrt(2*math.e/math.pi)  # coeficient k
    C=1/math.sqrt(2*math.pi)       # le coefficient de normalistion de la gaussienne
    while(len(L)<N):               # tant qu'on a pas simuler assez dobservation
        Z=Laplace()                              #on simule une observation d'une va de Laplace Z
        Y=random.random()*math.exp(-abs(Z))*k/2  #on simule une obervation d'unevariable aléaoire Y de loi uniforme sur [0,kg(Z)]
        if Y<math.exp(-Z**2/2)*C:  #On compare Y avec f(Z)
            L.append(Z)            #Si Y<f(Z), Z est une obervation d'une
    return L

Normale(2)


# def DensiteNormale(x,mu,sigma):
#     return 1/(sigma * math.sqrt(2*pi))*exp(-0.5*((x-mu)/sigma)**2)


# normale=np.random.normal(15,3.2,1000)
# plt.hist(normale,density=True,edgecolor='yellow', hatch='x', alpha = 0.2 ,label ='loi normale approchant la loi binomiale')
# lx=np.linspace(0,50,200)
# ly=[DensiteNormale(x,15,3.2) for x in lx]
# plt.plot(lx,ly,'rx', label = ' fonction de densité de la loi normale')
# plt.legend(loc='upper right')
# plt.show()

#Simulation de la variable 

N=100000 #Nombre d'obervations
G=Normale(N)


print("Notre échantillon a pour moyenne empirique",np.mean(G)," et pour variance empirique", np.var(G))


# tracé de l'histogramme et de la densité
# paramétrage de la plage de valeur prises en compte dans le calcul de l'histo (i.e. : toutes les valeurs)
a=math.floor(min(G))-1
b=math.floor(max(G))+1
Delta=0.1
N=int((b-a)/Delta)



# paramétrage la l'affichage
A=-5
B=5
plt.xlim(A,B)
plt.ylim(0,1)

# plt.hist(G,range=(a,b),bins=N,density=True)

x=np.linspace(A,B,101) # crée le vecteur [A, A+eps, A+2epx..... ,B] avec eps=(B-A)/100

plt.plot(x,np.exp(-x**2/2)/np.sqrt(2*np.pi))


plt.show()

Nmc=10000
a=-3
delta=0.06


def  Rejet_Normale():
    C=math.sqrt*(2*math.exp(1)/math.pi)
    k=0
    X=np.zeros(Nmc)
    for n in range(0, Nmc):
        U=rand()
        if U<1/2:
            Y=log(2*U)
        else :
            Y=-log(2*(1-U))
    
        f=(math.exp((-Y**2)/2))/sqrt(2*pi)
        g=math.exp(-abs(Y))/2
        if U<= f/(C*g):
            X[k]=Y
            k=k+1
    return X
X=Rejet_Normale()
print(X)

