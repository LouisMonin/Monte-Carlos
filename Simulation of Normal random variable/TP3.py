import random
import math
import numpy as np
from matplotlib import pyplot as plt 
from scipy.stats import triang


    
def fonctionRepartition(n):
    x=[0]*100
    for i in range(n):
        x=random.random()
        if x<1:
            x[i]=(x**2)/2
        else:
            x[i]=(-(2-x)**2)/(2+1)
        return x


# version (ultra) courte du progamme de simulation d'exponentielle
def exp(N,lam):
    E=[-math.log(1-random.random())/lam for k in range(N)]
    return E

# programme qui simule une observation d'une v.a normale par application du TCL sur des v.a. uniforme
def ObsNormaleTCL(n):
    G=0
    for i in range(n):
        G+=random.random()
    return (G-n/2)/math.sqrt(n/12)

# programme qui simule une v.a normale (c'est-à dire un echantillon d'observation de taille N)
def NormaleTCL(N,n):
    G=[]
    for i in range(N):
        G.append(ObsNormaleTCL(n))
    return G

# Simulation de la variable 
N=100000 # nombre d'obervations
n=10 # nombre de va sous-jacentes utilisé dans le TCL pour la simulation de la variable
G=NormaleTCL(N,n)

print("Notre échantillon a pour moyenne empirique",np.mean(G)," et pour variance empirique", np.var(G))



# tracé de l'histogramme et de la densité
# paramétrage de la plage de valeur prises en compte dans le calcul de l'histo (i.e. : toutes les valeurs)
a=math.floor(min(G))-1
b=math.floor(max(G))+1
Delta=0.1
N=int((b-a)/Delta)

# paramétrage la l'affichage
A=-3
B=3
plt.xlim(A,B)
plt.ylim(0,1)

plt.hist(G,range=(a,b),bins=N,density=True)

x=np.linspace(A,B,101) # crée le vecteur [A, A+eps, A+2epx..... ,B] avec eps=(B-A)/100

plt.plot(x,np.exp(-x**2/2)/np.sqrt(2*np.pi))  #

plt.show()

# programme qui simule une v.a normale (c'est-à dire un echantillon d'observation de taille N)
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

#tests 
print(NormaleBM(5))
print(NormaleBM(6))

# Simulation de la variable 
N=10000 #Nombre d'observation
G=NormaleBM(N)

print("Notre échantillon a pour moyenne empirique",np.mean(G)," et pour variance empirique", np.var(G))

# Simulation de la variable 
N=10000 #Nombre d'obervations
G=NormaleBM(N)

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

plt.hist(G,range=(a,b),bins=N,density=True)

x=np.linspace(A,B,101) # crée le vecteur [A, A+eps, A+2epx..... ,B] avec eps=(B-A)/100

plt.plot(x,np.exp(-x**2/2)/np.sqrt(2*np.pi))  #

plt.show()

# programme qui simule une v.a normale (c'est-à dire un echantillon d'observation de taille N)
def NormaleMars(N):
    L=[]
    while(len(L)<N):
        V1=2*random.random()-1
        V2=2*random.random()-1
        S=V1**2+V2**2
        if (S<1):
            X=math.sqrt(-2*math.log(S)/S)*V1     
            Y=math.sqrt(-2*math.log(S)/S)*V2
            L.append(X)
            L.append(Y)
    return L[0:N]

print(NormaleMars(6))
print(NormaleMars(5))

# Simulation de la variable 
N=100000 #Nombre d'obervations
G=NormaleMars(N)

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

plt.hist(G,range=(a,b),bins=N,density=True)

x=np.linspace(A,B,101) # crée le vecteur [A, A+eps, A+2epx..... ,B] avec eps=(B-A)/100

plt.plot(x,np.exp(-x**2/2)/np.sqrt(2*np.pi))  #

plt.show()

# programme qui simule une v.a normale (c'est-à dire un echantillon d'observation de taille N)
def NormaleMars(N):
    L=[]
    while(len(L)<N):
        V1=2*random.random()-1
        V2=2*random.random()-1
        S=V1**2+V2**2
        if (S<1):
            X=math.sqrt(-2*math.log(S)/S)*V1     
            Y=math.sqrt(-2*math.log(S)/S)*V2
            L.append(X)
            L.append(Y)
    return L[0:N]

print(NormaleMars(6))
print(NormaleMars(5))

# Simulation de la variable 
N=100000 #Nombre d'obervations
G=NormaleMars(N)

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

plt.hist(G,range=(a,b),bins=N,density=True)

x=np.linspace(A,B,101) # crée le vecteur [A, A+eps, A+2epx..... ,B] avec eps=(B-A)/100

plt.plot(x,np.exp(-x**2/2)/np.sqrt(2*np.pi))  #

plt.show()

def NormaleGenerale(N,mu,sigma):  # les paramètres sont l'esperance mu et l'ECART-TYPE
        G=np.array(NormaleBM(N))     # on transforme le résulat en tableau numpy pour pouvoir le manipuler plus aisément
        G=G*sigma+mu 
        G=list(G)                 # on repasse en liste (ça n'est en réalité pas nécessaire)
        return G
    
# Simulation de la variable 
N=10000# nbre d'obervations
mu=2
sigma=3
G=NormaleGenerale(N,mu,sigma)

print("Notre échantillon a pour moyenne empirique",np.mean(G)," et pour variance empirique", np.var(G))


# tracé de l'histogramme et de la densité
# paramétrage de la plage de valeur prises en compte dans le calcul de l'histo (i.e. : toutes les valeurs)
a=math.floor(min(G))-1
b=math.floor(max(G))+1
Delta=0.1
N=int((b-a)/Delta)

# paramétrage la l'affichage
A=-3
B=7
plt.xlim(A,B)
plt.ylim(0,1)

plt.hist(G,range=(a,b),bins=N,density=True)

x=np.linspace(A,B,101) # crée le vecteur [A, A+eps, A+2epx..... ,B] avec eps=(B-A)/100
plt.plot(x,np.exp(-(x-mu)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi)))  #

plt.show()

