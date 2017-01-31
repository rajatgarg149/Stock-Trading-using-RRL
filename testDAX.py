import numpy as np
import scipy.optimize as opt
from scipy.optimize import *
from rewardFunction import rewardFunction
from costFunction import costFunction
from updateFt import updateFt
from featureNormalize import featureNormalize
import matplotlib.pyplot as plt



retDAX = np.loadtxt('retDAX.txt')
DAX = np.loadtxt('DAX.txt')
M = 10
T = 1000 #% The number of time series inputs to the trader
N = 75
initial_theta = np.ones(M+2) #% initialize theta
X = retDAX[:] #% truncate input data
Xn , mu , sigma = featureNormalize(X)
#print fmin(lambda p : costFunction(Xn[0:M+T], X[0:M+T], p)[0],initial_theta)
theta = np.array([-16.7170524, 7.60991612, 2.35687358, 2.93113549 ,1.17745383 , 5.26301218 , -0.03128972 , -1.29282169 ,  0.17975399  , 5.72954635 ,
   7.63644061 , -1.18758394])

print theta
Ft = updateFt(Xn[0:M+T], theta, T)
#print Ft
miu = 1
delta = 0.001
Ret, sharp = rewardFunction(X[0:M+T], miu, delta, Ft, M)
#print sharp
#print Ret
Ret = Ret + 1
#%size(Ret), size(Ft)
for i in range(1,len(Ret)):
    Ret[i] = Ret[i-1]*Ret[i]
#print Ret

#subplot(3,1,1);
#plot(DAX(M+2:M+T+2));
#axis([0, T, min(DAX(M+2:M+T+2))*0.95, max(DAX(M+2:M+T+2))*1.05]);
#subplot(3,1,2);
#plot(Ft(2:end));
#axis([0, length(Ft)-1, -1.05, 1.05]);
#subplot(3,1,3);
#plot(Ret(:));
#axis([0, length(Ret), min(Ret), max(Ret)]);
#pause;
#%Ft(T:T+N)

pI = 10;
Ret = np.zeros(pI*N)
#%size(Ret)
Ft = np.zeros(pI*N)
for i in range(0,pI):
    Ftt = updateFt(Xn[T+i*N:T+(i+1)*N+M], theta, N)
    [Rett, sharp] = rewardFunction(X[T+i*N:T+(i+1)*N+M], miu, delta, Ftt, M)
    Rett = Rett + 1
    Ret[i*N:(i+1)*N] = Rett
    for j in range(i*N,(i+1)*N):
        if j-1 != -1:
            Ret[j] = Ret[j-1]*Ret[j]
    Ft[i*N:(i+1)*N] = Ftt[1:]
    #[theta, cost, EXITFLAG,OUTPUT] = fminunc(@(t)(costFunction(Xn(i*N+1:i*N+M+T), X(i*N+1:i*N+M+T), t)), theta, options)
    theta = fmin(lambda p : costFunction(Xn[i*N:i*N+M+T], X[i*N:i*N+M+T], p)[0],initial_theta,maxfun=100)
    print theta
print Ret
#z = [i for i in range(0,len(Ret))]
#plt.scatter(z,Ret)
plt.plot(Ret)
plt.show()



#figure(2);
#subplot(3,1,1);
#D = DAX(M+T+3:M+T+2+pI*N);
#plot(D);
#axis([0, pI*N, min(D)*0.95, max(D)*1.05]);
#subplot(3,1,2);
#plot(Ft(2:end));
#axis([0, length(Ft)-1, -1.05, 1.05]);
#subplot(3,1,3);
#plot(Ret(:));
#axis([0, length(Ret), min(Ret), max(Ret)]);

