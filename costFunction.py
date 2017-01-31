from math import sqrt
from sympy import *
from rewardFunction import rewardFunction
from updateFt import updateFt
import numpy as np
from numpy.matlib import repmat
from sympy import *
def costFunction( Xn, X, theta):
    miu = 1
    delta = 0.001        
    M = len(theta) - 2
    T = len(X) - M
    a, b = symbols('a b', real=True)
    dSdA = diff(a/(b-a*a)**.5,a)
    dSdB = diff(a/(b-a*a)**.5,b)
    Ft = updateFt(Xn, theta, T)
    #print Ft
    Ret, sharpe = rewardFunction(X, miu, delta, Ft, M)
    J = sharpe * -1
    dFt = np.zeros((T+1,M+2))
    for i in range(1,T+1):
        xt = [1]
        xt.extend(Xn[i-1:i+M-1])
        xt.extend([Ft[i-1]])
        tanh_out = np.tanh(sum(xt*theta))
        dFt[i] = (1 - tanh_out*tanh_out) * (xt + theta[M+1]*dFt[i-1])

    dRtFt = -1 * miu * delta * np.sign(Ft[1:]-Ft[:T])
    dRtFt = np.reshape(dRtFt,(T,1))
    dRtFtt = miu * X[M:M+T] + miu * delta * np.sign(Ft[1:]-Ft[:T])
    dRtFtt = np.reshape(dRtFtt,(T,1))
    #print dFt[1:].T.shape
    
    A = float(sum(Ret)) / T
    B = float(sum(Ret**2)) / T
    #print dSdA ,dSdB
    #prefix = (repmat((dSdA.subs(a,A)).subs(b,B), T, 1)/T) + np.reshape(((dSdB.subs(a,A)).subs(b,B)*2*Ret/T),(T,1))
    #print prefix.T
    #prefix = repmat(subs(subs(dSdA,a,A),b,B), T, 1)/T + subs(subs(dSdB,a,A),b,B)*2*Ret/T    
    #grad = sum(repmat(prefix', M+2, 1) .* (repmat(dRtFt', M+2, 1) .* dFt(:,2:end) + repmat(dRtFtt', M+2, 1) .* dFt(:,1:T)), 2)
    #grad = np.sum(repmat(prefix.T, M+2, 1) * (repmat(dRtFt.T, M+2, 1) * dFt[1:].T + repmat(dRtFtt.T, M+2, 1) * dFt[:T].T), 1)
    #print grad
    grad = 0
    grad = grad * -1
    #print len(dFt) , T
    #print J , grad
    return J , grad



