import numpy as np
def updateFt(X,theta,T):
	M = len(theta)-2
	Ft = np.zeros(T+1)
	#print theta
	xt = np.zeros(M+2).tolist()
	for i in range(1,T+1):
		xt[0] = 1
		xt[1:M+1] = X[i-1:i+M-1]
		xt[M+1] = Ft[i-1]
		Ft[i] = np.tanh(sum(xt*theta))
	#print Ft
	return Ft
