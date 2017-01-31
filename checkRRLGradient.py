from featureNormalize import featureNormalize
from costFunction import costFunction
from getNumericalGradient import getNumericalGradient
import numpy as np
retDAX = np.loadtxt('retDAX.txt')
DAX = np.loadtxt('DAX.txt')
M = 10
T = 1000
X = retDAX[0:M+T]
#X = np.array(X)
Xn , mu , sigma = featureNormalize(X)
debug_theta = np.zeros(M+2)
#debug_theta = np.ones(M+2)
#debug_theta = reshape(sin(debug_theta[0:len(debug_theta)]), size(debug_theta))
J, grad = costFunction(Xn, X, debug_theta)
#costFunc = @(p) costFunction(Xn, X, p)
costFunc = lambda p: costFunction(Xn, X, p)
numgrad = getNumericalGradient(costFunc, debug_theta)
for i in range(0,len(grad)):
	#print numgrad[i] , grad[i]
	print grad[i]



