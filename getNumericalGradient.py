import numpy as np

def getNumericalGradient( J, theta ):
	numgrad = np.zeros(np.size(theta))
	perturb = np.zeros(np.size(theta))
	e = 1e-4
	for p in range(0,len(theta)):
		perturb[p] = e
		loss1 , grad1 = J(theta - perturb)
		loss2 , grad2 = J(theta + perturb)
		#print loss1 , loss2
		numgrad[p] = (loss2 - loss1) / (2*e)
		perturb[p] = 0
	return numgrad

