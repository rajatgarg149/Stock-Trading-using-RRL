from math import sqrt
def sharpeRatio(Ret):
	T = len(Ret)
	mean_ret = float(sum(Ret))/T
	mean_sq_ret = float(sum(Ret**2))/T
	if (mean_ret == 0.0) & (mean_sq_ret == 0.0):
		return 0
	sharpe = mean_ret/sqrt(mean_sq_ret - mean_ret*mean_ret)
	return sharpe