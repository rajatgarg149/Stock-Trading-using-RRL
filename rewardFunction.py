from sharpeRatio import sharpeRatio
def rewardFunction(X,miu,delta,Ft,M):
	T = len(Ft)-1
	Ret = miu*(Ft[0:T]*X[M:M+T] - delta * abs(Ft[1:]-Ft[0:T]))
	sharpe = sharpeRatio(Ret)
	return Ret , sharpe