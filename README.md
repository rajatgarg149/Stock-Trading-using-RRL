# Stock-Trading-using-RRL
This trading strategy is built on the concepts described in the [paper](http://cs229.stanford.edu/proj2006/Molina-StockTradingWithRecurrentReinforcementLearning.pdf) by Molina Gabriel. The project implements the asset trader (agent) using recurrent reinforcement learning(RRL).

Key Elements : Utility Function, Policy and Gradient Ascent

## Utiliy Function:
One commonly used metric in financial engineering is Sharpe's ratio. For this project, sharpe ratio is our utility (or reward) function. The trader will attempt to maximize the sharpe's ratio which technically represents investment strategies with less volatile profit. 
[Sharpe ratio](/sharpeRatio.py) - This file calculates sharpe ratio for given investment returns.
[Reward](/rewardFunction.py) - This file calculates the sharpe ratios for all returns over window size M. 
* sharpeRatio.py
* updateFt.py
* rewardFunction.py
* costFunction.py
* featureNormalize.py
* checkRRLGradient.py
* getNumericalGradient.py
* testDAX.py
* retDAX.txt
* DAX.txt

Here checkRRLGradient.py and getNumericalGradient.py are the utility functions.
