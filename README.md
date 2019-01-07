# Stock-Trading-using-RRL
This trading strategy is built on the concepts described in the [paper](http://cs229.stanford.edu/proj2006/Molina-StockTradingWithRecurrentReinforcementLearning.pdf) by Molina Gabriel. The project implements the asset trader (agent) using recurrent reinforcement learning(RRL).

Key Elements : Utility Function, Policy and Gradient Ascent

## Utiliy Function:
One commonly used metric in financial engineering is Sharpe's ratio. For this project, sharpe ratio is our utility (or reward) function. The trader will attempt to maximize the sharpe's ratio which technically represents investment strategies with less volatile profit. 

[Sharpe ratio](/sharpeRatio.py) - This file calculates sharpe ratio for given investment returns.

[Reward](/rewardFunction.py) - This file calculates the sharpe ratios for all returns over window size M. 

## Policy:
Here the trader function Ft, is basically a neuron activated with ``tanh`` with the output between -1 and 1. The value of Ft determines the current action.

[Update Ft](/updateFt.py) - This file updates the Ft for the whole time interval.

## Gradient Ascent:
The optimization function needs to maximize the sharpe ratio. Hence, we require gradient ascent.
