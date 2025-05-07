# LEVEL 2 (ANSWERS)
## 1. How much does performance improve with tuned hyperparameters?

My somewhat randomly selected parameters ($\epsilon=0.2$, $\alpha=0.5$, $\gamma=0.9$) yield a mean reward of $64.96$.
Doing a random search and looking in the ranges ($\epsilon \in [0, 0.6]$, $\alpha \in [0.1, 1]$, $\gamma \in [0, 1]$)
the sweeper returns:
```
  'agent.alpha': 0.6175545161129,
  'agent.gamma': 0.6452185758447,
  'policy.epsilon': 0.1191789683171,
   Current incumbent has a performance of 78.71.
```
So the tuned hyperparameters improve from $64.96$ to $78.71$ which looks significant!

## 2. How does the learning rate affect training steps?
A large learning rate may help in quickly converging to the optimal solution (if it is "far away"), however if it is close it is prone to overshooting
and not converging. Even worse, if it is too high, it may diverge.
A small learning rate may help in finding the optimum if "it is close", however if the optimum is "far away" it may take a long time to reach that optimum,
so training will take a longer time.

Sidenote: A learning rate of $\alpha=0$ is useless, as no q-values are updated.

## 3. What value of $\epsilon$ yields the best performance?
Hypersweeper yields $\epsilon = 0.1191789683171$ which seems to be a good mix between exploration and exploitation.
