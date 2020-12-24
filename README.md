## Janken (Rock Paper Scissors) Algorithms

UCB (Upper Confidence Bound)<sup>1</sup>
* predictable by opponent
* needs high exploration constant and/or frequent resets 

PUCB (Predictor + Upper Confidence Bound)<sup>2</sup>
* predictor (ideally) detects changes in opponent stategy

LSTM (Long Short Term Memory)
* guesses the opponent policy distribution
* difficult to learn all possible opponent strategies (deeper network??)

SER4 (Successive Elimation Rounds with Randomized Round-Robin and Resets)<sup>3</sup>
* very predictable
* bad against high variance strategies

EXP3.R (EXP3 with Resets)<sup>3 4</sup>
* good against exploitation-baised strategies
* lots of hyperparameters to tune

## References
1: https://link.springer.com/article/10.1023/A:1013689704352</n>

2: https://link.springer.com/article/10.1007%2Fs10472-011-9258-6</n>

3: https://link.springer.com/article/10.1007/s41060-017-0050-5</n>

4: https://cseweb.ucsd.edu/~yfreund/papers/bandits.pdf

