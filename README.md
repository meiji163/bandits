## Janken (Rock Paper Scissors) Algorithms

RNN (LSTM or GRU) 
* guesses the opponent policy distribution based on sequence of previous moves
* difficult to learn all possible opponent strategies (deeper network?)

UCB (Upper Confidence Bound)<sup>1</sup>
* balance exploitation/exploration
* predictable by opponent
* needs high exploration constant and/or frequent resets 

PUCB (Predictor + Upper Confidence Bound)<sup>2</sup>
* predictor (ideally) detects changes in opponent stategy

SER4 (Successive Elimation Rounds with Randomized Round-Robin and Resets)<sup>3</sup>
* runs several random trials to find move with highest mean reward 
* assumes constant oppponent distribution
* bad against high variance strategies

EXP3.R (EXP3 with Resets)<sup>3 4</sup>
* updates probabilities based on mean rewards and prior 
* reset based on detection of maximum mean reward drift
* good against exploitation-biased strategies

Bayesian (Thompson Sampling)<sup>5</sup>
* use beta distribution to model reward probabilities and update based on observations
* assumes constant opponent distribution 

## References
1 https://link.springer.com/article/10.1023/A:1013689704352
2 https://link.springer.com/article/10.1007%2Fs10472-011-9258-6
3 https://link.springer.com/article/10.1007/s41060-017-0050-5
4 https://cseweb.ucsd.edu/~yfreund/papers/bandits.pdf
5 https://arxiv.org/pdf/1707.02038.pdf
