# Correctness

For the solver to be valid, the following must be true:
- the value stored in the cache for a candidate set must be the minimum total cost for that set.
- When the secret word is guessed correctly, we only count 1 for the guess itself, not another 1 for hitting the base case n_candidates=1. 
- We only prune a guess or candidate set if it is guaranteed to be EQUAL TO OR WORSE than the current best solution >= beta. 
- The lower bound must be a true lower bound, this means it is less than or equal to the true minimum total cost.
