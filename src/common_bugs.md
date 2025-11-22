# Correctness

For the solver to be valid, the following must be true:
- the value stored in the cache for a candidate set must be the minimum total cost for that set.
- The cost of the candidate partition containing only the guess (the correct partition) is zero. Therefore, it must be skipped when determining the lower bounds, or the total minimum cost. 
- We only prune a guess or candidate set if it is guaranteed to be EQUAL TO OR WORSE than the current best solution >= beta. 
- The lower bound must be a true lower bound, this means it is less than or equal to the true minimum total cost.
