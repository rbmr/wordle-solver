# Wordle Solver

## Definition
The game of wordle is defined by the following three components:
- The set of allowed guesses $G$.
- The set of (remaining) candidate words $C \subseteq G$.
- The set of responses $R$. 
- A mapping $f: (C\times G) \to R$ that maps each candidate $c \in C$, guess $g \in G$ pair to the corresponding response $r \in R$ that would be given by Wordle if the $c$ was the secret word, and you guessed $g$. 

## Optimal Strategy

The function we want to minimize is $E(C)$, the minimum expected number of guesses to find the secret word, assuming it is chosen uniformly at random from $C$.

### Base Cases

- If $|C| = 1$, you know the answer. $E(C) = 1$
- If $|C| = 2$, the optimal strategy is to pick a remaining candidate at random. $E(C) = 1.5$

### Recursive Step

For each guess $g \in G$ we partition the candidate set $C$ into smaller, disjoint subsets, $C_{g,r}$, where $C_{g,r}$ is the set of words $c\in C$ that would generate response $r$ if $g$ were guessed.

$$C_{g,r} = \{c \in C \mid f(c, g) = r\}$$

The expected number of total guesses starting from set $C$ with guess $g$, denoted $E(C, g)$, is the cost of the current guess (which is 1) plus the weighted average of the expected future guesses for all resulting subsets $C_{g,r}$.

$$E(C, g) = 1 + \sum_{r \in R \setminus \{\text{all green}\}} \left( \frac{|C_{g,r}|}{|C|} \times E(C_{g,r}) \right)$$

Where:
- $1$ is the cost of the current guess $g$.
- $\frac{|C_{g,r}|}{|C|}$ is the probability of receiving response $r$, as it's the proportion of possible secret words remaining in $C$ that lead to that response.
- $E(C_{g,r})$ is the minimum expected future guesses required for the new, smaller candidate set $C_{g,r}$, calculated recursively.

### Optimal Choice

The optimal guess $g^*$ for the set $C$ is the one that minimizes this expected value over all possible guesses $g \in G$:

$$g^* = \arg \min_{g \in G} E(C, g)$$

And the minimum expected number of guesses for the current set $C$ is defined by this optimal choice:

$$E(C) = E(C, g^*)$$

## Implementation

### Memoization

We can use a cache to store $E(C)$ and $g^*$ for each set of candidates $C$.

### Lower bounds on E(C)

Insight: Each response from Wordle provides information that reduces the candidate set. In the best case, responses partition candidates as evenly as possible.

- The number of possible responses $|R| = 3^{N_c}$
- To distinguish among $|C|$ possibilities requires at least $\log_{|R|}|C|$ responses in expectation. $E(C) \geq \log_{|R|}|C|$ 
- $\log_{|R|}|C| = \log_2|C| / \log_2|R| = \gamma \cdot \log_2|C|$ where we precompute $\gamma = 1 / \log_2|R| = 1 / \log_2 3^{N_c} = N_c / \log_2 3$
{N_c}
We can use this to compute a lower bound on $E(C, g)$ by applying the above equation to the remaining candidates $E(C_{g,r})$:

$$E(C, g) \geq 1 + \sum_{r \in R \setminus \{\text{all green}\}}\left( \frac{|C_{g,r}|}{|C|} \times \gamma \log_2|C_{g,r}|\right)$$

Factoring out constants for computational efficiency gives: 

$$E(C, g) \geq 1 + \frac{\gamma}{|C|} \sum_{r \in R \setminus \{\text{all green}\}}|C_{g,r}| \cdot \log_2|C_{g,r}|$$

### Upper bounds on E(C)

- Greedy upper bounds: The expected number of guesses using a greedy strategy (heuristic) is an upper bound on the optimal strategies number of guesses.
- Instead of initializing the lower bound to infinity, we can initialize it to the upper bound.

### Pruning

We can prune the search space using the lower and upper bounds.

If ever a lower bound for a guess's expected number of guesses exceeds the current minimum, we can skip the guess.

## Heuristics

### Pick Max Frequency

**Goal**: Maximize coverage of common letters across candidates.

Letter frequency across C:
$$\text{freq}(\ell) = |\{c \in C ~|~ \ell \in c\}|$$

For each guess $g$, score by unique letters:

$$\text{score}(g) = \sum_{\ell \in \text{unique}(g)} \text{freq}(\ell)$$
- where $\text{unique}(g)$ is the set of distinct letters in $g$. 

The approximation of the optimal guess is the guess that maximizes the score:

$$g^* = \arg\max_{g \in G} \text{score}(g)$$

### Pick Min Remaining

**Goal**: Minimize expected number of remaining candidates after the guess.

Expected remaining candidates for a given guess $g$:

$$\mathbb{E}[|C_{g,r}|] = \sum_{r \in R} \frac{|C_{g,r}|}{|C|} \cdot |C_{g,r}|= \frac{1}{|C|} \sum_{r \in R} |C_{g,r}|^2$$

$|C|$ is independent of the guess, so we leave it out in the score.

$$\text{score}(g) = \sum_{r \in R} |C_{g,r}|^2$$

The approximation of the optimal guess is the guess that minimizes the score:

$$g^* = \arg\min_{g \in G} \text{score}(g)$$