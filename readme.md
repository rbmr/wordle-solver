# Wordle Solver

## Word sets

The game of Wordle relies on a definition of the following components.

Each of the following sets contain words of length $N_c$, with characters A-Z, where $N_c = 5$ in classic Wordle.
- The set of allowed guesses $G$.
- The set of (remaining) candidates $C$. We will denote the initial (complete) set of candidates using $C_0$, such that $C \subseteq C_0 \subseteq G$. We assume it is equally probable for all initial candidates to be the secret word.
- The set of all possible responses $R$. Where each character in a response is either green, yellow, or gray. $|R| = 3^{N_c}$. We will the denote the all green response using $r_w$ ("win" response).

## Response function

The response to a guess may be defined using a function $f: (C\times G) \to R$.

This function maps each candidate $c \in C$, guess $g \in G$ pair to the corresponding response $r \in R$ that would be given by Wordle if the $c$ was the secret word, and you guessed $g$.

```plaintext
function compute_response(secret, guess):
    N_c = length(secret)
    assert N_c == length(guess)
    response = array(N_c, GRAY)
    used = array(N_c, false)
    
    // Pass 1: find greens
    for i in 0..N_c-1:
        if guess[i] == secret[i]:
            response[i] = GREEN
            used[i] = true
    
    // Pass 2: find yellows
    for i from 0..N_c-1:
        if response[i] == GREEN:
            continue
        for j in 0..N_c-1:
            if guess[i] == secret[j] AND !used[j]:
                response[i] = YELLOW
                used[j] = true
                break
    
    return response
```

## Partitions

We may partition a candidate set $C$ into subsets $C_{g,r}$, for each guess $g \in G$ response $r \in R$ pair. Where $C_{g,r}$ is the set of words $c\in C$ that would generate response $r$ if $g$ were guessed.

$$C_{g,r} = \{c \in C \mid f(c, g) = r\}$$

Properties: 
- $r_1 \neq r_2 \to C_{g,r_1} \cap C_{g,r_2} = \emptyset$
- $g \in C_{g,r} \to C_{g,r} = \{g\} \land r=\text{all green}$.
- $C_{g,r} = C_{0,g,r} \cap C$

## As a Markov Decision Process

Using the aforementioned definitions, the full game of Wordle may be defined as an infinite horizon Markov Decision Process (MDP) as follows.

- Each state $s$ is just the set of remaining candidate words $C$. We add a terminal win state $\emptyset$ for when the secret word has been guessed correctly. Using this definition, the set of all states $S$ is exactly the power set of the initial candidate set $C_0$.
- Each action is a guess $g \in G$.
- The transition function $P(s' \mid s, g)$ defines the probability of transitioning to a next state $s'$ given current state $s$ and guess $g$.

$$P(\emptyset \mid C, g) = \begin{cases} \frac{1}{|C|} & \text{if } g \in C \\ 0 & \text{otherwise} \end{cases}$$

$$P(C_{g,r} \mid C, g) = \frac{|C_{g,r} \setminus \{g\}|}{|C|}$$

$$P(\emptyset \mid \emptyset, \cdot) = 1$$

  - The cost function $Cost(s)$ penalizes every guess made.

$$Cost(s) = \begin{cases} 0 & \text{if } s = \emptyset \\ 1 & \text{otherwise} \end{cases}$$

  - The discount factor $\gamma = 1$

The goal is to find an optimal policy, $\pi^*(s) \to g$, that minimizes the total expected cost (total expected number of guesses).

Substituting these values into the Bellman optimality equations and simplifying gives:

$$Q^*(C, g) = 1 + \sum_{r \in \hat{R} } \frac{|C_{g,r}|}{|C|} V^*(C_{g,r})$$

$$V^*(C) = \min_{g \in G} Q^*(C, g)$$

Where $\hat{R} = \{r \in R \mid r \neq r_w \}$ is the set of all possible responses excluding the win response.

## Computing the Optimal Strategy feasibly

### The infeasibility

The reason we can't compute the optimal strategy in polynomial time is because the number states is exponential in the number of initial candidates.

$$|S| = 2^{|C_0|}$$

In order to make the algorithm feasible, we must drastically reduce the number of states to visit.

A first insight is to realize that not all states are reachable from $|C_0|$, however, an attempt to find all reachable states via a BFS traversal will quickly show that even this simplified problem is still intractable, showing the same exponential growth.

A next insight is then, that many of these states are only reachable by making terrible guesses that clearly don't correspond to the optimal value. For example, a guess filtering out only one candidate will lead to a new distinct state, but this state is likely irrelevant to the optimal strategy.

These insights combined lead to the following algorithmic optimizations.

### Integer optimization: Minimizing total cost

To avoid the computational overhead and precision issues of floating point arithmetic, we reformulate the objective function. Instead of minimizing the expected number of guesses (which requires division), we minimize the total number of guesses required to solve for all candidates in $C$.

Let $T^*(C)$ be the minimum total guesses for candidate set $C$. We can define the relationship to the expected value $V^*(C)$ as:

$$T^*(C) = |C| \cdot V^*(C)$$

We can then update the Bellman Equations as follows:

$$T^*(C) = |C| \cdot V^*(C) = \min_{g \in G} \left( |C| \cdot 1 + |C| \sum_{r \in \hat{R}} \frac{|C_{g,r}|}{|C|} V^*(C_{g,r}) \right) = \min_{g \in G} \left( |C| + \sum_{r \in \hat{R}} T^*(C_{g,r}) \right)$$

This gives the integer-only recurrence relation. Note that the term $|C|$ represents the fact that the current guess $g$ adds exactly 1 guess to the path of every candidate currently in the set.

Rewriting the recurrence relation in alternating recursive form we get:

$$T^*(C) = \min_{g \in G} T^*(C, g)$$

$$T^*(C, g) = |C| + \sum_{r \in \hat{R}} T^*(C_{g,r})$$

### Base Cases

- Empty Set: $T^*(\emptyset) = 0$
- Single Word: $T^*(\{c\}) = 1$ 
  - The word is guessed immediately.
- Two Words: $T^*(\{c_1, c_2\}) = 3$.
  - 1 guess identifies the first word.
  - 2 guesses identifies the second. 
  - Total: $1+2=3$.

### Memoization

We use a memoization to store T^*(C) for each set of candidates $C$ where $|C| > 2$.

### Lower bounds on $T^*(C)$

In a minimization problem, we require admissible lower bounds (optimistic estimates) on the cost to perform pruning.

Bound 1 (Information Theoretic):Each response from Wordle provides information that reduces the candidate set. To distinguish among $|C|$ possibilities requires at least $\log_{|R|}(|C|)$ responses in expectation. Converting this to total guesses:

$$T^*(C) \geq |C| \cdot \log_{|R|} |C| = |C| \cdot \gamma \log_2 |C|$$

Where $\gamma = 1 / \log_2(|R|) = 1 / \log_2(3^{N_c}) = 1 / (N_c \cdot \log_2(3))$.

Bound 2 (Minimum Depth / Pigeonhole): We derive the absolute minimum number of total guesses required for a set of size $|C|$ by assuming the best-case scenario for every guess. 
1. We must pick a single guess $g$. 
2. If the secret word happens to be $g$ we solve it in 1 guess. This happens for at most one candidate in $C$.  
3. For the remaining $|C| - 1$ candidates, the guess $g$ is incorrect. Therefore, we need at least 1 additional guess to solve them. Giving a path length of at least 2.

$$T^*(C) \geq 1 \cdot 1 + 2 \cdot (|C| - 1) = 2|C| - 1$$

The tightest lower bound $L(C)$ is just the maximum of the two bounds. Assuming $N_c = 5$, Bound 2 is tighter than bound 1 for all relevant candidate set sizes. On top of this, its also more efficient to compute.

Finally, we may compute a lower bound on the specific total cost of a guess $T^*(C,g)$ by using the actual computed optimal value $T^*(C_{g,r})$ where available (memoized), and the lower bound $L(C_{g,r})$ otherwise.

$$T_{LB}(C, g) = |C| + \sum_{r \in \hat{R} } \hat{T}(C_{g,r})$$

where:

$$\hat{T}(S) = \begin{cases} T^*(S) & \text{if } S \text{ is in cache} \\ L(S) & \text{otherwise} \end{cases}$$

### Upper bounds on $T^*(C)$

An upper bound on $T^*(C)$ can be computed using a greedy strategy (heuristic). Since we are minimizing cost, the total cost produced by any valid policy (even a suboptimal one) is a valid upper bound on the true minimal total cost.

Let $h(C)$ be a heuristic policy function that returns a guess $g$ for a set $C$. We can calculate the precise total cost of this policy, denoted as $UB(C)$, by simulating the game tree using $h(C)$ recursively.

$$T^*(C) \leq UB(C)$$

We use this $UB(C)$ to initialize our search. If we find a branch in our search tree with a lower bound exceeding $UB(C)$, we know that branch cannot possibly beat our heuristic, and we can prune it.

### Upper bounds on $V^*(C)$

An upper bound on $V^*(C)$ can be computed using a greedy strategy (heuristic). Since we are minimizing cost, a specific policy (even a suboptimal greedy one) provides a valid upper bound on the true minimal cost. These upper bounds can get relatively tight taking (only) polynomial time.

### Pruning

To solve the problem within a reasonable timeframe, we employ a Branch and Bound strategy to eliminate (prune) guesses that cannot possibly yield an optimal solution. We track the best solution found so far for the current set $C$, denoted as $\beta$, and discard any guess $g$ whose lower bound cost exceeds this value.

The pruning logic proceeds as follows:
1. Initialization ($\beta$): We first compute an upper bound for $T^*(C)$ using a heuristic policy. We set our initial best-known cost $\beta$ to this value. $$\beta \leftarrow UB_{heuristic}(C)$$
2. Guess Ordering: We sort the allowed guesses $g \in G$ based on the heuristic score. Processing promising guesses first allows us to lower $\beta$ earlier in the search, increasing the effectiveness of pruning for subsequent guesses.
3. Incremental Lower Bound Refinement: 
   1. For each guess $g$, we calculate an initial lower bound $T_{LB}(C, g)$ using the static lower bounds $L(S)$ (or memoized values if available) for all resulting partitions. $$T_{LB}(C, g) = |C| + \sum_{r \in \hat{R}} \hat{T}(C_{g,r})$$ 
   2. If $T_{LB}(C, g) \geq \beta$, the guess is immediately pruned. Otherwise, we incrementally refine it by computing the exact costs of the sub-problems.
   3. We iterate through the partitions $C_{g,r}$ sorted by size in descending order. We prioritize larger partitions because they contribute the most to the total cost, causing $T_{LB}$ to rise faster and triggering prune conditions earlier.
   4. For each partition $C_{g,r}$: If the exact cost $T^*(C_{g,r})$ is not yet known (not in cache), we recursively compute it. We update the running lower bound for the guess by replacing the optimistic estimate $L(C_{g,r})$ with the actual cost $T^*(C_{g,r})$. $$T_{LB}(C, g) \leftarrow T_{LB}(C, g) + \left( T^*(C_{g,r}) - L(C_{g,r}) \right)$$
   5. Check: After every update, if $T_{LB}(C, g) \geq \beta$, we stop processing partitions for this guess and prune it immediately.
4. Update Best:If we fully evaluate a guess $g$ (all partitions solved) and the final cost is strictly less than $\beta$, we update our best known solution:$$\beta \leftarrow T_{LB}(C, g)$$

### Representing $C$

The choice of data structure for the candidate set $C$ is really important. Let $N = |C_0|$ be the total number of initial candidates and $k = |C|$ be the number of elements in the current set ($k \le N$).

| Attribute / Operation                     | Bitset (based on $N$) | Sorted List of Indices (based on $k$) |
|:------------------------------------------|:----------------------|:--------------------------------------|
| **Space**                                 | $O(N/64)$             | $O(k)$                                |
| **Time to get element count**             | $O(N/64)$             | $O(1)$                                |
| **Time for intersection/union**           | $O(N/64)$             | $O(k_1 + k_2)$                        |
| **Time to check membership** ($i \in C$?) | $O(1)$                | $O(\log k)$                           |
| **Time to hash**                          | $O(N/64)$             | $O(k)$                                |
| **Time to iterate over elements**         | $O(N/64)$             | $O(k)$                                |
| **Time to create an empty set**           | $O(N/64)$             | $O(1)$                                |
| **Time to add an element**                | $O(1)$                | $O(k)$                                |
| **Time to add a maximum element**         | $O(1)$                | $O(1)$                                |

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