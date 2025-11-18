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

- The reward function $R(s)$ penalizes every guess made, and is therefore independent of the action $a$.

$$R(s) = \begin{cases} 0 & \text{if } s = \emptyset \\ -1 & \text{otherwise} \end{cases}$$

- The discount factor $\gamma = 1$

The goal is to find an optimal policy, $\pi^*(s) \to g$, that minimizes the total expected number of guesses. This is equivalent to maximizing the total expected reward in this formulation.

Substituting these values into the Bellman optimality equations and simplifying gives:

$$Q^*(C, g) = -1 + \sum_{r \in \hat{R} } \frac{|C_{g,r}|}{|C|} V^*(C_{g,r})$$

$$V^*(C) = \max_{g \in G} Q^*(C, g)$$

Where $\hat{R} = \{r \in R \mid r \neq r_w \}$ is the set of all possible responses excluding the win response.

## Computing the Optimal Strategy feasibly

### Base Cases

- $V^*(C) = 0$ if $C = \emptyset$ since the game is over.
- $V^*(C) = -1$ if $|C| = 1$ since the remaining answer is the secret word.
- $V^*(C) = -1.5$ if $|C| = 2$ since the optimal strategy is to pick a word at random. Either the current or the next guess is correct, 50-50 chance.

### Memoization

We can use a cache to store $V^*(C)$ and $\pi^*(C)$ for each set of candidates $C$ where $|C| > 2$.

### Upper bounds on $V^*(C)$

Bound 1: Each response from Wordle provides information that reduces the candidate set. In the best case, responses partition candidates as evenly as possible. Consequently, to distinguish among $|C|$ possibilities requires at least $\log_{|R|}(|C|)$ responses in expectation. We may rewrite this as $\log_{|R|}(|C|) = \log_2(|C|) / \log_2(|R|) = \gamma \log_2(|C|)$ where we precompute $\gamma = 1 / \log_2(|R|) = 1 / \log_2(3^{N_c}) = 1 / (N_c \cdot \log_2(3)) $.

$$V^*(C) \leq -\gamma \log_2|C|$$

Bound 2: Any guess $g \in C$ has a $1/|C|$ chance of being correct. If the guess is incorrect, it takes at least 1 additional guess to find the secret word. This means that in order to distinguish among $|C|$ possibilities, we need at least $1 \cdot \frac{1}{|C|} + 2 \cdot \frac{|C|-1}{|C|} = 2 - \frac{1}{|C|}$ guesses.

$$V^*(C) \leq \frac{1}{|C|} - 2$$

The tightest upper bound $U(C)$ is just the minimum of the two bounds. Assuming $N_c = 5$, Bound 2 is tighter than bound 1 for all candidate set sizes of $|C|$ all the way up until $|C| \approx 59043.5$ which is even larger than $|C_0|$. Consequently, Bound 2 is tighter in all cases, on top of also being more efficient to compute.

Finally, we may compute an upper bound on the optimal guess-value function $Q^*(C,g)$ by using the actual computed optimal state-value function $V^*(C)$ where available, and the aforementioned upper bound $U(C)$ otherwise.

$$Q_{UB}(C, g) = -1 + \sum_{r \in \hat{R} } \frac{|C_{g,r}|}{|C|} \hat{V}(C_{g,r})$$ 
where:
$$\hat{V}(C_{g,r}) = \begin{cases} V^*(C_{g,r}) & \text{if available} \\ U(C_{g,r}) & \text{otherwise} \end{cases}$$

### Lower bounds on $V^*(C)$

A lower bound on the $V^*(C)$ can be computed using a greedy strategy (heuristic). These lower bounds can get relatively tight taking (only) polynomial time.

### Pruning

Pruning: We keep track of a maximum value across all guesses $g \in G$ so far. If an upper bound for a guess' optimal value ever falls below the current maximum, we can skip the guess.

We want to prune as much as possible, which means:
- increasing the maximum value as early as possible
  - by initializing the maximum value to the expected value using a heuristic.
  - by going through the guesses from most to least promising (computed using the heuristic).
- increasing the upper bound for a guess as early as possible 
  - by going through the partitions in reverse order of size

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

### Computing the number of reachable states from |C_0|

The total number of states is $|S| = 2^{|C_0|}$, but not all of them are reachable from |C_0|.

You can find all reachable states and count them using a BFS traversal. 

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