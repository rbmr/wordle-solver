use std::collections::BTreeMap;
use log::warn;
use rayon::prelude::*;
use crate::resp::{get_resp, CORRECT};
use crate::strat::PolicyGraph;
use crate::words::{arr_to_word, N_CHARS};

pub struct SimStats {
    pub total_words: usize,
    pub distribution: BTreeMap<usize, usize>, // NumGuesses -> NumGames
}

impl SimStats {
    pub fn new() -> Self {
        Self {
            total_words: 0,
            distribution: BTreeMap::new(),
        }
    }

    #[inline]
    pub fn inc(&mut self, guesses: usize) {
        self.total_words += 1;
        *self.distribution.entry(guesses).or_insert(0) += 1;
    }

    #[inline]
    pub fn total_guesses(&self) -> usize {
        self.distribution.iter().map(|(k, v)| k * v).sum()
    }

    #[inline]
    pub fn mean(&self) -> f64 {
        if self.total_words == 0 { return 0.0; }
        self.total_guesses() as f64 / self.total_words as f64
    }

    #[inline]
    pub fn variance(&self, mean: f64) -> f64 {
        if self.total_words == 0 { return 0.0; }
        let sum_sq_diff: f64 = self.distribution.iter()
            .map(|(&k, &v)| {
                let diff = k as f64 - mean;
                (diff * diff) * v as f64
            })
            .sum();
        sum_sq_diff / self.total_words as f64
    }

    #[inline]
    pub fn min_max(&self) -> (usize, usize) {
        let min = *self.distribution.keys().min().unwrap_or(&0);
        let max = *self.distribution.keys().max().unwrap_or(&0);
        (min, max)
    }
}

/// Simulates a single game for a specific answer using the provided strategy.
/// Returns the number of guesses taken, or None if the strategy fails (e.g. loops or incomplete).
pub fn trace_game(
    strategy: &PolicyGraph,
    answer: &[u8; N_CHARS],
    all_guesses: &[[u8; N_CHARS]],
) -> Option<usize> {
    let mut steps = 0;
    let mut node_idx = Some(strategy.root);
    const MAX_DEPTH: usize = 20; // Safety breaker for loops

    while let Some(curr_idx) = node_idx {
        steps += 1;
        if steps > MAX_DEPTH { return None; }

        let node = &strategy.nodes[curr_idx];

        // Retrieve the guess word from the global list using the index stored in the node
        let guess = &all_guesses[node.guess];

        // Calculate the actual response for this guess against the true answer
        let resp = get_resp(guess, answer);

        // Check for win condition (GGGGG)
        if resp == CORRECT {
            return Some(steps);
        }

        // Convert response bytes to String key (e.g., "BGYBB") for graph traversal
        let resp_str = arr_to_word(&resp);

        // Transition to the next state
        node_idx = node.children.get(&resp_str).copied();
    }

    None
}

/// Runs a full simulation of the strategy against all provided candidate answers.
pub fn simulate_strategy(
    strategy: &PolicyGraph,
    answers: &[[u8; N_CHARS]],
    all_guesses: &[[u8; N_CHARS]],
) -> SimStats {
    // Run simulations in parallel
    let results: Vec<Option<usize>> = answers
        .par_iter()
        .map(|answer| trace_game(strategy, answer, all_guesses))
        .collect();

    // Collect results in SimStats object
    let mut stats = SimStats::new();
    for res in results {
        match res {
            Some(steps) => stats.inc(steps),
            None => { warn!("Strategy failed to complete a game!"); }
        }
    }
    stats
}