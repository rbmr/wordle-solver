use std::collections::BTreeMap;
use bitvec::prelude::*;
use crate::cache::MemoCache;
use crate::resp::{generate_partitions};
use crate::policy::{pick_max_freq, pick_min_remaining, pick_optimal};
use crate::words::N_CHARS;

pub struct SimStats {
    pub total_words: usize,
    // Map<NumGuesses, CountOfWords>
    pub distribution: BTreeMap<usize, usize>,
}

impl SimStats {
    pub fn new() -> Self {
        Self {
            total_words: 0,
            distribution: BTreeMap::new(),
        }
    }

    pub fn inc(&mut self, guesses: usize) {
        self.total_words += 1;
        *self.distribution.entry(guesses).or_insert(0) += 1;
    }

    pub fn total_guesses(&self) -> usize {
        self.distribution.iter().map(|(k, v)| k * v).sum()
    }

    pub fn mean(&self) -> f64 {
        let sum_prod: usize = self.distribution.iter()
            .map(|(k, v)| k * v)
            .sum();
        sum_prod as f64 / self.total_words as f64
    }

    pub fn variance(&self, mean: f64) -> f64 {
        let sum_sq_diff: f64 = self.distribution.iter()
            .map(|(&k, &v)| {
                let diff = k as f64 - mean;
                (diff * diff) * v as f64
            })
            .sum();
        sum_sq_diff / self.total_words as f64
    }

    pub fn min_max(&self) -> (usize, usize) {
        let min = *self.distribution.keys().min().unwrap_or(&0);
        let max = *self.distribution.keys().max().unwrap_or(&0);
        (min, max)
    }
}

/// A unified struct holding all necessary state for a policy decision.
pub struct PolicyState<'a> {
    pub candidates: &'a BitVec<u64, Lsb0>,
    pub all_guesses: &'a [[u8; N_CHARS]],
    pub all_candidates: &'a [[u8; N_CHARS]],
    pub response_cache: &'a [u8],
    pub c_idx_to_g_idx_map: &'a Vec<usize>,
    pub memo: Option<&'a MemoCache>,
}

pub type PolicyFn = fn(&PolicyState) -> usize;

/// Context to hold shared data for the recursive heuristic simulation.
struct SimContext<'a> {
    response_cache: &'a [u8],
    all_guesses: &'a [[u8; N_CHARS]],
    all_candidates: &'a [[u8; N_CHARS]],
    c_idx_to_g_idx_map: &'a Vec<usize>,
    memo: Option<&'a MemoCache>,
    stats: SimStats,
}

impl<'a> SimContext<'a> {

    /// The core recursive function to compute the total guesses for a given heuristic.
    fn run(
        &mut self,
        candidates: &BitVec<u64, Lsb0>,
        current_depth: usize,
        find_guess_fn: PolicyFn,
    ) {
        let n_candidates = candidates.count_ones();

        // Base Cases
        assert!(n_candidates > 0);
        if n_candidates == 1 {
            self.stats.inc(current_depth);
            return;
        }
        if n_candidates == 2 {
            self.stats.inc(current_depth);
            self.stats.inc(current_depth + 1);
            return;
        }

        // Find the guess g_idx according to the heuristic policy π(C)
        let state = PolicyState {
            candidates,
            all_guesses: self.all_guesses,
            all_candidates: self.all_candidates,
            response_cache: self.response_cache,
            c_idx_to_g_idx_map: self.c_idx_to_g_idx_map,
            memo: self.memo,
        };

        // Check if the guess is in the candidate set
        let guess_idx = find_guess_fn(&state);

        // Check if the guess is in the candidate set (green response)
        for c_idx in candidates.iter_ones() {
            if self.c_idx_to_g_idx_map[c_idx] == guess_idx {
                self.stats.inc(current_depth);
                break;
            }
        }

        // Partition the set C based on the chosen guess g
        let partitions = generate_partitions(
            candidates, self.response_cache,
            guess_idx, self.all_candidates.len()
        );

        // Recurse.
        for (partition_candidates, _) in partitions {
            self.run(&partition_candidates, current_depth+1, find_guess_fn);
        }
    }
}

#[inline]
pub fn min_remaining_policy_wrapper(state: &PolicyState) -> usize {
    let n_guesses = state.all_guesses.len();
    let all_guesses_indices: Vec<usize> = (0..n_guesses).collect();
    pick_min_remaining(state.candidates, &all_guesses_indices, state.response_cache, state.all_candidates.len())
}

#[inline]
pub fn max_frequency_policy_wrapper(state: &PolicyState) -> usize {
    let candidate_indices: Vec<usize> = state.candidates.iter_ones().collect();
    let n_guesses = state.all_guesses.len();
    let all_guesses_indices: Vec<usize> = (0..n_guesses).collect();
    pick_max_freq(&candidate_indices, &all_guesses_indices, state.all_candidates, state.all_guesses)
}

#[inline]
pub fn min_remaining_hardmode_policy_wrapper(state: &PolicyState) -> usize {
    let hard_mode_guesses: Vec<usize> = state.candidates.iter_ones()
        .map(|c_idx| state.c_idx_to_g_idx_map[c_idx])
        .collect();
    pick_min_remaining(state.candidates, &hard_mode_guesses, state.response_cache, state.all_candidates.len())
}

#[inline]
pub fn max_frequency_hardmode_policy_wrapper(state: &PolicyState) -> usize {
    let candidate_indices: Vec<usize> = state.candidates.iter_ones().collect();
    let hard_mode_guesses: Vec<usize> = candidate_indices.iter()
        .map(|&c_idx| state.c_idx_to_g_idx_map[c_idx])
        .collect();
    pick_max_freq(&candidate_indices, &hard_mode_guesses, state.all_candidates, state.all_guesses)
}

pub fn optimal_cache_policy_wrapper(state: &PolicyState) -> usize {
    if let Some(cache) = state.memo {
        match pick_optimal(
            state.candidates,
            state.all_guesses,
            state.all_candidates.len(),
            state.response_cache,
            cache
        ) {
            Ok(idx) => idx,
            Err(e) => {
                panic!("Optimal policy failed during simulation: {}", e);
            }
        }
    } else {
        panic!("Optimal policy requires a cache!");
    }
}

pub const MIN_REMAINING_POLICY: PolicyFn = min_remaining_policy_wrapper;
pub const MAX_FREQUENCY_POLICY: PolicyFn = max_frequency_policy_wrapper;
pub const MIN_REMAINING_HARDMODE_POLICY: PolicyFn = min_remaining_hardmode_policy_wrapper;
pub const MAX_FREQUENCY_HARDMODE_POLICY: PolicyFn = max_frequency_hardmode_policy_wrapper;
pub const OPTIMAL_CACHE_POLICY: PolicyFn = optimal_cache_policy_wrapper;

pub fn simulate<'a>(
    initial_candidates: &'a BitVec<u64, Lsb0>,
    all_guesses: &'a [[u8; N_CHARS]],
    all_candidates: &'a [[u8; N_CHARS]],
    response_cache: &'a [u8],
    c_idx_to_g_idx_map: &'a Vec<usize>,
    memo: Option<&'a MemoCache>,
    find_guess_fn: PolicyFn,
) -> SimStats {
    let mut context = SimContext {
        response_cache, all_guesses, all_candidates,
        c_idx_to_g_idx_map, memo, stats: SimStats::new(),
    };
    context.run(initial_candidates, 1, find_guess_fn);
    context.stats
}