use std::collections::HashMap;
use bitvec::prelude::*;
use ndarray::{ArrayView2};

use crate::game::{N_CHARS, CORRECT,};
use crate::score::{pick_max_freq, pick_min_remaining};

pub type PolicyFn<'a> = fn(
    &BitVec<u64, Lsb0>,
    ArrayView2<'a, u8>,
    ArrayView2<'a, u8>,
    ArrayView2<'a, [u8; N_CHARS]>,
    &'a Vec<usize>,
) -> usize;

pub fn compute_partitions(candidates: &BitVec<u64, Lsb0>, response_cache: ArrayView2<[u8; N_CHARS]>, guess_idx: usize) -> HashMap<[u8; N_CHARS],BitVec<u64>> {
    let mut partitions: HashMap<[u8; N_CHARS],BitVec<u64>> = HashMap::new();

    // Iterate over the current candidates
    for candidate_idx in candidates.iter_ones() {
        let response = response_cache[[guess_idx, candidate_idx]];
        partitions
            .entry(response)
            .or_insert_with(|| bitvec![u64, Lsb0; 0; candidates.len()])
            .set(candidate_idx, true);
    }
    partitions
}


/// Context to hold shared data for the recursive heuristic simulation.
struct SimContext<'a> {
    response_cache: ArrayView2<'a, [u8; N_CHARS]>,
    all_guesses_arr: ArrayView2<'a, u8>,
    all_candidates_arr: ArrayView2<'a, u8>,
    c_idx_to_g_idx_map: &'a Vec<usize>,
}

impl<'a> SimContext<'a> {

    /// The core recursive function to compute the total guesses (UB) for a given heuristic.
    fn simulate_policy(
        &mut self,
        candidates: &BitVec<u64, Lsb0>,
        find_guess_fn: &PolicyFn<'a>,
    ) -> usize {
        let n_candidates = candidates.count_ones();

        // Base Cases
        assert!(n_candidates > 0);
        if n_candidates == 1 { return 1; }
        if n_candidates == 2 { return 3; }

        // Find the guess g_idx according to the heuristic policy π(C)
        let guess_idx = find_guess_fn(
            candidates,
            self.all_guesses_arr,
            self.all_candidates_arr,
            self.response_cache,
            self.c_idx_to_g_idx_map
        );

        // Partition the set C based on the chosen guess g
        let partitions = compute_partitions(
            candidates, self.response_cache, guess_idx
        );

        // Calculate Total Cost.
        let mut total_cost = n_candidates;
        for (response, partition) in partitions {
            if response != CORRECT {
                total_cost += self.simulate_policy(&partition, find_guess_fn);
            }
        }
        total_cost
    }
}

#[inline]
pub fn min_remaining_policy_wrapper(
    candidates: &BitVec<u64, Lsb0>,
    all_guesses_arr: ArrayView2<u8>,
    _all_candidates_arr: ArrayView2<u8>,
    response_cache: ArrayView2<[u8; N_CHARS]>,
    _c_idx_to_g_idx_map: &Vec<usize>,
) -> usize {
    let n_guesses = all_guesses_arr.nrows();
    let all_guesses: Vec<usize> = (0..n_guesses).collect();
    pick_min_remaining(candidates, &all_guesses, response_cache)
}

#[inline]
pub fn max_frequency_policy_wrapper(
    candidates: &BitVec<u64, Lsb0>,
    all_guesses_arr: ArrayView2<u8>,
    all_candidates_arr: ArrayView2<u8>,
    _response_cache: ArrayView2<[u8; N_CHARS]>,
    _c_idx_to_g_idx_map: &Vec<usize>,
) -> usize {
    let candidate_indices: Vec<usize> = candidates.iter_ones().collect();
    let n_guesses = all_guesses_arr.nrows();
    let all_guesses: Vec<usize> = (0..n_guesses).collect();
    pick_max_freq(&candidate_indices, &all_guesses, all_candidates_arr, all_guesses_arr)
}

#[inline]
pub fn min_remaining_hardmode_policy_wrapper(
    candidates: &BitVec<u64, Lsb0>,
    _all_guesses_arr: ArrayView2<u8>,
    _all_candidates_arr: ArrayView2<u8>,
    response_cache: ArrayView2<[u8; N_CHARS]>,
    c_idx_to_g_idx_map: &Vec<usize>,
) -> usize {
    let hard_mode_guesses: Vec<usize> = candidates.iter_ones()
        .map(|c_idx| c_idx_to_g_idx_map[c_idx])
        .collect();
    pick_min_remaining(candidates, &hard_mode_guesses, response_cache)
}

#[inline]
pub fn max_frequency_hardmode_policy_wrapper(
    candidates: &BitVec<u64, Lsb0>,
    all_guesses_arr: ArrayView2<u8>,
    all_candidates_arr: ArrayView2<u8>,
    _response_cache: ArrayView2<[u8; N_CHARS]>,
    c_idx_to_g_idx_map: &Vec<usize>,
) -> usize {
    let candidate_indices: Vec<usize> = candidates.iter_ones().collect();
    let hard_mode_guesses: Vec<usize> = candidate_indices.iter()
        .map(|&c_idx| c_idx_to_g_idx_map[c_idx])
        .collect();
    pick_max_freq(&candidate_indices, &hard_mode_guesses, all_candidates_arr, all_guesses_arr)
}

pub type UniversalPolicyFn = for<'a> fn(
    &BitVec<u64, Lsb0>,
    ArrayView2<'a, u8>,
    ArrayView2<'a, u8>,
    ArrayView2<'a, [u8; N_CHARS]>,
    &'a Vec<usize>,
) -> usize;

pub const MIN_REMAINING_POLICY: UniversalPolicyFn = min_remaining_policy_wrapper;
pub const MAX_FREQUENCY_POLICY: UniversalPolicyFn = max_frequency_policy_wrapper;
pub const MIN_REMAINING_HARDMODE_POLICY: UniversalPolicyFn = min_remaining_hardmode_policy_wrapper;
pub const MAX_FREQUENCY_HARDMODE_POLICY: UniversalPolicyFn = max_frequency_hardmode_policy_wrapper;


pub fn simulate<'a>(
    initial_candidates: &'a BitVec<u64, Lsb0>,
    all_guesses_arr: ArrayView2<'a, u8>,
    all_candidates_arr: ArrayView2<'a, u8>,
    response_cache: ArrayView2<'a, [u8; N_CHARS]>,
    c_idx_to_g_idx_map: &'a Vec<usize>,
    find_guess_fn: &PolicyFn<'a>,
) -> usize {
    let mut context = SimContext {
        response_cache, all_guesses_arr, all_candidates_arr, c_idx_to_g_idx_map
    };
    context.simulate_policy(initial_candidates, find_guess_fn)
}