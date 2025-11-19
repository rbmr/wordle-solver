use bitvec::prelude::*;
use crate::resp::{CORRECT_IDX, N_RESPONSES};
use crate::score::{pick_max_freq, pick_min_remaining};
use crate::words::N_CHARS;

pub type PolicyFn<'a> = fn(
    &BitVec<u64, Lsb0>,
    &'a [[u8; N_CHARS]],
    &'a [[u8; N_CHARS]],
    &'a [u8],
    &'a Vec<usize>,
) -> usize;

pub fn compute_partitions(
    candidates: &BitVec<u64, Lsb0>,
    response_cache: &[u8],
    guess_idx: usize,
    n_candidates_total: usize,
) -> Vec<(u8,BitVec<u64>)> {
    let mut partitions: [Option<BitVec<u64, Lsb0>>; N_RESPONSES] =
        std::array::from_fn(|_| None);
    for c_idx in candidates.iter_ones() {
        let resp_idx = response_cache[guess_idx * n_candidates_total + c_idx] as usize;
        partitions[resp_idx]
            .get_or_insert_with(|| bitvec![u64, Lsb0; 0; candidates.len()])
            .set(c_idx, true);
    }
    partitions
        .into_iter()
        .enumerate()
        .filter_map(|(idx, opt_bv)| {
            opt_bv.map(|bv| (idx as u8, bv))
        })
        .collect()
}


/// Context to hold shared data for the recursive heuristic simulation.
struct SimContext<'a> {
    response_cache: &'a [u8],
    all_guesses: &'a [[u8; N_CHARS]],
    all_candidates: &'a [[u8; N_CHARS]],
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
            self.all_guesses,
            self.all_candidates,
            self.response_cache,
            self.c_idx_to_g_idx_map
        );

        // Partition the set C based on the chosen guess g
        let partitions = compute_partitions(
            candidates,
            self.response_cache,
            guess_idx,
            self.all_candidates.len()
        );

        // Calculate Total Cost.
        let mut total_cost = n_candidates;
        for (response, partition) in partitions {
            if response as usize != CORRECT_IDX {
                total_cost += self.simulate_policy(&partition, find_guess_fn);
            }
        }
        total_cost
    }
}

#[inline]
pub fn min_remaining_policy_wrapper(
    candidates: &BitVec<u64, Lsb0>,
    all_guesses: &[[u8; N_CHARS]],
    all_candidates: &[[u8; N_CHARS]],
    response_cache: &[u8],
    _c_idx_to_g_idx_map: &Vec<usize>,
) -> usize {
    let n_guesses = all_guesses.len();
    let all_guesses_indices: Vec<usize> = (0..n_guesses).collect();
    pick_min_remaining(candidates, &all_guesses_indices, response_cache, all_candidates.len())
}

#[inline]
pub fn max_frequency_policy_wrapper(
    candidates: &BitVec<u64, Lsb0>,
    all_guesses: &[[u8; N_CHARS]],
    all_candidates: &[[u8; N_CHARS]],
    _response_cache: &[u8],
    _c_idx_to_g_idx_map: &Vec<usize>,
) -> usize {
    let candidate_indices: Vec<usize> = candidates.iter_ones().collect();
    let n_guesses = all_guesses.len();
    let all_guesses_indices: Vec<usize> = (0..n_guesses).collect();
    pick_max_freq(&candidate_indices, &all_guesses_indices, all_candidates, all_guesses)
}

#[inline]
pub fn min_remaining_hardmode_policy_wrapper(
    candidates: &BitVec<u64, Lsb0>,
    _all_guesses: &[[u8; N_CHARS]],
    all_candidates: &[[u8; N_CHARS]],
    response_cache: &[u8],
    c_idx_to_g_idx_map: &Vec<usize>,
) -> usize {
    let hard_mode_guesses: Vec<usize> = candidates.iter_ones()
        .map(|c_idx| c_idx_to_g_idx_map[c_idx])
        .collect();
    pick_min_remaining(candidates, &hard_mode_guesses, response_cache, all_candidates.len())
}

#[inline]
pub fn max_frequency_hardmode_policy_wrapper(
    candidates: &BitVec<u64, Lsb0>,
    all_guesses: &[[u8; N_CHARS]],
    all_candidates: &[[u8; N_CHARS]],
    _response_cache: &[u8],
    c_idx_to_g_idx_map: &Vec<usize>,
) -> usize {
    let candidate_indices: Vec<usize> = candidates.iter_ones().collect();
    let hard_mode_guesses: Vec<usize> = candidate_indices.iter()
        .map(|&c_idx| c_idx_to_g_idx_map[c_idx])
        .collect();
    pick_max_freq(&candidate_indices, &hard_mode_guesses, all_candidates, all_guesses)
}

pub type UniversalPolicyFn = for<'a> fn(
    &BitVec<u64, Lsb0>,
    &'a [[u8; N_CHARS]],
    &'a [[u8; N_CHARS]],
    &'a [u8],
    &'a Vec<usize>,
) -> usize;

pub const MIN_REMAINING_POLICY: UniversalPolicyFn = min_remaining_policy_wrapper;
pub const MAX_FREQUENCY_POLICY: UniversalPolicyFn = max_frequency_policy_wrapper;
pub const MIN_REMAINING_HARDMODE_POLICY: UniversalPolicyFn = min_remaining_hardmode_policy_wrapper;
pub const MAX_FREQUENCY_HARDMODE_POLICY: UniversalPolicyFn = max_frequency_hardmode_policy_wrapper;


pub fn simulate<'a>(
    initial_candidates: &'a BitVec<u64, Lsb0>,
    all_guesses: &'a [[u8; N_CHARS]],
    all_candidates: &'a [[u8; N_CHARS]],
    response_cache: &'a [u8],
    c_idx_to_g_idx_map: &'a Vec<usize>,
    find_guess_fn: &PolicyFn<'a>,
) -> usize {
    let mut context = SimContext {
        response_cache, all_guesses, all_candidates, c_idx_to_g_idx_map
    };
    context.simulate_policy(initial_candidates, find_guess_fn)
}