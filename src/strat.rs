use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use rayon::prelude::IntoParallelIterator;
use crate::bits::BitSet;
use crate::cache::MemoCache;
use crate::resp::{get_lazy_partitions, get_partition_counts, PartitionMap};
use crate::words::{letter_to_index, N_CHARS, N_LETTERS};

/// Given a list of candidate indices, computes the number of candidates that contain each letter.
pub fn compute_letter_frequencies(
    candidate_indices: &[usize],
    all_candidates: &[[u8; N_CHARS]],
) -> [usize; N_LETTERS] {
    candidate_indices
        .into_par_iter()
        .map(|c_idx| {
            let candidate_word = &all_candidates[*c_idx];
            let mut freq_contribution = [0usize; N_LETTERS];
            for &letter_byte in candidate_word {
                freq_contribution[letter_to_index(letter_byte)] = 1;
            }
            freq_contribution
        })
        .reduce(|| [0usize; N_LETTERS], |mut acc, freq_contribution| {
            for i in 0..N_LETTERS {
                acc[i] += freq_contribution[i];
            }
            acc
        })
}

#[inline]
pub fn get_max_freq_score(
    guess_word: &[u8; N_CHARS],
    letter_frequencies: &[usize; N_LETTERS],
) -> usize {
    let mut score = 0;
    let mut seen_letter = [false; N_LETTERS];
    for &byte in guess_word.iter() {
        let letter_idx = letter_to_index(byte);
        if seen_letter[letter_idx] {
            continue
        }
        seen_letter[letter_idx] = true;
        score += letter_frequencies[letter_idx];
    }
    score
}

pub fn pick_max_freq(
    candidate_indices: &[usize],
    guess_indices: &[usize],
    all_candidates: &[[u8; N_CHARS]],
    all_guesses: &[[u8; N_CHARS]],
) -> usize {
    let letter_frequencies: [usize; N_LETTERS] =
        compute_letter_frequencies(candidate_indices, all_candidates);

    guess_indices
        .par_iter()
        .map(|&g_idx| {
            let guess_word = &all_guesses[g_idx];
            let score = get_max_freq_score(guess_word, &letter_frequencies);
            (usize::MAX - score, g_idx)
        })
        .min()
        .map(|(_, g_idx)| g_idx)
        .expect("No guesses available.")
}

#[inline]
pub fn get_min_remaining_score(
    candidates: &BitSet,
    partition_map: &PartitionMap,
    g_idx: usize,
) -> usize {
    let mut sum = 0;
    let partitions = get_partition_counts(g_idx, candidates, partition_map);
    for (_resp_idx, p_size) in partitions {
        sum += p_size * p_size
    }
    sum
}

pub fn pick_min_remaining(
    candidates: &BitSet,
    guess_indices: &[usize],
    partition_map: &PartitionMap,
) -> usize {
    guess_indices
        .par_iter()
        .map(|&g_idx| {
            let score = get_min_remaining_score(candidates, partition_map, g_idx);
            (score, g_idx)
        })
        .min()
        .map(|(_, g_idx)| g_idx)
        .expect("No guesses available.")
}

#[derive(Debug, thiserror::Error)]
pub enum OptimalGuessError {
    #[error("The current set of candidates is not present in the cache.")]
    CacheMiss,
    #[error("Cache inconsistency: No guess matches the stored optimal cost {0}.")]
    Inconsistency(usize),
}

pub fn pick_optimal(
    candidates: &BitSet,
    n_total_guesses: usize,
    partition_map: &PartitionMap,
    memo: &MemoCache,
) -> Result<usize, OptimalGuessError> {

    // 1. Check if we have a solution for these candidates
    let target_total_cost = match memo.get(candidates) {
        Some(val) => *val,
        None => return Err(OptimalGuessError::CacheMiss),
    };

    let n_candidates = candidates.count_ones();

    // 2. Search for the guess that yields this specific cost
    let found_guess = (0..n_total_guesses)
        .into_par_iter()
        .find_map_first(|g_idx| {
            // Replicate the evaluate_guess logic for a single depth check
            let mut current_guess_cost = n_candidates;

            // Use lazy partitions to avoid unnecessary allocation
            let partitions = get_lazy_partitions(g_idx, candidates, partition_map);

            for (_resp_idx, p_cand, p_size) in partitions {
                if p_size == 1 {
                    current_guess_cost += 1;
                } else if p_size == 2 {
                    current_guess_cost += 3;
                } else {
                    // For p_size > 2, p_cand is guaranteed Some
                    let p_cand = p_cand.unwrap();
                    match memo.get(&p_cand) {
                        Some(cost) => current_guess_cost += *cost,
                        None => return None // Path not in cache, so this isn't the optimal path
                    }
                }
            }

            if current_guess_cost == target_total_cost {
                Some(g_idx)
            } else {
                None
            }
        });

    match found_guess {
        Some(g_idx) => Ok(g_idx),
        None => Err(OptimalGuessError::Inconsistency(target_total_cost)),
    }
}