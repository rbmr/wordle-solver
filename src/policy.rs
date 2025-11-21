use rayon::iter::ParallelIterator;
use bitvec::order::Lsb0;
use bitvec::vec::BitVec;
use rayon::prelude::IntoParallelIterator;
use crate::cache::MemoCache;
use crate::part::{generate_partitions, get_partition_counts};
use crate::utils::sum_of_squares;
use crate::words::{letter_to_index, N_CHARS, N_LETTERS};

pub trait Policy: Sync + Send {
    fn pick(&self, candidates: &BitVec<u64, Lsb0>) -> usize;
}

pub struct MaxFreqPolicy<'a> {
    pub all_candidates: &'a [[u8; N_CHARS]],
    pub all_guesses: &'a [[u8; N_CHARS]],
}

impl<'a> Policy for MaxFreqPolicy<'a> {

    #[inline]
    fn pick(&self, candidates: &BitVec<u64, Lsb0>) -> usize {
        let candidate_indices: Vec<usize> = candidates.iter_ones().collect();
        let letter_frequencies: [usize; N_LETTERS] =
            compute_letter_frequencies(&candidate_indices, self.all_candidates);

        (0..self.all_guesses.len())
            .into_par_iter()
            .map(|g_idx| {
                let guess_word = &self.all_guesses[g_idx];
                let score = get_max_freq_score(guess_word, &letter_frequencies);
                (usize::MAX - score, g_idx)
            })
            .min()
            .map(|(_, g_idx)| g_idx)
            .expect("No guesses available.")
    }
}

pub struct MinRemainingPolicy<'a> {
    pub response_cache: &'a [u8],
    pub n_total_candidates: usize,
    pub n_total_guesses: usize,
}

impl<'a> Policy for MinRemainingPolicy<'a> {

    #[inline]
    fn pick(&self, candidates: &BitVec<u64, Lsb0>) -> usize {
        (0..self.n_total_guesses)
            .into_par_iter()
            .map(|g_idx| {
                let score = get_min_remaining_score(
                    candidates, self.response_cache,
                    g_idx, self.n_total_candidates
                );
                (score, g_idx)
            })
            .min()
            .map(|(_, g_idx)| g_idx)
            .expect("No guesses available.")
    }
}

pub struct OptimalPolicy<'a> {
    pub all_guesses: &'a [[u8; N_CHARS]],
    pub response_cache: &'a [u8],
    pub memo: &'a MemoCache,
    pub n_total_candidates: usize,
}

impl<'a> Policy for OptimalPolicy<'a> {

    #[inline]
    fn pick(&self, candidates: &BitVec<u64, Lsb0>) -> usize {
       // Check if cache contains the solution for the current candidates.
        let target_total_cost = match self.memo.get(candidates) {
            Some(val) => *val,
            None => panic!("Optimal Policy Cache Miss"),
        };
        let n_candidates = candidates.count_ones();

        // Find the first guess that satisfies the optimal cost.
        (0..self.all_guesses.len())
            .into_par_iter()
            .find_map_first(|g_idx| {
                let partitions = generate_partitions(
                    candidates, self.response_cache,
                    g_idx, self.n_total_candidates
                );

                let mut current_guess_cost = n_candidates;

                for (partition_candidates, partition_size) in partitions {

                    if partition_size == 1 {
                        current_guess_cost += 1;
                    } else if partition_size == 2 {
                        current_guess_cost += 3;
                    } else {
                        match self.memo.get(&partition_candidates) {
                            Some(cost) => current_guess_cost += *cost,
                            None => return None
                        }
                    }
                }

                if current_guess_cost == target_total_cost {
                    Some(g_idx)
                } else {
                    None
                }
            })
            .expect("Optimal Policy Inconsistency: No guess matches stored cost.")
    }
}

/// Given a list of candidate indices, computes the number of candidates that contain each letter.
#[inline]
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

#[inline]
pub fn get_min_remaining_score(
    candidates: &BitVec<u64, Lsb0>,
    response_cache: &[u8],
    g_idx: usize,
    n_total_candidates: usize,
) -> usize {
    sum_of_squares(get_partition_counts(
        candidates, response_cache, g_idx, n_total_candidates
    ))
}