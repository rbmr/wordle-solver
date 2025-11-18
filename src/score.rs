use bitvec::vec::BitVec;
use rayon::iter::ParallelIterator;
use crate::game::{get_idx, N_LETTERS, N_CHARS};
use ndarray::ArrayView2;
use rayon::prelude::IntoParallelIterator;

/// Given a list of candidate indices, computes the number of candidates that contain each letter.
pub fn compute_letter_frequencies(
    candidate_indices: &Vec<usize>,
    all_candidates: ArrayView2<u8>,
) -> [usize; N_LETTERS] {
    candidate_indices
        .into_par_iter()
        .map(|c_idx| {
            let candidate_word = all_candidates.row(*c_idx);
            let mut freq_contribution = [0usize; N_LETTERS];
            for &letter_byte in candidate_word {
                freq_contribution[get_idx(&letter_byte)] = 1;
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

pub fn score_max_frequency(
    guess_word: &[u8; N_CHARS],
    letter_frequencies: &[usize; N_LETTERS],
) -> usize {
    let mut score = 0;
    let mut seen_letter = [false; N_LETTERS];
    for &byte in guess_word.iter() {
        let letter_idx = get_idx(&byte);
        if seen_letter[letter_idx] {
            continue
        }
        seen_letter[letter_idx] = true;
        score += letter_frequencies[letter_idx];
    }
    score
}

pub fn score_min_remaining(
    partitions: &Vec<BitVec<u64>>,
) -> usize {
    partitions
        .into_par_iter()
        .map(|partition| {
            let partition_size = partition.count_ones();
            partition_size * partition_size
        })
        .sum()
}