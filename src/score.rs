use std::iter::Sum;
use std::ops::Mul;
use bitvec::order::Lsb0;
use bitvec::vec::BitVec;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use crate::game::{letter_to_index, N_LETTERS, N_CHARS, N_RESPONSES, response_to_index};
use ndarray::ArrayView2;
use rayon::prelude::IntoParallelIterator;

/// Given a list of candidate indices, computes the number of candidates that contain each letter.
pub fn compute_letter_frequencies(
    candidate_indices: &[usize],
    all_candidates: ArrayView2<u8>,
) -> [usize; N_LETTERS] {
    candidate_indices
        .into_par_iter()
        .map(|c_idx| {
            let candidate_word = all_candidates.row(*c_idx);
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
    all_candidates_arr: ArrayView2<u8>,
    all_guesses_arr: ArrayView2<u8>,
) -> usize {
    let letter_frequencies: [usize; N_LETTERS] =
        compute_letter_frequencies(candidate_indices, all_candidates_arr);

    guess_indices
        .par_iter()
        .map(|&g_idx| {
            let guess_slice: &[u8] = all_guesses_arr.row(g_idx).to_slice().unwrap();
            let guess_word: &[u8; N_CHARS] = guess_slice.try_into().expect("Guess word must be 5 bytes long.");
            let score = get_max_freq_score(guess_word, &letter_frequencies);
            (usize::MAX - score, g_idx)
        })
        .min()
        .map(|(_, g_idx)| g_idx)
        .expect("No guesses available.")
}

#[inline]
pub fn sum_of_squares<T, I>(xs: I) -> T
where
    T: Mul<Output = T> + Sum + Copy,
    I: IntoIterator<Item = T>,
{
    xs.into_iter().map(|x| x * x).sum()
}

pub fn get_min_remaining_score(
    candidates: &BitVec<u64, Lsb0>,
    response_cache: ArrayView2<[u8; N_CHARS]>,
    g_idx: usize,
) -> usize {
    let mut counts = [0usize; N_RESPONSES];
    for c_idx in candidates.iter_ones() {
        let response = response_cache[[g_idx, c_idx]];
        let index = response_to_index(&response);
        counts[index] += 1;
    }
    sum_of_squares(counts)
}

pub fn pick_min_remaining(
    candidates: &BitVec<u64, Lsb0>,
    all_guesses: &[usize],
    response_cache: ArrayView2<[u8; N_CHARS]>,
) -> usize {
    all_guesses
        .par_iter()
        .map(|&g_idx| {
            let score = get_min_remaining_score(candidates, response_cache, g_idx);
            (score, g_idx)
        })
        .min()
        .map(|(_, g_idx)| g_idx)
        .expect("No guesses available.")
}