use rayon::iter::ParallelIterator;
use rayon::iter::IndexedParallelIterator;
use log::info;
use rayon::slice::ParallelSliceMut;
use crate::utils::{format_bytes};
use crate::words::{is_letter_char, letter_to_index, N_CHARS, N_LETTERS};

pub const B: u8 = b'B';
pub const Y: u8 = b'Y';
pub const G: u8 = b'G';

pub const B_IDX: usize = 0;
pub const Y_IDX: usize = 1;
pub const G_IDX: usize = 2;

const IDX_TO_RESP_CHAR: [u8; 3] = [B, Y, G];
const CHAR_TO_RESP_IDX: [usize; 4] = [0, Y_IDX, B_IDX, G_IDX];

pub const N_RESPONSES: usize = 243; // 3^N_CHARS = 243
const POWERS_OF_3: [usize; N_CHARS] = [1, 3, 9, 27, 81];
pub const CORRECT: [u8; N_CHARS] = [G; N_CHARS];

pub const CORRECT_IDX: usize = N_RESPONSES - 1;

/// Checks if a given byte is a response character (B, Y, G).
#[inline]
pub fn is_resp_char(c: u8) -> bool {
    (c == B) | (c == Y) | (c == G)
}

/// Converts a Wordle response [B, Y, G, ...] to a unique index 0-242.
#[inline]
pub fn response_to_index(response: &[u8; N_CHARS]) -> usize {
    let mut index = 0;
    for (i, &byte) in response.iter().enumerate() {
        let val = CHAR_TO_RESP_IDX[(byte & 3) as usize];
        index += val * POWERS_OF_3[i];
    }
    index
}

#[inline]
pub fn index_to_response(mut index: usize) -> [u8; N_CHARS] {
    let mut response = [0; N_CHARS];
    for i in 0..N_CHARS {
        let (q, r) = (index / 3, index % 3);
        index = q;
        // SAFETY: r is guaranteed to be 0, 1, or 2 by % 3
        response[i] = unsafe { *IDX_TO_RESP_CHAR.get_unchecked(r) };
    }
    response
}

/// Computes the response for a given (guess, candidate) pair.
pub fn get_resp(guess: &[u8; N_CHARS], candidate: &[u8; N_CHARS]) -> [u8; N_CHARS] {
    debug_assert!(guess.iter().all(|&c| is_letter_char(c)), "Guess must only contain letters");
    debug_assert!(candidate.iter().all(|&c| is_letter_char(c)), "Candidate must only contain letters");

    // Count frequencies in candidate
    let mut cand_counts = [0u8; N_LETTERS];
    for &letter in candidate {
        cand_counts[letter_to_index(letter)] += 1;
    }

    // Initially all black response
    let mut response = [B; N_CHARS];

    // First pass: Check greens
    for i in 0..N_CHARS {
        if guess[i] == candidate[i] {
            response[i] = G;
            cand_counts[letter_to_index(guess[i])] -= 1;
        }
    }

    // Second pass: Check yellows
    for i in 0..N_CHARS {
        if response[i] != G {
            let letter_idx = letter_to_index(guess[i]);
            if cand_counts[letter_idx] > 0 {
                response[i] = Y;
                cand_counts[letter_idx] -= 1;
            }
        }
    }

    response
}

/// Precomputes responses for all (guess, candidate) pairs.
pub fn compute_response_cache(
    guesses: &[[u8; N_CHARS]],
    candidates: &[[u8; N_CHARS]],
) -> Box<[u8]> {

    info!("Building response cache...");
    let n_guesses = guesses.len();
    let n_candidates = candidates.len();
    let element_count = n_guesses * n_candidates;
    let mut cache_data = vec![0u8; element_count];

    cache_data
        .par_chunks_mut(n_candidates) // each slice has length n_candidates (one row)
        .enumerate() // enumerate to get g_idx
        .for_each(|(g_idx, row_slice)| {
            let guess = &(guesses[g_idx]);
            for (c_idx, slot) in row_slice.iter_mut().enumerate() {
                let candidate = &candidates[c_idx];
                let resp = get_resp(guess, candidate);
                *slot = response_to_index(&resp) as u8;
            }
        });

    // Log the final cache size
    let element_size = size_of::<u8>();
    let total_bytes = element_count * element_size;
    info!("Response cache built successfully. (~{})", format_bytes(total_bytes as f32));

    // Return the fully computed cache.
    cache_data.into_boxed_slice()
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitwise_hashing_logic() {
        // Verify that our bitwise trick actually maps the characters to
        // the intended values: B=0, Y=1, G=2

        // B is ASCII 66 (01000010), ends in 10 (2). Map[2] should be 0.
        assert_eq!(CHAR_TO_RESP_IDX[(B & 3) as usize], B_IDX);

        // Y is ASCII 89 (01011001), ends in 01 (1). Map[1] should be 1.
        assert_eq!(CHAR_TO_RESP_IDX[(Y & 3) as usize], Y_IDX);

        // G is ASCII 71 (01000111), ends in 11 (3). Map[3] should be 2.
        assert_eq!(CHAR_TO_RESP_IDX[(G & 3) as usize], G_IDX);
    }

    #[test]
    fn test_min_max_values() {
        // Test Index all black
        let all_black = [B, B, B, B, B];
        assert_eq!(response_to_index(&all_black), 0);
        assert_eq!(index_to_response(0), all_black);

        // Test Index all green
        let all_green = CORRECT;
        assert_eq!(response_to_index(&all_green), CORRECT_IDX);
        assert_eq!(index_to_response(CORRECT_IDX), all_green);
    }

    #[test]
    fn test_mixed_values() {
        // Test an arbitrary pattern: B Y G B Y
        // 0*1 + 1*3 + 2*9 + 0*27 + 1*81 = 3 + 18 + 81 = 102
        let pattern = [B, Y, G, B, Y];
        let idx = response_to_index(&pattern);
        assert_eq!(idx, 102);
        assert_eq!(index_to_response(102), pattern);
    }

    #[test]
    fn test_round_trip_all() {
        // Brute force verify all 243 combinations
        for i in 0..N_RESPONSES {
            let resp = index_to_response(i);
            let idx = response_to_index(&resp);
            assert_eq!(i, idx, "Failed round trip at index {}", i);
        }
    }

    #[test]
    fn test_all_indices_produce_unique_patterns() {
        use std::collections::HashSet;

        let mut seen_patterns = HashSet::new();

        for i in 0..N_RESPONSES {
            let resp = index_to_response(i);
            let is_unique = seen_patterns.insert(resp);
            assert!(is_unique, "Duplicate pattern {:?} for index {}!", resp, i);
        }
    }

    #[test]
    fn test_all_responses_contain_valid_chars() {
        for i in 0..N_RESPONSES {
            let resp = index_to_response(i);
            for (char_idx, &c) in resp.iter().enumerate() {
                assert!(
                    is_resp_char(c),
                    "Pattern {:?} contains invalid character {} at pos {}",
                    resp, c, char_idx
                );
            }
        }
    }
}