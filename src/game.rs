use std::collections::{HashMap};
use log::info;
use ndarray::{par_azip, Array2, ArrayView1, ArrayView2, Zip};

pub const N_CHARS: usize = 5;
pub const N_LETTERS: usize = 26; // A-Z

pub const N_RESPONSES: usize = 243; // 3^5 = 243

const POWERS_OF_3: [usize; N_CHARS] = [1, 3, 9, 27, 81];

pub const B: u8 = b'B';
pub const G: u8 = b'G';
pub const Y: u8 = b'Y';

pub const CORRECT: [u8; N_CHARS] = [G; N_CHARS];
const KB: f32 = 1024.0;
const MB: f32 = 1024.0 * KB;
const GB: f32 = 1024.0 * MB;

/// Converts a Wordle response [B, Y, G, ...] to a unique index 0-242.
#[inline]
pub fn response_to_index(response: &[u8; N_CHARS]) -> usize {
    let mut index = 0;
    for i in 0..N_CHARS {
        let val = match response[i] {
            B => 0,
            Y => 1,
            G => 2,
            _ => unreachable!(), // Should not happen
        };
        index += val * POWERS_OF_3[i];
    }
    index
}

/// Checks if a given byte is a letter (A-Z).
#[inline]
pub fn is_letter_char(c: u8) -> bool {
    (c >= b'A') & (c <= b'Z')
}

/// Converts a letter byte (A-Z) to an index 0-25.
#[inline]
pub fn letter_to_index(c: u8) -> usize {
    (c - b'A') as usize
}


/// Computes the response for a given (guess, candidate) pair.
pub fn get_resp(guess: ArrayView1<u8>, candidate: ArrayView1<u8>) -> [u8; N_CHARS] {
    assert_eq!(guess.len(), N_CHARS, "Guess length must be N_CHARS");
    assert_eq!(candidate.len(), N_CHARS, "Candidate length must be N_CHARS");
    assert!(guess.iter().all(|&c| is_letter_char(c)), "Guess must only contain letters");
    assert!(candidate.iter().all(|&c| is_letter_char(c)), "Candidate must only contain letters");
    let mut response = [B; N_CHARS];
    let mut cand_counts = [0u8; N_LETTERS];
    for &letter in candidate {
        cand_counts[letter_to_index(letter)] += 1;
    }

    Zip::from(&mut response)
        .and(guess)
        .and(candidate)
        .for_each(|resp_char, &g, &c| {
            if g == c {
                *resp_char = G;
                cand_counts[letter_to_index(g)] -= 1;
            }
        });

    Zip::from(&mut response)
        .and(guess)
        .for_each(|resp_char, &g| {
            // Check if this letter is not already Green
            if *resp_char != G {
                let letter_idx = letter_to_index(g);
                // Check if this letter is left in the candidate
                if cand_counts[letter_idx] > 0 {
                    *resp_char = Y; // Mark as Yellow
                    cand_counts[letter_idx] -= 1; // "Use up" this letter
                }
            }
        });

    response
}

/// Formats a byte count into a human-readable string (KB, MB, GB).
fn format_bytes(bytes: f32) -> String {
    if bytes >= GB {
        format!("{:.2} GB", bytes / GB)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes / MB)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes / KB)
    } else {
        format!("{} bytes", bytes)
    }
}

/// Precomputes responses for all (guess, candidate) pairs.
pub fn compute_response_cache(
    guesses: ArrayView2<u8>,
    candidates: ArrayView2<u8>,
) -> Array2<[u8; N_CHARS]> {

    info!("Building response cache...");
    let n_guesses = guesses.nrows();
    let n_candidates = candidates.nrows();
    assert_eq!(guesses.ncols(), candidates.ncols());

    // Initialize the main response cache.
    let mut response_cache = Array2::from_elem((n_guesses, n_candidates), [B; N_CHARS]);

    // Populate the cache in parallel.
    par_azip!((index (g_idx, c_idx), response_slot in &mut response_cache) {
        // Get the specific word for this pair
        let guess = guesses.row(g_idx);
        let candidate = candidates.row(c_idx);

        // Call our efficient get_resp directly.
        *response_slot = get_resp(guess, candidate);
    });

    // Log the final cache size
    let element_count = n_guesses * n_candidates;
    let element_size = size_of::<[u8; N_CHARS]>();
    let total_bytes = element_count * element_size;
    info!("Response cache built successfully. (~{})", format_bytes(total_bytes as f32));

    // Return the fully computed cache.
    response_cache
}

/// Precomputes a mapping from a `candidate_index` to its corresponding `guess_index`.
pub fn compute_cidx_to_gidx_map(
    all_candidates_arr: ArrayView2<u8>,
    all_guesses_arr: ArrayView2<u8>,
) -> Vec<usize> {

    info!("Building candidate to guess mapping...");
    let n_guesses = all_guesses_arr.nrows();
    let n_candidates = all_candidates_arr.nrows();

    // Get the mappings from guess words to guess indices.
    let mut guess_word_to_index: HashMap<Vec<u8>, usize> = HashMap::with_capacity(n_guesses);
    for g_idx in 0..n_guesses {
        let guess_word_vec = all_guesses_arr.row(g_idx).to_vec();
        guess_word_to_index.insert(guess_word_vec, g_idx);
    }

    // Create the mapping from candidate indices to guess indices.
    let c_idx_to_g_idx_map: Vec<usize> = (0..n_candidates)
        .map(|c_idx| {
            // Get the candidate word as a Vec<u8> to use as a key
            let candidate_word_vec = all_candidates_arr.row(c_idx).to_vec();
            *guess_word_to_index
                .get(&candidate_word_vec)
                .expect("A candidate word was not found in the guess list. Check word files.")
        })
        .collect();

    // Log the final cache size
    let total_bytes = c_idx_to_g_idx_map.len() * size_of::<usize>();
    info!("candidate to guess mapping built successfully. (~{})", format_bytes(total_bytes as f32));

    // Return the fully computed cache.
    c_idx_to_g_idx_map
}

/// Computes the lower bounds for each partition size from 0 up until n_candidates inclusive.
pub fn compute_lower_bounds(n_candidates: usize) -> Vec<usize> {
    (0..=n_candidates).map(|n| lower_bound(n)).collect()
}

/// Computes a lower bound on the optimal expected guesses for a given partition size.
#[inline]
pub fn lower_bound(n_candidates: usize) -> usize {
    if n_candidates == 0 {
        return 0;
    } 2 * n_candidates - 1
}