use std::collections::{HashMap};
use std::iter::Sum;
use std::ops::Mul;
use log::info;
use crate::words::N_CHARS;

const KB: f32 = 1024.0;
const MB: f32 = 1024.0 * KB;
const GB: f32 = 1024.0 * MB;

/// Formats a byte count into a human-readable string (KB, MB, GB).
pub fn format_bytes(bytes: f32) -> String {
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

/// Precomputes a mapping from a `candidate_index` to its corresponding `guess_index`.
pub fn compute_cidx_to_gidx_map(
    all_candidates: &[[u8; N_CHARS]],
    all_guesses: &[[u8; N_CHARS]],
) -> Vec<usize> {

    info!("Building candidate to guess mapping...");
    let n_guesses = all_guesses.len();
    let n_candidates = all_candidates.len();

    // Get the mappings from guess words to guess indices.
    let mut guess_word_to_index: HashMap<&[u8; N_CHARS], usize> = HashMap::with_capacity(n_guesses);
    for (i, word) in all_guesses.iter().enumerate() {
        guess_word_to_index.insert(word, i);
    }

    // Create the mapping from candidate indices to guess indices.
    let c_idx_to_g_idx_map: Vec<usize> = (0..n_candidates)
        .map(|c_idx| {
            let candidate_word = &all_candidates[c_idx];
            *guess_word_to_index
                .get(candidate_word)
                .expect("A candidate word was not found in the guess list. Check word files.")
        })
        .collect();

    // Log the final cache size
    let total_bytes = c_idx_to_g_idx_map.len() * size_of::<usize>();
    info!("candidate to guess mapping built successfully. (~{})", format_bytes(total_bytes as f32));

    // Return the fully computed cache.
    c_idx_to_g_idx_map
}

#[inline]
pub fn sum_of_squares<T, I>(xs: I) -> T
where
    T: Mul<Output = T> + Sum + Copy,
    I: IntoIterator<Item = T>,
{
    xs.into_iter().map(|x| x * x).sum()
}
