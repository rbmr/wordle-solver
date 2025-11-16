use std::collections::HashSet;
use log::{debug, info};
use once_cell::sync::Lazy;
use wordle_solver::{words_to_arr, parse_words, N_CHARS, compute_response_cache, compute_optimal_strategy, arr_to_string};

/// The default list of candidate words.
pub static CANDIDATES: Lazy<HashSet<String>> = Lazy::new(|| {
    debug!("One-time parse: Loading default candidates...");
    let file_contents = include_str!("../words/candidates.txt");
    parse_words(file_contents, N_CHARS)
});

/// The default list of words to guess from.
pub static GUESSES: Lazy<HashSet<String>> = Lazy::new(|| {
    debug!("One-time parse: Loading default guesses...");
    let file_contents = include_str!("../words/guesses.txt");
    parse_words(file_contents, N_CHARS)
});

fn main() {
    env_logger::init();

    // Load words and convert them to ndarray Arrays
    info!("Accessing candidate list...");
    let candidates_arr = words_to_arr(&CANDIDATES, N_CHARS)
        .expect("Failed to convert candidates to array");

    info!("Accessing guess list...");
    let guesses_arr = words_to_arr(&GUESSES, N_CHARS)
        .expect("Failed to convert guesses to array");

    // Compute the response cache
    let response_cache = compute_response_cache(guesses_arr.view(), candidates_arr.view());

    info!("Starting optimal strategy computation... (this will take a long time)");

    let (optimal_idx, exp_guesses) = compute_optimal_strategy(
        response_cache.view(),
        candidates_arr.view(),
        guesses_arr.view(),
    );

    info!("Optimal Strategy Computation Complete");

    let optimal_guess = arr_to_string(guesses_arr.row(optimal_idx));
    info!("Optimal Strategy: Guess {} (expected guesses: {})", optimal_guess, exp_guesses);
}