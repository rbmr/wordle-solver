use bitvec::bitvec;
use bitvec::order::Lsb0;
use bitvec::vec::BitVec;
use log::{info};
use wordle_solver::game::{compute_cidx_to_gidx_map, compute_response_cache, N_CHARS};
use wordle_solver::sim::{max_frequency_hardmode_policy_wrapper, max_frequency_policy_wrapper, min_remaining_hardmode_policy_wrapper, min_remaining_policy_wrapper, simulate, PolicyFn};
use wordle_solver::solver::compute_optimal_move;
use wordle_solver::words::{arr_to_string, words_to_arr, CANDIDATES, GUESSES};

fn main() {
    env_logger::init();

    // Load words and convert them to Arrays
    info!("Accessing candidate list...");
    let candidates_arr = words_to_arr(&CANDIDATES, N_CHARS)
        .expect("Failed to convert candidates to array");
    let n_candidates = candidates_arr.nrows();

    info!("Accessing guess list...");
    let guesses_arr = words_to_arr(&GUESSES, N_CHARS)
        .expect("Failed to convert guesses to array");

    // Compute caches and initial candidates
    let response_cache = compute_response_cache(guesses_arr.view(), candidates_arr.view());
    let initial_candidates: BitVec<u64, Lsb0> = bitvec![u64, Lsb0; 1; n_candidates];
    let c_idx_to_g_idx_map = compute_cidx_to_gidx_map(
        candidates_arr.view(), guesses_arr.view(),
    );

    info!("Starting Heuristic Simulations:");

    let policies: [(&str, PolicyFn); 4] = [
        ("1. Min Remaining (Normal Mode)", min_remaining_policy_wrapper),
        ("2. Max Frequency (Normal Mode)", max_frequency_policy_wrapper),
        ("3. Min Remaining (Hard Mode)", min_remaining_hardmode_policy_wrapper),
        ("4. Max Frequency (Hard Mode)", max_frequency_hardmode_policy_wrapper),
    ];

    for (name, find_guess_fn) in policies.iter() {
        info!("-> Running simulation for {}", name);

        let total_cost = simulate(
            &initial_candidates,
            guesses_arr.view(),
            candidates_arr.view(),
            response_cache.view(),
            &c_idx_to_g_idx_map,
            find_guess_fn,
        );
        let avg_guesses = total_cost as f64 / n_candidates as f64;

        info!("   Total Cost: {}", total_cost);
        info!("   Average Guesses: {:.4}", avg_guesses);
    }

    info!("Running Optimal Solver (Branch & Bound)...");

    let (best_guess_idx, min_total_cost) = compute_optimal_move(
        response_cache.view(),
        candidates_arr.view(),
        guesses_arr.view(),
    );

    let best_word = arr_to_string(guesses_arr.row(best_guess_idx));
    let avg_guesses = min_total_cost as f64 / n_candidates as f64;

    info!("Optimal Result:");
    info!("   Best Start Word: {} (Index {})", best_word, best_guess_idx);
    info!("   Minimum Total Cost: {}", min_total_cost);
    info!("   Minimum Average Guesses: {:.4}", avg_guesses);
}

