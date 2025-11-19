use bitvec::bitvec;
use bitvec::order::Lsb0;
use bitvec::vec::BitVec;
use log::{info};
use wordle_solver::resp::compute_response_cache;
use wordle_solver::utils::{compute_cidx_to_gidx_map};
use wordle_solver::sim::{max_frequency_hardmode_policy_wrapper, max_frequency_policy_wrapper, min_remaining_hardmode_policy_wrapper, min_remaining_policy_wrapper, simulate, PolicyFn};
use wordle_solver::solver::compute_optimal_move;
use wordle_solver::words::{arr_to_word, words_to_arr, CANDIDATES, GUESSES};

fn main() {
    env_logger::init();

    // Load words and convert them to Arrays
    info!("Accessing candidate list...");
    let candidates = words_to_arr(&CANDIDATES);
    let n_candidates = candidates.len();

    info!("Accessing guess list...");
    let guesses = words_to_arr(&GUESSES);

    // Compute caches and initial candidates
    let response_cache = compute_response_cache(&guesses, &candidates);
    let initial_candidates: BitVec<u64, Lsb0> = bitvec![u64, Lsb0; 1; n_candidates];
    let c_idx_to_g_idx_map = compute_cidx_to_gidx_map(&candidates, &guesses);

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
            &guesses,
            &candidates,
            &response_cache,
            &c_idx_to_g_idx_map,
            find_guess_fn,
        );
        let avg_guesses = total_cost as f64 / n_candidates as f64;

        info!("   Total Cost: {}", total_cost);
        info!("   Average Guesses: {:.4}", avg_guesses);
    }

    info!("Running Optimal Solver (Branch & Bound)...");

    let (best_guess_idx, min_total_cost) = compute_optimal_move(
        &response_cache, &candidates, &guesses,
    );

    let best_word = arr_to_word(&guesses[best_guess_idx]);
    let avg_guesses = min_total_cost as f64 / n_candidates as f64;

    info!("Optimal Result:");
    info!("   Best Start Word: {} (Index {})", best_word, best_guess_idx);
    info!("   Minimum Total Cost: {}", min_total_cost);
    info!("   Minimum Average Guesses: {:.4}", avg_guesses);
}

