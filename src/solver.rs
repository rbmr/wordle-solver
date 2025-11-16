use rayon::iter::{IntoParallelIterator, ParallelIterator};
use ndarray::ArrayView2;
use bitvec::prelude::{bitvec, BitVec, Lsb0};
use log::{info, warn};
use rayon::iter::IntoParallelRefIterator;
use crate::db::{generate_db_name, CacheManager};
use crate::game::{CORRECT, N_CHARS, compute_cidx_to_gidx_map, compute_partitions, Partition};
use crate::score::{score_min_remaining};

pub const LOG_3_OF_2: f64 = 0.63092975357; // log3(2) = 1.0 / log2(3)

pub const GAMMA: f64 = LOG_3_OF_2 / (N_CHARS as f64); // as specified in the readme

fn compute_lower_bound(
    n_candidates: usize,
    partitions: &Vec<Partition>,
) -> f64 {
    let mut sum = 0.0;
    for partition in partitions {
        // all green responses don't add to the sum E(C_{gr}) = 0
        if partition.response == CORRECT {
            continue;
        }
        let size = partition.candidates.count_ones() as f64;
        if size > 0.0 {
            // n * log2(n)
            sum += size * size.log2();
        } else {
            warn!("Unexpected partition size {}, ignoring.", size);
        }
    }
    // 1 + (γ / |C|) * Σ(|C_gr| * log2(|C_gr|))
    1.0 + (GAMMA / n_candidates as f64) * sum
}

fn find_min_expected_guesses(
    candidate_set: &BitVec<u64>,
    memo_cache: &CacheManager<BitVec<u64>, (f64, usize)>,
    response_cache: ArrayView2<[u8; N_CHARS]>,
    c_idx_to_g_idx_map: &Vec<usize>,
    n_guesses: usize,
) -> (f64, usize) {
    // Check memoization cache
    if let Some(solution) = memo_cache.get(candidate_set) {
        return solution;
    }

    // Base Cases
    let n_candidates = candidate_set.count_ones();
    if n_candidates == 1 {
        // If |C| = 1, we know the answer. Cost is 1.
        let c_idx = candidate_set.iter_ones().next().unwrap();
        let g_idx = c_idx_to_g_idx_map[c_idx];
        let solution = (1.0, g_idx);
        memo_cache.insert(candidate_set, &solution);
        return solution;
    }
    if n_candidates == 2 {
        // If |C| = 2, optimal to guess one. Expected cost is 1.5
        let c_idx = candidate_set.iter_ones().next().unwrap();
        let g_idx = c_idx_to_g_idx_map[c_idx];
        let solution = (1.5, g_idx);
        memo_cache.insert(candidate_set, &solution);
        return solution;
    }


    // Compute score and partitions for all guesses in parallel
    let mut scored_guesses: Vec<(usize, usize, Vec<Partition>)> = (0..n_guesses)
        .into_par_iter()
        .map(|g_idx| {
            let partitions = compute_partitions(candidate_set, response_cache, g_idx);
            (g_idx, score_min_remaining(&partitions), partitions)
        })
        .collect();

    scored_guesses.sort_unstable_by_key(|&(_, score, _)| score);

    // Recursive Step: Find the optimal guess g*
    let mut min_so_far = f64::INFINITY;
    let mut best_guess_idx = 0; // Will be overwritten by the first valid guess

    // Iterate over *all* possible guesses in G
    for (g_idx, _, partitions) in scored_guesses {

        // Pruning Step (using Lower Bound)
        let lower_bound_for_g = compute_lower_bound(n_candidates, &partitions);
        if lower_bound_for_g >= min_so_far {
            continue; // Prune this guess; it can't be better than our current best
        }

        // Calculate the true expected value E(C, g)
        let sum_of_weighted_futures: f64 = partitions
            .par_iter()
            .filter_map(|partition| {
                if partition.response == CORRECT {
                    None
                } else {
                    Some(partition)
                }
            })
            .map(|partition| {
                let partition_size = partition.candidates.count_ones();
                if partition_size == n_candidates {
                    return f64::INFINITY;
                }

                let (expected_future_guesses, _) = find_min_expected_guesses(
                    &partition.candidates,
                    memo_cache,
                    response_cache,
                    c_idx_to_g_idx_map,
                    n_guesses,
                );
                expected_future_guesses * (partition_size as f64)
            })
            .sum();
        let current_expected_value = 1.0 + sum_of_weighted_futures / (n_candidates as f64);

        // Update minimum
        // Check if this guess 'g' is the best one we've seen
        if current_expected_value < min_so_far {
            min_so_far = current_expected_value;
            best_guess_idx = g_idx;
        }
    }

    // Store in cache and return
    let solution = (min_so_far, best_guess_idx);
    memo_cache.insert(candidate_set, &solution);
    solution
}


pub fn compute_optimal_strategy(
    response_cache: ArrayView2<[u8; N_CHARS]>,
    all_candidates_arr: ArrayView2<u8>,
    all_guesses_arr: ArrayView2<u8>,
) -> (usize, f64) {
    info!("Starting optimal strategy computation...");
    let n_candidates = all_candidates_arr.nrows();
    let n_guesses = all_guesses_arr.nrows();

    // Set up the Database (Memoization Cache)
    let db_name = generate_db_name(all_candidates_arr, all_guesses_arr);
    let memo_cache: CacheManager<BitVec<u64>, (f64, usize)> = CacheManager::new(&db_name);
    info!("Opened memoization cache with {} existing entries.", memo_cache.len());

    // Build the c_idx -> g_idx map, used for base cases.
    let c_idx_to_g_idx_map = compute_cidx_to_gidx_map(all_candidates_arr, all_guesses_arr);

    // Create the initial candidate set (all candidates are possible)
    let initial_candidate_set = bitvec![u64, Lsb0; 1; n_candidates];

    // Call the core recursive function
    info!("Starting recursive search for optimal strategy...");
    let (min_expected_guesses, best_guess_idx) = find_min_expected_guesses(
        &initial_candidate_set,
        &memo_cache,
        response_cache,
        &c_idx_to_g_idx_map,
        n_guesses,
    );

    info!("Computation complete. Flushing cache to disk...");
    memo_cache.flush();
    info!("Optimal first guess index: {}, Expected guesses: {:.4}", best_guess_idx, min_expected_guesses);

    // Return (best_guess_idx, min_expected_guesses)
    (best_guess_idx, min_expected_guesses)
}