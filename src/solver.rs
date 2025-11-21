use std::cmp::{Reverse};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use bitvec::bitvec;
use bitvec::order::Lsb0;
use bitvec::vec::BitVec;
use log::info;
use crate::sim::simulate_strategy;
use crate::cache::{compute_context_hash, MemoCache};
use crate::utils::format_bytes;
use crate::words::N_CHARS;
use crate::build::StrategyBuilder;
use crate::part::{generate_partitions, get_partition_counts};
use crate::policy::MinRemainingPolicy;

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

/// Computes a lower bound on the optimal expected guesses for a given partition size.
/// Only valid for n_candidates > 0
#[inline]
pub fn lower_bound(n_candidates: usize) -> usize {
    2 * n_candidates - 1
}

/// Filter out guesses that don't provide information, or beat beta.
/// Then returns these guesses sorted by min_remaining score.
#[inline]
pub fn filter_and_sort_guesses(
    guesses: &[usize],
    response_cache: &[u8],
    candidates: &BitVec<u64, Lsb0>,
    n_candidates: usize,
    n_candidates_total: usize,
    beta: usize,
) -> Vec<usize> {

    let mut scored_guesses: Vec<(usize, usize)> = Vec::with_capacity(guesses.len());
    for &g_idx in guesses {

        // Get partition counts
        let counts = get_partition_counts(
            candidates, response_cache,
            g_idx, n_candidates_total,
        );

        // Compute guess score, lb, and information gain
        let mut guess_lb = n_candidates;
        let mut sum_squares = 0;
        let mut non_zero_buckets = 0;
        for c in counts {
            if c > 0 {
                guess_lb += lower_bound(c);
                sum_squares += c * c;
                non_zero_buckets += 1;
            }
        }

        // Skip guesses with no information gain
        if non_zero_buckets <= 1 {
            continue;
        }

        // Skip guesses that will not beat beta
        if guess_lb >= beta {
            continue;
        }

        scored_guesses.push((g_idx, sum_squares));
    }

    // Sort the guesses based on score
    scored_guesses.sort_unstable_by_key(|(_, score)| *score);

    // Create guesses to pass on to child nodes.
    // Only guesses that partition current candidates, may partition subsets of candidates.
    scored_guesses.iter().map(|(g_idx, _)| *g_idx).collect()
}


struct SolverContext<'a> {
    response_cache: &'a [u8],
    memo: &'a MemoCache,
    n_total_candidates: usize,
}

impl<'a> SolverContext<'a> {

    /// Computes the minimum total cost for the candidate set if it is <= beta,
    /// otherwise returns usize::MAX.
    fn evaluate_candidates(
        &mut self,
        candidates: &BitVec<u64, Lsb0>,
        guesses: &[usize],
        mut beta: usize
    ) -> usize {
        let n_candidates = candidates.count_ones();

        // Base Cases
        if n_candidates == 1 { return 1; }
        if n_candidates == 2 { return 3; }

        // Memoization Check
        if let Some(cost) = self.memo.get(candidates) {
            return *cost;
        }

        // Lower Bound Check
        if lower_bound(n_candidates) > beta {
            return usize::MAX;
        }

        // Filter out guesses that don't provide information, or beat beta, sorted by score.
        let next_guesses = filter_and_sort_guesses(
            guesses, self.response_cache,
            candidates, n_candidates,
            self.n_total_candidates, beta
        );

        // Iterate over all reasonable moves and recurse.
        let mut lowest_guess_cost = usize::MAX;
        for &g_idx in &next_guesses {

            // Generate actual BitVec partitions
            let partitions = generate_partitions(
                candidates, self.response_cache,
                g_idx, self.n_total_candidates,
            );

            // Calculate exact cost for this guess
            let guess_cost = self.evaluate_guess(&partitions, &next_guesses, beta, n_candidates);
            if guess_cost < beta {
                beta = guess_cost;
                lowest_guess_cost = guess_cost;
            }
        }

        if lowest_guess_cost != usize::MAX {
            // We computed the true minimal cost of at least one guess.
            self.memo.insert(candidates.clone(), beta);
        }

        lowest_guess_cost
    }

    /// Computes the minimum total cost for the given guess if it is <= beta,
    /// otherwise returns usize::MAX.
    fn evaluate_guess(
        &mut self,
        partitions: &Vec<(BitVec<u64, Lsb0>, usize)>,
        next_guesses: &[usize],
        beta: usize,
        n_candidates: usize,
    ) -> usize {

        // Calculate initial lower bound for the guess
        let mut guess_lb = n_candidates;
        let mut unresolved_partitions = Vec::with_capacity(partitions.len());

        for (partition_candidates, partition_size) in partitions {

            // Update lower bound using this partition's size
            if *partition_size == 1 {
                guess_lb += 1;
            } else if *partition_size == 2 {
                guess_lb += 3;
            } else if let Some(cached_val) = self.memo.get(partition_candidates) {
                guess_lb += *cached_val;
            } else {
                unresolved_partitions.push((partition_candidates, partition_size));
                guess_lb += lower_bound(*partition_size)
            }

            if guess_lb > beta { return usize::MAX; }
        }

        // Sort the unresolved partitions by descending size to fail fast.
        unresolved_partitions.sort_unstable_by_key(|(_, n)| Reverse(*n));

        // Recurse, tightening lower bound.
        for (partition_candidates, partition_size) in unresolved_partitions {
            let p_lb = lower_bound(*partition_size);
            let p_cost = self.evaluate_candidates(
                &partition_candidates,
                &next_guesses,
                beta - (guess_lb - p_lb) // max cost for child
            );
            if p_cost == usize::MAX {
                return usize::MAX;
            }
            // p_cost != usize::MAX
            // guarantees guess_lb + (p_cost - p_lb) <= beta, because of max cost for child
            guess_lb += p_cost - p_lb;
        }
        guess_lb
    }
}


/// Simple struct to manage cross-thread best guess state.
struct BestGuess {
    beta: AtomicUsize,
    solution: Mutex<(usize, usize)>,
}

impl BestGuess {
    fn new(heuristic_cost: usize) -> Self {
        Self {
            beta: AtomicUsize::new(heuristic_cost+1),
            solution: Mutex::new((usize::MAX, usize::MAX)),
        }
    }

    /// Fast check for the worker loops.
    #[inline(always)]
    fn get_beta(&self) -> usize {
        self.beta.load(Ordering::Relaxed)
    }

    /// Thread-safe update.
    /// Updates BOTH the atomic gatekeeper and the storage mutex.
    fn update(&self, guess_idx: usize, cost: usize) {
        self.beta.fetch_min(cost, Ordering::Relaxed);
        let mut guard = self.solution.lock().unwrap();
        if cost < guard.1 {
            *guard = (guess_idx, cost);
        }
    }

    /// Extract the final result.
    fn unwrap(self) -> (usize, usize) {
        self.solution.into_inner().unwrap()
    }
}

/// Computes the optimal guess and its corresponding minimum total expected cost
/// for the initial set of candidates.
///
/// This function serves as the parallel entry point for the Branch and Bound algorithm.
pub fn compute_optimal_move(
    response_cache: &[u8],
    all_candidates: &[[u8; N_CHARS]],
    all_guesses: &[[u8; N_CHARS]],
    memo: &MemoCache,
) -> (usize, usize) {

    // Setup
    let n_total_candidates = all_candidates.len();
    let n_total_guesses = all_guesses.len();
    info!("Starting solver for {} candidates...", n_total_candidates);
    let initial_candidates = bitvec![u64, Lsb0; 1; n_total_candidates];
    let c_idx_to_g_idx = compute_cidx_to_gidx_map(all_candidates, all_guesses);

    // Compute initial heuristic cost using the "Min Remaining" heuristic.
    let policy = MinRemainingPolicy { response_cache, n_total_candidates, n_total_guesses };
    let context_hash = compute_context_hash(all_guesses, all_candidates);
    let strat = StrategyBuilder::new(
        response_cache, n_total_candidates,
        n_total_guesses, policy
    ).build(&initial_candidates, context_hash);
    let heuristic_cost = simulate_strategy(&strat, all_candidates, all_guesses).total_guesses();
    info!("Initial Heuristic Upper Bound (Beta): {}", heuristic_cost);

    // Sort guesses by heuristic to prioritize promising branches.
    let all_guesses: Vec<usize> = (0..n_total_guesses).into_iter().collect();
    let promising_guesses = filter_and_sort_guesses(
        all_guesses.as_slice(), response_cache,
        &initial_candidates, n_total_candidates, n_total_candidates,
        heuristic_cost + 1
    );
    let total_tasks = promising_guesses.len();
    info!("Sorted {} promising guesses.", total_tasks);

    // Setup cross-thread shared variables.
    let processed_count = AtomicUsize::new(0);
    let queue_cursor = AtomicUsize::new(0);
    let best_guess = BestGuess::new(heuristic_cost);
    info!("Starting parallel guess evaluation...");

    rayon::scope(|s| {
        let num_threads = rayon::current_num_threads();
        for _ in 0..num_threads {
            s.spawn(|_| {
                loop {
                    // Fetch the next item from the queue.
                    let idx = queue_cursor.fetch_add(1, Ordering::Relaxed);
                    if idx >= total_tasks { break; }
                    let g_idx = promising_guesses[idx];

                    // Process item
                    let partitions = generate_partitions(
                        &initial_candidates, response_cache,
                        g_idx, n_total_candidates
                    );
                    let mut solver = SolverContext { response_cache, memo, n_total_candidates };
                    let current_beta = best_guess.get_beta();
                    let cost = solver.evaluate_guess(
                        &partitions, &promising_guesses,
                        current_beta, n_total_candidates
                    );

                    // Handle result
                    if cost < current_beta {
                        best_guess.update(g_idx, cost);
                    }

                    // Logging
                    let finished = processed_count.fetch_add(1, Ordering::Relaxed) + 1;
                    let current_beta = best_guess.get_beta();
                    let pct = (finished as f64 / total_tasks as f64) * 100.0;
                    info!("Progress: {:>5}/{} ({:>4.1}%) | Current Best: {} | Guess: {}, Cost: {}",
                        finished, total_tasks, pct, current_beta, g_idx, cost
                    );
                }
            })
        };
    });

    let best_result = best_guess.unwrap();
    memo.insert(initial_candidates.clone(), best_result.1);

    info!("Optimal solution found: Guess Index {}, Total Cost {}", best_result.0, best_result.1);
    best_result
}
