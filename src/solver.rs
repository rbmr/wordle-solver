use std::cmp::{Reverse};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use log::info;
use crate::bits::BitSet;
use crate::cache::MemoCache;
use crate::resp::{get_lazy_partitions, PartitionMap, CORRECT_IDX};
use crate::sim::{simulate, MinRemainingPolicy};
use crate::words::N_CHARS;

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
    partition_map: &PartitionMap,
    candidates: &BitSet,
    n_candidates: usize,
    beta: usize,
) -> Vec<usize> {

    let mut scored_guesses: Vec<(usize, usize)> = Vec::with_capacity(guesses.len());

    for &g_idx in guesses {

        // SAFETY: g_idx is bounded by guesses generation, which matches partition_map size
        let row = unsafe { partition_map.get_unchecked(g_idx) };

        // Compute guess score, lb, and information gain
        let mut sum_squares = 0;
        let mut guess_lb = n_candidates;
        let mut non_empty_partitions: usize = 0;
        for (resp_idx, mask) in row {
            let count = mask.intersection_count(candidates);
            if count > 0 {
                non_empty_partitions += 1;
                sum_squares += count * count;
                if *resp_idx != CORRECT_IDX {
                    guess_lb += lower_bound(count);
                }
            }
        }

        // Skip guesses that don't provide any information
        if non_empty_partitions <= 1 { continue; }

        // Skip guesses that dont beat beta
        if guess_lb >= beta { continue; }

        scored_guesses.push((g_idx, sum_squares));
    }

    // Sort the guesses based on score
    scored_guesses.sort_unstable_by_key(|(_, score)| *score);

    // Create guesses to pass on to child nodes.
    // Only guesses that partition current candidates, may partition subsets of candidates.
    scored_guesses.iter().map(|(g_idx, _)| *g_idx).collect()
}


struct SolverContext<'a> {
    partition_map: &'a PartitionMap,
    memo: &'a MemoCache,
}

impl<'a> SolverContext<'a> {

    /// Computes the minimum total cost for the candidate set if it is <= beta,
    /// otherwise returns usize::MAX.
    fn evaluate_candidates(
        &self,
        candidates: &BitSet,
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
            guesses, self.partition_map, candidates, n_candidates, beta
        );

        // Iterate over all reasonable moves and recurse.
        let mut lowest_guess_cost = usize::MAX;
        for g_idx in &next_guesses {

            // Calculate exact cost for this guess
            let guess_cost = self.evaluate_guess(*g_idx, candidates, &next_guesses, beta, n_candidates);
            if guess_cost < beta {
                beta = guess_cost;
                lowest_guess_cost = guess_cost;
            }
        }

        if lowest_guess_cost != usize::MAX {
            self.memo.insert(candidates.clone(), lowest_guess_cost);
        }

        lowest_guess_cost
    }

    /// Computes the minimum total cost for the given guess if it is <= beta,
    /// otherwise returns usize::MAX.
    fn evaluate_guess(
        &self,
        g_idx: usize,
        candidates: &BitSet,
        next_guesses: &[usize],
        beta: usize,
        n_candidates: usize,
    ) -> usize {

        // Calculate initial lower bound for the guess
        let mut guess_lb = n_candidates;
        let partitions = get_lazy_partitions(g_idx, &candidates, self.partition_map);
        let mut unresolved_partitions = Vec::with_capacity(partitions.len());

        for (resp_idx, p_cand, p_size) in partitions {
            if resp_idx == CORRECT_IDX { continue; }

            // Update lower bound using this partition's size
            if p_size == 1 {
                guess_lb += 1;
            } else if p_size == 2 {
                guess_lb += 3;
            } else {
                // p_size > 2, so p_cand is guaranteed to be Some
                let p_cand = p_cand.unwrap();
                if let Some(cached_val) = self.memo.get(&p_cand) {
                    guess_lb += *cached_val;
                } else {
                    unresolved_partitions.push((p_cand, p_size));
                    guess_lb += lower_bound(p_size)
                }
            }

            if guess_lb > beta { return usize::MAX; }
        }

        // Recurse, tightening lower bound.
        for (partition_candidates, partition_size) in unresolved_partitions {
            let p_lb = lower_bound(partition_size);
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
    all_candidates: &[[u8; N_CHARS]],
    all_guesses: &[[u8; N_CHARS]],
    partition_map: &PartitionMap,
    memo: &MemoCache,
) -> (usize, usize) {

    // Setup
    let n_total_candidates = all_candidates.len();
    let n_total_guesses = all_guesses.len();
    info!("Starting solver for {} candidates...", n_total_candidates);
    let initial_candidates = BitSet::ones(n_total_candidates);

    // Compute initial heuristic cost using the "Min Remaining" heuristic.
    let policy = MinRemainingPolicy { partition_map, n_total_candidates, n_total_guesses };
    let heuristic_cost = simulate(partition_map, &initial_candidates, &policy).total_guesses();
    info!("Initial Heuristic Upper Bound (Beta): {}", heuristic_cost);

    // Sort guesses by heuristic to prioritize promising branches.
    let solver = SolverContext { partition_map, memo };
    let all_guesses: Vec<usize> = (0..n_total_guesses).into_iter().collect();
    let promising_guesses = filter_and_sort_guesses(
        &all_guesses, partition_map, &initial_candidates, n_total_candidates, heuristic_cost
    );
    let total_tasks = promising_guesses.len();
    info!("Sorted {} promising guesses.", total_tasks);

    // Setup progress bar
    info!("Starting parallel guess evaluation...");
    let queue_cursor = AtomicUsize::new(0);
    let progress = AtomicUsize::new(0);
    let best_guess = BestGuess::new(heuristic_cost);

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
                    let current_beta = best_guess.get_beta();
                    let cost = solver.evaluate_guess(
                        g_idx,  &initial_candidates, &promising_guesses,
                        current_beta, n_total_candidates
                    );

                    // Handle result
                    if cost < current_beta {
                        best_guess.update(g_idx, cost);
                    }

                    // Handle logging
                    let finished = progress.fetch_add(1, Ordering::Relaxed) + 1;
                    let pct = finished as f64 / total_tasks as f64 * 100.0;
                    info!("Progress {}/{} ({:.2}%) | Guess {}, Cost {}, Best {}",
                        finished, total_tasks, pct, g_idx, cost, best_guess.get_beta()
                    );
                }
            })
        };
    });
    let best_result = best_guess.unwrap();
    info!("Optimal solution found: Guess Index {}, Total Cost {}", best_result.0, best_result.1);
    best_result
}
