use std::cmp::{Reverse};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use bitvec::bitvec;
use bitvec::order::Lsb0;
use bitvec::vec::BitVec;
use log::info;
use crate::cache::MemoCache;
use crate::resp::{generate_partitions, ResponseCache, CORRECT_IDX, N_RESPONSES};
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
    response_cache: &ResponseCache,
    candidates: &BitVec<u64, Lsb0>,
    n_candidates: usize,
    beta: usize,
) -> Vec<usize> {

    let mut scored_guesses: Vec<(usize, usize)> = Vec::with_capacity(guesses.len());
    let mut counts = [0usize; N_RESPONSES];
    let mut found_resp = [0usize; N_RESPONSES];
    let mut n_found_resp: usize = 0;

    for &g_idx in guesses {

        // Clear response counts
        while n_found_resp > 0 {
            let resp_idx = found_resp[n_found_resp - 1];
            n_found_resp -= 1;
            counts[resp_idx] = 0;
        }

        // Populate response counts
        let cache_row = response_cache.get_row(g_idx);
        let words = candidates.as_raw_slice();
        for (i, &word) in words.iter().enumerate() {
            if word == 0 { continue; }
            let mut w = word;
            let base_idx = i << 6; // Multiply by 64.
            while w != 0 {
                let tz = w.trailing_zeros(); // Find the bit
                w &= w - 1; // Clear bit
                let c_idx = base_idx + tz as usize;
                unsafe {
                    let resp_idx = *cache_row.get_unchecked(c_idx) as usize;
                    if *counts.get_unchecked(resp_idx) == 0 {
                        *found_resp.get_unchecked_mut(n_found_resp) = resp_idx;
                        n_found_resp += 1;
                    }
                    *counts.get_unchecked_mut(resp_idx) += 1;
                }
            }
        }

        // Compute guess score, lb, and information gain
        let mut sum_squares = 0;
        let mut guess_lb = n_candidates;

        for &resp_idx in found_resp[..n_found_resp].iter() {
            let &c = unsafe { counts.get_unchecked(resp_idx) };
            sum_squares += c * c;
            if resp_idx != CORRECT_IDX {
                guess_lb += lower_bound(c);
            }
        }

        // Skip guesses with no information gain
        if n_found_resp <= 1 {
            continue;
        }

        // Skip guesses that dont beat beta
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
    response_cache: &'a ResponseCache,
    memo: &'a MemoCache,
}

impl<'a> SolverContext<'a> {

    /// Computes the minimum total cost for the candidate set if it is <= beta,
    /// otherwise returns usize::MAX.
    fn evaluate_candidates(
        &self,
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
            guesses, self.response_cache, candidates, n_candidates, beta
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
        candidates: &BitVec<u64, Lsb0>,
        next_guesses: &[usize],
        beta: usize,
        n_candidates: usize,
    ) -> usize {

        // Generate all partitions for the current candidates
        let partitions = generate_partitions(g_idx, &candidates, self.response_cache);

        // Calculate initial lower bound for the guess
        let mut guess_lb = n_candidates;
        let mut unresolved_partitions = Vec::with_capacity(partitions.len());

        for (partition_candidates, partition_size) in partitions {

            // Update lower bound using this partition's size
            if partition_size == 1 {
                guess_lb += 1;
            } else if partition_size == 2 {
                guess_lb += 3;
            } else if let Some(cached_val) = self.memo.get(&partition_candidates) {
                guess_lb += *cached_val;
            } else {
                unresolved_partitions.push((partition_candidates, partition_size));
                guess_lb += lower_bound(partition_size)
            }

            if guess_lb > beta { return usize::MAX; }
        }

        // Sort the unresolved partitions by descending size to fail fast.
        unresolved_partitions.sort_unstable_by_key(|(_, n)| Reverse(*n));

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
    response_cache: &ResponseCache,
    memo: &MemoCache,
) -> (usize, usize) {

    // Setup
    let n_total_candidates = all_candidates.len();
    let n_total_guesses = all_guesses.len();
    info!("Starting solver for {} candidates...", n_total_candidates);
    let initial_candidates = bitvec![u64, Lsb0; 1; n_total_candidates];

    // Compute initial heuristic cost using the "Min Remaining" heuristic.
    let policy = MinRemainingPolicy { response_cache, n_total_candidates, n_total_guesses };
    let heuristic_cost = simulate(response_cache, &initial_candidates, &policy).total_guesses();
    info!("Initial Heuristic Upper Bound (Beta): {}", heuristic_cost);

    // Sort guesses by heuristic to prioritize promising branches.
    let solver = SolverContext { response_cache, memo };
    let all_guesses: Vec<usize> = (0..n_total_guesses).into_iter().collect();
    let promising_guesses = filter_and_sort_guesses(
        &all_guesses, response_cache, &initial_candidates, n_total_candidates, heuristic_cost
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
