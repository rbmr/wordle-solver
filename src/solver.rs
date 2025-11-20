use std::cmp::{Reverse};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use bitvec::bitvec;
use bitvec::order::Lsb0;
use bitvec::vec::BitVec;
use log::info;
use crate::utils::{compute_cidx_to_gidx_map};
use crate::resp::{N_RESPONSES, CORRECT_IDX};
use crate::sim::{simulate, MIN_REMAINING_POLICY};
use crate::words::N_CHARS;

/// Computes a lower bound on the optimal expected guesses for a given partition size.
/// Only valid for n_candidates > 0
#[inline]
pub fn lower_bound(n_candidates: usize) -> usize {
    2 * n_candidates - 1
}

#[inline]
pub fn get_partition_counts(
    candidates: &BitVec<u64, Lsb0>,
    response_cache: &[u8],
    g_idx: usize,
    n_total_candidates: usize,
) -> [usize; N_RESPONSES] {
    let mut counts = [0usize; N_RESPONSES];
    for c_idx in candidates.iter_ones() {
        let resp_idx = response_cache[g_idx * n_total_candidates + c_idx] as usize;
        // SAFETY: get_unchecked is safe here per definition of response_to_index
        unsafe { *counts.get_unchecked_mut(resp_idx) += 1; }
    }
    counts
}

#[inline]
pub fn generate_partitions(
    candidates: &BitVec<u64, Lsb0>,
    response_cache: &[u8],
    g_idx: usize,
    n_total_candidates: usize,
) -> Vec<(BitVec<u64, Lsb0>, usize)> {

    // Create arrays for partitions and for counts.
    let mut partitions: [Option<BitVec<u64, Lsb0>>; N_RESPONSES] =
        std::array::from_fn(|_| None);
    let mut counts = [0usize; N_RESPONSES];

    // Iterate over all candidates to fill partitions.
    for c_idx in candidates.iter_ones() {
        let resp_idx = response_cache[g_idx * n_total_candidates + c_idx] as usize;
        if resp_idx == CORRECT_IDX { continue; } // Don't create partition for Green response.
        partitions[resp_idx]
            .get_or_insert_with(|| bitvec![u64, Lsb0; 0; candidates.len()])
            .set(c_idx, true);
        counts[resp_idx] += 1;
    }

    // Collect all created BitVecs and their corresponding counts.
    partitions
        .into_iter()
        .zip(counts)
        .filter_map(|(p_opt, count)| p_opt.map(|p| (p, count)))
        .collect()
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

type MemoCache = HashMap<BitVec<u64, Lsb0>, usize>;

struct SolverContext<'a> {
    response_cache: &'a [u8],
    memo: MemoCache,
    n_total_candidates: usize,
}

impl<'a> SolverContext<'a> {

    fn new(response_cache: &'a [u8], n_total_candidates: usize) -> Self {
        Self { response_cache, memo: HashMap::new(), n_total_candidates }
    }

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
        if let Some(&cost) = self.memo.get(candidates) {
            return cost;
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
                guess_lb += 2;
            } else if let Some(&cached_val) = self.memo.get(partition_candidates) {
                guess_lb += cached_val;
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

/// Computes the optimal guess and its corresponding minimum total expected cost
/// for the initial set of candidates.
///
/// This function serves as the parallel entry point for the Branch and Bound algorithm.
pub fn compute_optimal_move(
    response_cache: &[u8],
    all_candidates: &[[u8; N_CHARS]],
    all_guesses: &[[u8; N_CHARS]],
) -> (usize, usize) {

    // Setup
    let n_candidates = all_candidates.len();
    let n_guesses = all_guesses.len();
    info!("Starting solver for {} candidates...", n_candidates);
    let initial_candidates = bitvec![u64, Lsb0; 1; n_candidates];
    let c_idx_to_g_idx = compute_cidx_to_gidx_map(all_candidates, all_guesses);

    // Compute initial heuristic cost using the "Min Remaining" heuristic.
    let heuristic_cost = simulate(
        &initial_candidates, all_guesses, all_candidates,
        response_cache, &c_idx_to_g_idx, &MIN_REMAINING_POLICY,
    );
    info!("Initial Heuristic Upper Bound (Beta): {}", heuristic_cost);

    // We use an AtomicUsize to share the best-known cost (beta) across threads.
    let global_beta = AtomicUsize::new(heuristic_cost);
    let processed_count = AtomicUsize::new(0);

    // Sort guesses by heuristic to prioritize promising branches.
    let all_guesses: Vec<usize> = (0..n_guesses).into_iter().collect();
    let promising_guesses = filter_and_sort_guesses(
        all_guesses.as_slice(), response_cache,
        &initial_candidates, n_candidates, n_candidates,
        heuristic_cost
    );

    let best_result = promising_guesses
        .clone()
        .into_iter()
        .map(|g_idx| {

            let result = (|| {

                // Compute partitions for the guess
                let partitions = generate_partitions(
                    &initial_candidates, response_cache,
                    g_idx, n_candidates
                );

                // Get current beta
                let current_beta = global_beta.load(Ordering::Relaxed);

                // Solve exactly using local SolverContext.
                let mut solver = SolverContext::new(response_cache, n_candidates);
                let cost = solver.evaluate_guess(
                    &partitions, &promising_guesses,
                    current_beta, n_candidates
                );

                if cost < current_beta {
                    global_beta.fetch_min(cost, Ordering::Relaxed);
                }

                (g_idx, cost)
            })();

            // Update progress and log
            let finished = processed_count.fetch_add(1, Ordering::Relaxed) + 1;
            let current_beta = global_beta.load(Ordering::Relaxed);
            let pct = (finished as f64 / n_guesses as f64) * 100.0;
            info!("Progress: {:>5}/{} ({:>5.1}%) | Current Best: {}", finished, n_guesses, pct, current_beta);

            result
        })
        .min_by_key(|&(_, cost)| cost)
        .unwrap();

    info!("Optimal solution found: Guess Index {}, Total Cost {}", best_result.0, best_result.1);
    best_result
}
