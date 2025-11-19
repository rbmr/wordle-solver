use rayon::iter::ParallelIterator;
use std::cmp::{Reverse};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use bitvec::bitvec;
use bitvec::order::Lsb0;
use bitvec::vec::BitVec;
use log::info;
use rayon::iter::IntoParallelIterator;
use crate::utils::{compute_cidx_to_gidx_map};
use crate::resp::{N_RESPONSES};
use crate::score::get_min_remaining_score;
use crate::sim::{simulate, MIN_REMAINING_POLICY};
use crate::words::N_CHARS;

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

pub fn compute_partitions_and_counts(
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
        partitions[resp_idx]
            .get_or_insert_with(|| bitvec![u64, Lsb0; 0; candidates.len()])
            .set(c_idx, true);
        counts[resp_idx] += 1;
    }

    // Collect all created BitVecs and their corresponding counts.
    partitions
        .into_iter()
        .zip(counts)
        .filter_map(|(p_option, count)| {
            // Only yield if the partition was actually created (p_option is Some)
            p_option.map(|p| (p, count))
        })
        .collect()
}

type MemoCache = HashMap<BitVec<u64, Lsb0>, usize>;

struct ScoredPartitions {
    score: usize,
    partitions: Vec<(BitVec<u64, Lsb0>, usize)>,
}


struct SolverContext<'a> {
    response_cache: &'a [u8],
    memo: MemoCache,
    n_candidates_total: usize,
}

impl<'a> SolverContext<'a> {

    fn new(response_cache: &'a [u8], n_candidates_total: usize) -> Self {
        Self { response_cache, memo: HashMap::new(), n_candidates_total }
    }

    /// Computes the minimum total cost for the given guess if it is < beta,
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
        let mut missing_partitions = Vec::with_capacity(partitions.len());
        for (partition_candidates, partition_size) in partitions {
            if let Some(&cached_val) = self.memo.get(partition_candidates) {
                guess_lb += cached_val;
            } else {
                missing_partitions.push((partition_candidates, partition_size));
                guess_lb += lower_bound(*partition_size)
            }
            if guess_lb > beta {
                return usize::MAX;
            }
        }

        // Sort the missing partitions by descending size.
        missing_partitions.sort_unstable_by_key(|(_, n)| Reverse(*n));

        // Recurse, tightening lower bound.
        for (partition_candidates, partition_size) in missing_partitions {
            let p_lb = lower_bound(*partition_size);
            let p_cost = self.evaluate_candidates(
                &partition_candidates,
                &next_guesses,
                beta - (guess_lb - p_lb)
            );
            if p_cost == usize::MAX {
                return usize::MAX;
            }
            guess_lb += p_cost - p_lb;
        }
        guess_lb
    }

    /// Computes the minimum total cost for the candidate set if it is < beta,
    /// otherwise returns usize::MAX.
    fn evaluate_candidates(
        &mut self,
        candidates: &BitVec<u64, Lsb0>,
        guesses: &[usize],
        mut beta: usize
    ) -> usize {
        let n_candidates = candidates.count_ones();

        // Base Cases
        assert!(n_candidates > 0, "Cannot solve empty candidate set");
        if n_candidates == 1 { return 1; }
        if n_candidates == 2 { return 3; }

        // Memoization Check
        if let Some(&cost) = self.memo.get(candidates) {
            return cost;
        }

        // Lower Bound Check
        let lb = lower_bound(n_candidates);
        if lb > beta {
            return usize::MAX;
        }

        // Preprocess guesses
        let mut valid_moves: Vec<ScoredPartitions> = Vec::with_capacity(guesses.len());
        let mut next_guesses: Vec<usize> = Vec::with_capacity(guesses.len());
        for &g_idx in guesses {
            let partitions = compute_partitions_and_counts(
                candidates,
                self.response_cache,
                g_idx,
                self.n_candidates_total,
            );
            if partitions.len() <= 1 {
                continue; // No additional information. Skip.
            }

            let mut score: usize = 0;
            for (_, count) in &partitions {
                score += count * count;
            }

            valid_moves.push(ScoredPartitions { score, partitions });
            next_guesses.push(g_idx);
        }

        // Sort the guesses based on score
        valid_moves.sort_unstable_by_key(|m| m.score);

        // Iterate over all valid moves and recurse.
        let mut lowest_guess_cost = usize::MAX;
        for m in valid_moves {
            let ScoredPartitions { partitions, .. } = m;
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
        &initial_candidates,
        all_guesses,
        all_candidates,
        response_cache,
        &c_idx_to_g_idx,
        &MIN_REMAINING_POLICY,
    );
    info!("Initial Heuristic Upper Bound (Beta): {}", heuristic_cost);

    // We use an AtomicUsize to share the best-known cost (beta) across threads.
    let global_beta = AtomicUsize::new(heuristic_cost);
    let processed_count = AtomicUsize::new(0);

    // Sort guesses by heuristic to prioritize promising branches.
    let mut sorted_guesses: Vec<(usize, usize)> = (0..n_guesses)
        .into_par_iter()
        .map(|g_idx| {
            let score = get_min_remaining_score(
                &initial_candidates,
                response_cache,
                g_idx,
                n_candidates,
            );
            (g_idx, score)
        })
        .collect();
    sorted_guesses.sort_unstable_by_key(|&(_, score)| score);

    let guesses: Vec<usize> = sorted_guesses.iter().map(|&(g_idx, _)| g_idx).collect();
    let n_guesses = guesses.len();

    let best_result = sorted_guesses
        .into_iter()
        .map(|(g_idx, _)| {

            let result = (|| {
                // Compute partitions for the guess
                let partitions_n_counts = compute_partitions_and_counts(
                    &initial_candidates,
                    response_cache,
                    g_idx,
                    n_candidates
                );
                if partitions_n_counts.len() <= 1 {
                    // This guess provides no info.
                    return (g_idx, usize::MAX);
                }

                // Get initial lower bound
                let mut lb = n_candidates;
                for (_, count) in &partitions_n_counts {
                    lb += lower_bound(*count);
                }

                // Check against global beta.
                let current_beta = global_beta.load(Ordering::Relaxed);
                if lb >= current_beta {
                    return (g_idx, usize::MAX);
                }

                // Solve exactly using local SolverContext.
                let mut solver = SolverContext::new(response_cache, n_candidates);
                let cost = solver.evaluate_guess(
                    &partitions_n_counts,
                    &guesses,
                    current_beta,
                    n_candidates
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
