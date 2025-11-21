use std::collections::BTreeMap;
use bitvec::prelude::*;
use crate::cache::MemoCache;
use crate::resp::{generate_all_partitions, ResponseCache, CORRECT_IDX};
use crate::strat::{pick_max_freq, pick_min_remaining, pick_optimal};
use crate::words::N_CHARS;

pub struct SimStats {
    pub total_words: usize,
    // Map<NumGuesses, CountOfWords>
    pub distribution: BTreeMap<usize, usize>,
}

impl SimStats {
    pub fn new() -> Self {
        Self {
            total_words: 0,
            distribution: BTreeMap::new(),
        }
    }

    pub fn inc(&mut self, guesses: usize) {
        self.total_words += 1;
        *self.distribution.entry(guesses).or_insert(0) += 1;
    }

    pub fn total_guesses(&self) -> usize {
        self.distribution.iter().map(|(k, v)| k * v).sum()
    }

    pub fn mean(&self) -> f64 {
        self.total_guesses() as f64 / self.total_words as f64
    }

    pub fn variance(&self, mean: f64) -> f64 {
        let sum_sq_diff: f64 = self.distribution.iter()
            .map(|(&k, &v)| {
                let diff = k as f64 - mean;
                (diff * diff) * v as f64
            })
            .sum();
        sum_sq_diff / self.total_words as f64
    }

    pub fn min_max(&self) -> (usize, usize) {
        let min = *self.distribution.keys().min().unwrap_or(&0);
        let max = *self.distribution.keys().max().unwrap_or(&0);
        (min, max)
    }
}

impl<'a> Simulation<'a> {

    pub fn run(
        &mut self,
        candidates: &BitVec<u64, Lsb0>,
        current_depth: usize,
    ) {
        let n_candidates = candidates.count_ones();

        // Base Cases
        assert!(n_candidates > 0);
        if n_candidates == 1 {
            self.stats.inc(current_depth);
            return;
        }
        if n_candidates == 2 {
            self.stats.inc(current_depth);
            self.stats.inc(current_depth + 1);
            return;
        }

        // Check if the guess is in the candidate set
        let guess_idx = self.policy.pick(candidates);

        // Partition the candidates based on the chosen guess g
        let partitions = generate_all_partitions(guess_idx, candidates, self.response_cache);
        assert!(partitions.len() > 1, "Partitioning failed, policy is broken.");

        // Recurse.
        for (resp_idx, (partition_candidates, _)) in partitions {
            if resp_idx == CORRECT_IDX {
                self.stats.inc(current_depth);
                continue;
            }
            self.run(&partition_candidates, current_depth + 1);
        }
    }
}

/// Context to hold shared data for the recursive heuristic simulation.
struct Simulation<'a> {
    response_cache: &'a ResponseCache,
    stats: SimStats,
    policy: &'a dyn Policy,
}

pub trait Policy: Sync + Send {
    fn pick(&self, candidates: &BitVec<u64, Lsb0>) -> usize;
}

pub struct MaxFreqPolicy<'a>{
    pub all_candidates: &'a [[u8; N_CHARS]],
    pub all_guesses: &'a [[u8; N_CHARS]],
}

impl<'a> Policy for MaxFreqPolicy<'a> {
    fn pick(&self, candidates: &BitVec<u64, Lsb0>) -> usize {
        let candidate_indices: Vec<usize> = candidates.iter_ones().collect();
        let all_guesses_indices: Vec<usize> = (0..self.all_guesses.len()).collect();
        pick_max_freq(&candidate_indices, &all_guesses_indices, self.all_candidates, self.all_guesses)
    }
}
pub struct MinRemainingPolicy<'a> {
    pub response_cache: &'a ResponseCache,
    pub n_total_candidates: usize,
    pub n_total_guesses: usize,
}

impl<'a> Policy for MinRemainingPolicy<'a> {

    #[inline]
    fn pick(&self, candidates: &BitVec<u64, Lsb0>) -> usize {
        let all_guess_indices: Vec<usize> = (0..self.n_total_guesses).collect();
        pick_min_remaining(candidates, &all_guess_indices, self.response_cache)
    }
}

pub struct OptimalPolicy<'a> {
    pub all_guesses: &'a [[u8; N_CHARS]],
    pub response_cache: &'a ResponseCache,
    pub memo: &'a MemoCache,
    pub n_total_candidates: usize,
}

impl<'a> Policy for OptimalPolicy<'a> {
    #[inline]
    fn pick(&self, candidates: &BitVec<u64, Lsb0>) -> usize {
        pick_optimal(
            candidates, self.all_guesses.len(),
            self.response_cache, self.memo
        ).unwrap()
    }
}

pub fn simulate<'a>(
    response_cache: &ResponseCache,
    initial_candidates: &BitVec<u64, Lsb0>,
    policy: &'a dyn Policy,
) -> SimStats {
    let mut sim = Simulation {
        response_cache, stats: SimStats::new(), policy,
    };
    sim.run(initial_candidates, 1);
    sim.stats
}