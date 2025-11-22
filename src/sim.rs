use std::collections::BTreeMap;
use crate::bits::BitSet;
use crate::resp::{get_partitions, PartitionMap, CORRECT_IDX};
use crate::strat::{pick_max_freq, pick_min_remaining};
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

/// Context to hold shared data for the recursive heuristic simulation.
struct Simulation<'a> {
    partition_map: &'a PartitionMap,
    stats: SimStats,
    policy: &'a dyn Policy,
}

impl<'a> Simulation<'a> {

    pub fn run(
        &mut self,
        candidates: &BitSet,
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
        let partitions = get_partitions(guess_idx, candidates, self.partition_map);
        assert!(partitions.len() > 1, "Partitioning failed, policy is broken.");

        // Recurse.
        for (resp_idx, p_cand, _) in partitions {
            if resp_idx == CORRECT_IDX {
                self.stats.inc(current_depth);
                continue;
            }
            self.run(&p_cand, current_depth + 1);
        }
    }
}

pub trait Policy: Sync + Send {
    fn pick(&self, candidates: &BitSet) -> usize;
}

pub struct MaxFreqPolicy<'a>{
    pub all_candidates: &'a [[u8; N_CHARS]],
    pub all_guesses: &'a [[u8; N_CHARS]],
}

impl<'a> Policy for MaxFreqPolicy<'a> {
    fn pick(&self, candidates: &BitSet) -> usize {
        let candidate_indices: Vec<usize> = candidates.iter_ones().collect();
        let all_guesses_indices: Vec<usize> = (0..self.all_guesses.len()).collect();
        pick_max_freq(&candidate_indices, &all_guesses_indices, self.all_candidates, self.all_guesses)
    }
}
pub struct MinRemainingPolicy<'a> {
    pub partition_map: &'a PartitionMap,
    pub n_total_candidates: usize,
    pub n_total_guesses: usize,
}

impl<'a> Policy for MinRemainingPolicy<'a> {

    #[inline]
    fn pick(&self, candidates: &BitSet) -> usize {
        let all_guess_indices: Vec<usize> = (0..self.n_total_guesses).collect();
        pick_min_remaining(candidates, &all_guess_indices, self.partition_map)
    }
}

pub fn simulate<'a>(
    partition_map: &PartitionMap,
    initial_candidates: &BitSet,
    policy: &'a dyn Policy,
) -> SimStats {
    let mut sim = Simulation { partition_map, stats: SimStats::new(), policy };
    sim.run(initial_candidates, 1);
    sim.stats
}