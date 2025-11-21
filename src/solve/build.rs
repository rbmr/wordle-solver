use std::collections::HashMap;
use bitvec::order::Lsb0;
use bitvec::prelude::BitVec;
use log::info;
use crate::game::resp::index_to_response;
use crate::game::strat::{Strategy, StrategyNode};
use crate::game::words::{arr_to_word};
use crate::solve::policy::Policy;
use crate::solve::utils::{get_partitions};

pub struct StrategyBuilder<'a, P:Policy> {
    response_cache: &'a [u8],
    n_total_candidates: usize,
    policy: P,
    all_guess_indices: Vec<usize>,
    visited: HashMap<BitVec<u64, Lsb0>, usize>, // Candidate Set -> Node Index
    nodes: Vec<StrategyNode>,
}

impl<'a, P: Policy> StrategyBuilder<'a, P> {

    pub fn new(
        response_cache: &'a [u8],
        n_total_candidates: usize,
        n_total_guesses: usize,
        policy: P,
    ) -> Self {
        Self {
            response_cache,
            n_total_candidates,
            policy,
            // In case of hard mode, this would be different.
            all_guess_indices: (0..n_total_guesses).collect(),
            visited: HashMap::new(),
            nodes: Vec::new(),
        }
    }

    pub fn build(mut self, initial_candidates: &BitVec<u64, Lsb0>, context_hash: u64) -> Strategy {
        info!("Building Strategy...");
        let root_idx = self.process_state(initial_candidates.clone());
        info!("Strategy built. Nodes: {}", self.nodes.len());

        Strategy {
            context_hash,
            root: root_idx,
            nodes: self.nodes,
        }
    }

    fn process_state(&mut self, candidates: BitVec<u64, Lsb0>) -> usize {
        // Check Memoization
        if let Some(&idx) = self.visited.get(&candidates) {
            return idx;
        }

        // Ask Policy for the best guess
        let guess_idx = self.policy.pick(&candidates, self.all_guess_indices.as_slice());

        // Generate Partitions
        let partitions = get_partitions(
            &candidates, self.response_cache,
            guess_idx, self.n_total_candidates,
        );

        // Create Children Map
        let mut children_map = HashMap::new();
        for (r_idx, p) in partitions {
            // Recurse: Get the index of the child node
            let child_node_idx = self.process_state(p);

            // Convert numeric index to String key
            let resp_arr = index_to_response(r_idx);
            let resp_str = arr_to_word(&resp_arr);

            children_map.insert(resp_str, child_node_idx);
        }

        // Construct the node
        let node = StrategyNode{
            guess: guess_idx,
            children: children_map,
        };

        // Add the node to the strategy
        let node_idx = self.nodes.len();
        self.nodes.push(node);
        self.visited.insert(candidates.clone(), node_idx);

        node_idx
    }
}