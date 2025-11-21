use std::collections::{HashMap};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use serde::{Deserialize, Serialize};

/// A single node in the strategy DAG.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StrategyNode {
    /// The index of the guess in the sorted guesses list.
    pub guess: usize,
    pub children: HashMap<String, usize>,
}

/// The root container for the strategy.
#[derive(Serialize, Deserialize, Debug)]
pub struct Strategy {
    /// A stable hash of the word lists used to generate the strategy.
    pub context_hash: u64,
    /// The index of the starting node in the nodes list.
    pub root: usize,
    /// The flat list of all nodes.
    pub nodes: Vec<StrategyNode>,
}

impl Strategy {
    pub fn to_json_file(&self, path: &Path) -> anyhow::Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }

    pub fn from_json_file(path: &Path) -> anyhow::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let strategy = serde_json::from_reader(reader)?;
        Ok(strategy)
    }
}