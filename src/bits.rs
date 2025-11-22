use std::fmt;
use serde::{Deserialize, Serialize};

/// A lightweight, dense bitset backed by a vector of u64s.
#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BitSet {
    blocks: Vec<u64>,
    pub(crate) len: usize,
}

impl BitSet {
    /// Create a new BitSet with `len` bits, all set to 0.
    pub fn new(len: usize) -> Self {
        let num_blocks = (len + 63) / 64;
        let blocks = vec![0; num_blocks];
        Self { blocks, len }
    }

    /// Create a new BitSet with `len` bits, all set to 1.
    pub fn ones(len: usize) -> Self {
        let num_blocks = (len + 63) / 64;
        let mut blocks = vec![u64::MAX; num_blocks];
        let remainder = len % 64;
        if remainder != 0 {
            // Create a mask with 'remainder' 1s in the LSBs
            let mask = (1u64 << remainder) - 1;
            blocks[num_blocks - 1] &= mask;
        }

        Self { blocks, len }
    }

    #[inline(always)]
    pub fn set(&mut self, index: usize, value: bool) {
        if index >= self.len { return; }
        let (block_idx, bit_idx) = (index / 64, index % 64);
        if value {
            self.blocks[block_idx] |= 1 << bit_idx;
        } else {
            self.blocks[block_idx] &= !(1 << bit_idx);
        }
    }

    #[inline(always)]
    pub fn get(&self, index: usize) -> bool {
        if index >= self.len { return false; }
        let (block_idx, bit_idx) = (index / 64, index % 64);
        (self.blocks[block_idx] & (1 << bit_idx)) != 0
    }

    #[inline(always)]
    pub fn count_ones(&self) -> usize {
        self.blocks.iter().map(|&b| b.count_ones() as usize).sum()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.blocks.iter().all(|&b| b == 0)
    }

    /// Computes the intersection of self and other, returning the count of ones.
    /// This is the "Hot Path" for the heuristic.
    #[inline(always)]
    pub fn intersection_count(&self, other: &BitSet) -> usize {
        self.blocks.iter()
            .zip(&other.blocks)
            .map(|(a, b)| (a & b).count_ones() as usize)
            .sum()
    }

    /// Returns a new BitSet representing the intersection of self and other.
    pub fn intersect(&self, other: &BitSet) -> BitSet {
        let blocks = self.blocks.iter()
            .zip(&other.blocks)
            .map(|(a, b)| a & b)
            .collect();
        BitSet { blocks, len: self.len }
    }

    /// Returns an iterator over the indices of set bits.
    pub fn iter_ones(&self) -> BitSetIter<'_> {
        BitSetIter::new(self)
    }

    /// Access the raw blocks (useful for serialization if needed).
    pub fn as_slice(&self) -> &[u64] {
        &self.blocks
    }
}

impl fmt::Debug for BitSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BitSet(len={}, ones={})", self.len, self.count_ones())
    }
}

/// Iterator for set bits.
pub struct BitSetIter<'a> {
    bitset: &'a BitSet,
    block_idx: usize,
    current_block: u64,
}

impl<'a> BitSetIter<'a> {
    fn new(bitset: &'a BitSet) -> Self {
        let current_block = if bitset.blocks.is_empty() { 0 } else { bitset.blocks[0] };
        Self { bitset, block_idx: 0, current_block }
    }
}

impl<'a> Iterator for BitSetIter<'a> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.current_block != 0 {
                let tz = self.current_block.trailing_zeros(); // Find the bit
                self.current_block &= self.current_block - 1; // Clear the bit
                return Some(self.block_idx * 64 + tz as usize);
            }

            // Move to next block
            self.block_idx += 1;
            if self.block_idx >= self.bitset.blocks.len() {
                return None;
            }
            self.current_block = self.bitset.blocks[self.block_idx];
        }
    }
}