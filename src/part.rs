use rayon::iter::ParallelIterator;
use rayon::iter::IndexedParallelIterator;
use bitvec::bitvec;
use bitvec::order::Lsb0;
use bitvec::prelude::BitVec;
use log::info;
use rayon::prelude::ParallelSliceMut;
use crate::resp::{get_resp, response_to_index, CORRECT_IDX, N_RESPONSES};
use crate::words::N_CHARS;
use crate::utils::format_bytes;



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

/// Returns all non-zero sized partitions and their counts, excluding the Green response.
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

/// Returns all non-zero sized partitions with their counts and resp_idx, excluding the Green response.
#[inline]
pub fn get_partitions(
    candidates: &BitVec<u64, Lsb0>,
    response_cache: &[u8],
    g_idx: usize,
    n_total_candidates: usize,
) -> Vec<(usize, BitVec<u64, Lsb0>)> {

    // Create arrays for partitions and for counts.
    let mut partitions: [Option<BitVec<u64, Lsb0>>; N_RESPONSES] =
        std::array::from_fn(|_| None);

    // Iterate over all candidates to fill partitions.
    for c_idx in candidates.iter_ones() {
        let resp_idx = response_cache[g_idx * n_total_candidates + c_idx] as usize;
        if resp_idx == CORRECT_IDX { continue; } // Don't create partition for Green response.
        partitions[resp_idx]
            .get_or_insert_with(|| bitvec![u64, Lsb0; 0; candidates.len()])
            .set(c_idx, true);
    }

    // Collect all created BitVecs and their corresponding counts.
    partitions
        .into_iter()
        .enumerate()
        .filter_map(|(resp_idx, p_opt)| p_opt.map(|p| (resp_idx, p)))
        .collect()
}