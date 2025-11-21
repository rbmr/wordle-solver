use rayon::iter::ParallelIterator;
use rayon::iter::IndexedParallelIterator;
use bitvec::bitvec;
use bitvec::order::Lsb0;
use bitvec::prelude::BitVec;
use log::info;
use rayon::prelude::ParallelSliceMut;
use crate::game::resp::{get_resp, response_to_index, CORRECT_IDX, N_RESPONSES};
use crate::game::words::N_CHARS;
use crate::utils::format_bytes;

/// Precomputes responses for all (guess, candidate) pairs.
pub fn compute_response_cache(
    guesses: &[[u8; N_CHARS]],
    candidates: &[[u8; N_CHARS]],
) -> Box<[u8]> {

    info!("Building response cache...");
    let n_guesses = guesses.len();
    let n_candidates = candidates.len();
    let element_count = n_guesses * n_candidates;
    let mut cache_data = vec![0u8; element_count];

    cache_data
        .par_chunks_mut(n_candidates) // each slice has length n_candidates (one row)
        .enumerate() // enumerate to get g_idx
        .for_each(|(g_idx, row_slice)| {
            let guess = &(guesses[g_idx]);
            for (c_idx, slot) in row_slice.iter_mut().enumerate() {
                let candidate = &candidates[c_idx];
                let resp = get_resp(guess, candidate);
                *slot = response_to_index(&resp) as u8;
            }
        });

    // Log the final cache size
    let element_size = size_of::<u8>();
    let total_bytes = element_count * element_size;
    info!("Response cache built successfully. (~{})", format_bytes(total_bytes as f32));

    // Return the fully computed cache.
    cache_data.into_boxed_slice()
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