use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon::iter::IndexedParallelIterator;
use log::info;
use rayon::slice::ParallelSliceMut;
use crate::bits::BitSet;
use crate::words::{is_letter_char, letter_to_index, N_CHARS, N_LETTERS};

pub const B: u8 = b'B';
pub const Y: u8 = b'Y';
pub const G: u8 = b'G';

pub const B_IDX: usize = 0;
pub const Y_IDX: usize = 1;
pub const G_IDX: usize = 2;

const IDX_TO_RESP_CHAR: [u8; 3] = [B, Y, G];
const CHAR_TO_RESP_IDX: [usize; 4] = [0, Y_IDX, B_IDX, G_IDX];

pub const N_RESPONSES: usize = 243; // 3^N_CHARS = 243
const POWERS_OF_3: [usize; N_CHARS] = [1, 3, 9, 27, 81];
pub const CORRECT: [u8; N_CHARS] = [G; N_CHARS];

pub const CORRECT_IDX: usize = N_RESPONSES - 1;

/// Checks if a given byte is a response character (B, Y, G).
#[inline]
pub fn is_resp_char(c: u8) -> bool {
    (c == B) | (c == Y) | (c == G)
}

/// Converts a Wordle response [B, Y, G, ...] to a unique index 0-242.
#[inline]
pub fn response_to_index(response: &[u8; N_CHARS]) -> usize {
    let mut index = 0;
    for (i, &byte) in response.iter().enumerate() {
        let val = CHAR_TO_RESP_IDX[(byte & 3) as usize];
        index += val * POWERS_OF_3[i];
    }
    index
}

#[inline]
pub fn index_to_response(mut index: usize) -> [u8; N_CHARS] {
    let mut response = [0; N_CHARS];
    for i in 0..N_CHARS {
        let (q, r) = (index / 3, index % 3);
        index = q;
        // SAFETY: r is guaranteed to be 0, 1, or 2 by % 3
        response[i] = unsafe { *IDX_TO_RESP_CHAR.get_unchecked(r) };
    }
    response
}

/// Computes the response for a given (guess, candidate) pair.
pub fn get_resp(guess: &[u8; N_CHARS], candidate: &[u8; N_CHARS]) -> [u8; N_CHARS] {
    debug_assert!(guess.iter().all(|&c| is_letter_char(c)), "Guess must only contain letters");
    debug_assert!(candidate.iter().all(|&c| is_letter_char(c)), "Candidate must only contain letters");

    // Count frequencies in candidate
    let mut cand_counts = [0u8; N_LETTERS];
    for &letter in candidate {
        cand_counts[letter_to_index(letter)] += 1;
    }

    // Initially all black response
    let mut response = [B; N_CHARS];

    // First pass: Check greens
    for i in 0..N_CHARS {
        if guess[i] == candidate[i] {
            response[i] = G;
            cand_counts[letter_to_index(guess[i])] -= 1;
        }
    }

    // Second pass: Check yellows
    for i in 0..N_CHARS {
        if response[i] != G {
            let letter_idx = letter_to_index(guess[i]);
            if cand_counts[letter_idx] > 0 {
                response[i] = Y;
                cand_counts[letter_idx] -= 1;
            }
        }
    }

    response
}

pub struct ResponseCache {
    data: Vec<u8>,
    stride: usize, // n_total_candidates

}

impl ResponseCache {
    pub fn new(data: Vec<u8>, stride: usize) -> Self {
        assert_eq!(data.len() % stride, 0, "Data length must be multiple of stride");
        Self { data, stride }
    }

    #[inline(always)]
    pub fn get(&self, g_idx: usize, c_idx: usize) -> u8 {
        unsafe { *self.data.get_unchecked(g_idx * self.stride + c_idx) }
    }

    #[inline]
    pub fn get_row(&self, g_idx: usize) -> &[u8] {
        let start = g_idx * self.stride;
        let end = start + self.stride;
        unsafe { self.data.get_unchecked(start..end) }
    }

    #[inline]
    pub fn stride(&self) -> usize {
        self.stride
    }
}

/// Precomputes responses for all (guess, candidate) pairs.
pub fn compute_response_cache(
    guesses: &[[u8; N_CHARS]],
    candidates: &[[u8; N_CHARS]],
) -> ResponseCache {

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
    info!("Response cache built successfully. (~{})", format_bytes(total_bytes));

    // Return the fully computed cache.
    ResponseCache::new(cache_data, n_candidates)
}

pub type PartitionMap = Vec<Vec<(usize, BitSet)>>;

/// Builds the partition map (Inverted Index)
pub fn build_partition_map(cache: &ResponseCache) -> PartitionMap {
    info!("Building partition map (reverse index)...");
    let n_candidates = cache.stride();
    let n_guesses = cache.data.len() / n_candidates;
    let map: Vec<Vec<(usize, BitSet)>> = (0..n_guesses).into_par_iter().map(|g_idx| {
            // Create sparse array of partitions
            let mut partitions: [Option<BitSet>; N_RESPONSES] = std::array::from_fn(|_| None);

            // Populate partitions
            let row = cache.get_row(g_idx);
            for (c_idx, &resp_byte) in row.iter().enumerate() {
                let resp_idx = resp_byte as usize;
                partitions[resp_idx]
                    .get_or_insert_with(|| BitSet::new(n_candidates))
                    .set(c_idx, true);
            }

            // Collapse into compact vector
            partitions
                .into_iter()
                .enumerate()
                .filter_map(|(r_idx, opt_bs)| {
                    opt_bs.map(|bs| (r_idx, bs))
                })
                .collect()
        })
        .collect();

    // Log size
    let bytes = estimate_partition_map_size(&map);
    info!("Partition map built. (~{})", format_bytes(bytes));

    map
}

pub fn estimate_partition_map_size(map: &PartitionMap) -> usize {
    let mut total_bytes = 0;
    total_bytes += size_of_val(map); // Size of the Vec struct itself (ptr, cap, len)
    for row in map {
        // Add the inline overhead of the inner Vec (each row)
        total_bytes += size_of_val(row);
        for (_, bs) in row {
            // Add the inline size of the (usize, BitSet) tuple element
            total_bytes += size_of::<(usize, BitSet)>();
            // Add the heap-allocated payload (u64 blocks) for the BitSet
            total_bytes += bs.as_slice().len() * size_of::<u64>();
        }
    }
    total_bytes
}

const KB: usize = 1024;
const MB: usize = 1024;
const GB: usize = 1024;

/// Formats a byte count into a human-readable string (KB, MB, GB).
pub fn format_bytes(bytes: usize) -> String {
    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} bytes", bytes)
    }
}

/// Fast intersection count using BitSets.
#[inline]
pub fn get_partition_counts(
    g_idx: usize,
    candidates: &BitSet,
    partition_map: &PartitionMap,
) -> Vec<(usize, usize)> { // Returns (resp_idx, count)
    let partitions = unsafe { partition_map.get_unchecked(g_idx) };
    let mut counts = Vec::with_capacity(partitions.len());
    for (resp_idx, mask) in partitions {
        let count = mask.intersection_count(candidates);
        if count > 0 {
            counts.push((*resp_idx, count));
        }
    }
    counts
}

/// Fast bitwise intersection using BitSets.
#[inline]
pub fn get_partitions(
    g_idx: usize,
    candidates: &BitSet,
    partition_map: &PartitionMap,
) -> Vec<(usize, BitSet, usize)> {
    let partitions = unsafe { partition_map.get_unchecked(g_idx) };
    let mut result = Vec::with_capacity(partitions.len());
    for (resp_idx, mask) in partitions {
        let p_count = mask.intersection_count(candidates);
        if p_count > 0 {
            let p_cand = mask.intersect(candidates);
            result.push((*resp_idx, p_cand, p_count));
        }
    }
    result
}

/// Fast bitwise intersection using BitSets, only creates BitSets for partitions with >2 elements.
#[inline]
pub fn get_lazy_partitions(
    g_idx: usize,
    candidates: &BitSet,
    partition_map: &PartitionMap,
) -> Vec<(usize, Option<BitSet>, usize)> {
    let partitions = unsafe { partition_map.get_unchecked(g_idx) };
    let mut result = Vec::with_capacity(partitions.len());
    for (resp_idx, mask) in partitions {
        let count = mask.intersection_count(candidates);
        if count == 0 { continue; }
        let p_cand;
        if count > 2 {
            p_cand = Some(mask.intersect(candidates))
        } else {
            p_cand = None
        }
        result.push((*resp_idx, p_cand, count));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitwise_hashing_logic() {
        // Verify that our bitwise trick actually maps the characters to
        // the intended values: B=0, Y=1, G=2

        // B is ASCII 66 (01000010), ends in 10 (2). Map[2] should be 0.
        assert_eq!(CHAR_TO_RESP_IDX[(B & 3) as usize], B_IDX);

        // Y is ASCII 89 (01011001), ends in 01 (1). Map[1] should be 1.
        assert_eq!(CHAR_TO_RESP_IDX[(Y & 3) as usize], Y_IDX);

        // G is ASCII 71 (01000111), ends in 11 (3). Map[3] should be 2.
        assert_eq!(CHAR_TO_RESP_IDX[(G & 3) as usize], G_IDX);
    }

    #[test]
    fn test_min_max_values() {
        // Test Index all black
        let all_black = [B, B, B, B, B];
        assert_eq!(response_to_index(&all_black), 0);
        assert_eq!(index_to_response(0), all_black);

        // Test Index all green
        let all_green = CORRECT;
        assert_eq!(response_to_index(&all_green), CORRECT_IDX);
        assert_eq!(index_to_response(CORRECT_IDX), all_green);
    }

    #[test]
    fn test_mixed_values() {
        // Test an arbitrary pattern: B Y G B Y
        // 0*1 + 1*3 + 2*9 + 0*27 + 1*81 = 3 + 18 + 81 = 102
        let pattern = [B, Y, G, B, Y];
        let idx = response_to_index(&pattern);
        assert_eq!(idx, 102);
        assert_eq!(index_to_response(102), pattern);
    }

    #[test]
    fn test_round_trip_all() {
        // Brute force verify all 243 combinations
        for i in 0..N_RESPONSES {
            let resp = index_to_response(i);
            let idx = response_to_index(&resp);
            assert_eq!(i, idx, "Failed round trip at index {}", i);
        }
    }

    #[test]
    fn test_all_indices_produce_unique_patterns() {
        use std::collections::HashSet;

        let mut seen_patterns = HashSet::new();

        for i in 0..N_RESPONSES {
            let resp = index_to_response(i);
            let is_unique = seen_patterns.insert(resp);
            assert!(is_unique, "Duplicate pattern {:?} for index {}!", resp, i);
        }
    }

    #[test]
    fn test_all_responses_contain_valid_chars() {
        for i in 0..N_RESPONSES {
            let resp = index_to_response(i);
            for (char_idx, &c) in resp.iter().enumerate() {
                assert!(
                    is_resp_char(c),
                    "Pattern {:?} contains invalid character {} at pos {}",
                    resp, c, char_idx
                );
            }
        }
    }
}