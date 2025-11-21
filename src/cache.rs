use std::collections::hash_map::DefaultHasher;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::time::Instant;

use dashmap::DashMap;
use bitvec::prelude::*;
use log::{info, warn, error};
use serde::{Serialize, Deserialize};

pub type MemoCache = DashMap<BitVec<u64, Lsb0>, usize>;
#[derive(Deserialize)]
struct PersistedCache {
    context_hash: u64,
    data: MemoCache,
}

#[derive(Serialize)]
struct PersistedCacheRef<'a> {
    context_hash: u64,
    data: &'a MemoCache,
}

pub fn new_cache() -> MemoCache {
    DashMap::new()
}

/// Computes a deterministic hash of the game configuration (guesses and candidates).
/// If these lists change, the hash changes, invalidating old cache files.
pub fn compute_context_hash<T: Hash>(guesses: &[T], candidates: &[T]) -> u64 {
    let mut hasher = DefaultHasher::new();
    // We hash the lengths to prevent collision on edge cases
    guesses.len().hash(&mut hasher);
    candidates.len().hash(&mut hasher);
    // We hash the content
    guesses.hash(&mut hasher);
    candidates.hash(&mut hasher);
    hasher.finish()
}

/// Attempts to save the cache to disk with a context hash.
/// Returns an error if encoding failed.
pub fn save_cache(
    cache: &MemoCache,
    path: &Path,
    context_hash: u64,
) -> std::io::Result<()> {
    info!("Persisting cache to {:?}...", path);
    let start = Instant::now();

    // Add a hash to the cache.
    let envelope = PersistedCacheRef {
        context_hash,
        data: cache,
    };

    // Encode and save to disk.
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    let config = bincode::config::standard();
    match bincode::serde::encode_into_std_write(&envelope, &mut writer, config) {
        Ok(bytes) => {
            info!("Cache saved. Size: {:.2} MB. Time: {:.2?}. Items: {}",
                bytes as f64 / (1024.0 * 1024.0),
                start.elapsed(),
                cache.len()
            );
            Ok(())
        },
        Err(e) => {
            Err(std::io::Error::new(std::io::ErrorKind::Other, e))
        }
    }
}

/// Attempts to load the cache from disk.
/// Validates that the file exists, is valid bincode, AND matches the provided words hash.
pub fn load_cache(
    path: &Path,
    context_hash: u64,
) -> std::io::Result<MemoCache> {

    // Validate the file exists.
    if !path.exists() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "Cache file not found"
        ));
    }

    // Load the file
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let config = bincode::config::standard();
    let envelope: PersistedCache = bincode::serde::decode_from_std_read(&mut reader, config)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    // Validate the hash
    if envelope.context_hash != context_hash {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "Context mismatch! Cache was built for different words. (File: {:x}, Current: {:x})",
                envelope.context_hash, context_hash
            )
        ));
    }

    Ok(envelope.data)
}

pub fn just_load_cache(path: &Path, context_hash: u64) -> MemoCache {
    match load_cache(path, context_hash) {
        Ok(cache) => {
            info!("Cache successfully loaded from {:?}. Items: {}", path, cache.len());
            cache
        },
        Err(e) => {
            // Distinguish between "Not Found" (Normal) and "Mismatch/Corrupt" (Warning)
            if e.kind() == std::io::ErrorKind::NotFound {
                info!("No cache found at {:?}. Starting fresh.", path);
            } else {
                warn!("Could not load cache from {:?}: {}. Starting with a fresh cache.", path, e);
            }
            new_cache()
        }
    }
}

pub fn just_save_cache(
    cache: &MemoCache,
    path: &Path,
    context_hash: u64,
) {
    if let Err(e) = save_cache(cache, path, context_hash) {
        error!("Could not save cache to {:?}: {}.", path, e);
    }
}