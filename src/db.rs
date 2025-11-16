use bitvec::prelude::BitVec;
use log::{info, trace};
use ndarray::ArrayView2;
use sled::Db;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use bincode::config::standard;
use serde::{Serialize, de::DeserializeOwned};
use bincode::serde::{encode_to_vec, decode_from_slice};

/// Generates a unique database name based on the content of the word lists.
pub fn generate_db_name(
    all_candidates_arr: ArrayView2<u8>,
    all_guesses_arr: ArrayView2<u8>,
) -> String {
    let n_candidates = all_candidates_arr.nrows();
    let n_guesses = all_guesses_arr.nrows();

    let mut hasher = DefaultHasher::new();
    all_candidates_arr.as_slice().unwrap().hash(&mut hasher);
    all_guesses_arr.as_slice().unwrap().hash(&mut hasher);
    let data_hash = hasher.finish();

    let db_name = format!(
        "{}_{}_{:x}.sled",
        n_candidates, n_guesses, data_hash
    );
    info!("Using database file: {}", &db_name);
    db_name
}

/// A wrapper for the Sled on-disk database.
pub struct CacheManager<K,V> {
    db: Db,
    _marker: std::marker::PhantomData<(K, V)>,
}

impl<K,V> CacheManager<K, V>
where
    K: Serialize + DeserializeOwned,
    V: Serialize + DeserializeOwned
{
    /// Opens or creates a database at the given path.
    pub fn new(path: &str) -> Self {
        let db = sled::open(path).expect("Failed to open sled database");
        Self { db, _marker: std::marker::PhantomData}
    }

    /// Inserts a (K, V) pair into the database.
    pub fn insert(&self, key: &K, value: &V) {
        let key_bytes = encode_to_vec(key, bincode::config::standard())
            .expect("Failed to serialize key");
        let val_bytes = encode_to_vec(value, bincode::config::standard())
            .expect("Failed to serialize value");
        trace!("Inserting value into database");
        self.db.insert(key_bytes, val_bytes).expect("DB insert failed");
    }

    /// Retrieves a value from the database, deserializing it.
    pub fn get(&self, key: &K) -> Option<V> {
        let key_bytes = encode_to_vec(key, bincode::config::standard())
            .expect("Failed to serialize key for get");
        match self.db.get(key_bytes).expect("DB get failed") {
            Some(val_bytes) => {
                let (value, _bytes_read): (V, usize) =
                    decode_from_slice(&val_bytes, bincode::config::standard())
                        .expect("Failed to deserialize value");
                Some(value)
            }
            None => None,
        }
    }

    /// Checks if a key exists in the database.
    pub fn contains_key(&self, key: &BitVec<u64>) -> bool {
        let key_bytes = encode_to_vec(key, standard())
            .expect("Failed to serialize key for contains_key");
        self.db.contains_key(key_bytes).expect("DB contains_key failed")
    }

    /// Returns the number of entries in the database.
    pub fn len(&self) -> usize {
        self.db.len()
    }

    /// Flushes data to disk.
    pub fn flush(&self) {
        self.db.flush().expect("DB flush failed");
    }
}