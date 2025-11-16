use ndarray::{Array2, ArrayView1};
use std::collections::HashSet;
use std::fs;
use std::io;
use std::path::Path;
use thiserror::Error;
use log::{info, warn};

#[derive(Error, Debug)]
pub enum ConversionError {
    #[error("Invalid word length for \"{word}\": expected {expected}, found {found}")]
    LengthMismatch {
        expected: usize,
        found: usize,
        word: String,
    },
}

/// Parses a string of words into a vector of unique words of the given length.
pub fn parse_words(content: &str, n_chars: usize) -> HashSet<String> {
    let mut valid_words: HashSet<String> = HashSet::new();
    for word in content.split_whitespace() {
        if word.len() == n_chars {
            valid_words.insert(word.to_uppercase());
        } else {
            warn!("Skipping invalid word: \"{}\"", word);
        }
    }
    valid_words
}

/// Loads a list of words from a file into a vector of unique words of the given length.
pub fn load_words(words_file: &Path, n_chars: usize) -> Result<HashSet<String>, io::Error> {
    let all_text = fs::read_to_string(words_file)?;
    let words = parse_words(&all_text, n_chars);
    info!(
        "Successfully loaded {} unique {}-letter words from {:?}.",
        words.len(),
        n_chars,
        words_file
    );
    Ok(words)
}

/// Converts a set of equal-length strings to a 2D byte array.
pub fn words_to_arr(words: &HashSet<String>, n_chars: usize) -> Result<Array2<u8>, ConversionError> {
    let n_samples = words.len();

    let mut sorted_words: Vec<&String> = words.iter().collect();
    sorted_words.sort();

    let mut flat_data: Vec<u8> = Vec::with_capacity(n_samples * n_chars);
    for word in sorted_words {
        if word.len() != n_chars {
            return Err(ConversionError::LengthMismatch {
                expected: n_chars,
                found: word.len(),
                word: word.to_string(),
            });
        }
        flat_data.extend_from_slice(word.as_bytes());
    }

    let array = Array2::from_shape_vec((n_samples, n_chars), flat_data)
        .expect("Array shape and data length mismatch. This is a bug.");

    Ok(array)
}


pub fn arr_to_string(arr: ArrayView1<u8>) -> String {
    let bytes: &[u8] = arr.as_slice()
        .expect("ArrayView data must be contiguous");

    str::from_utf8(bytes)
        .expect("Byte array must be valid UTF-8")
        .to_string()
}