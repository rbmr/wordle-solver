use std::collections::HashSet;
use std::fs;
use std::io;
use std::path::Path;
use log::{debug, info, warn};
use once_cell::sync::Lazy;

pub const N_CHARS: usize = 5;
pub const N_LETTERS: usize = 26; // A-Z

/// Checks if a given byte is a letter (A-Z).
#[inline]
pub fn is_letter_char(c: u8) -> bool {
    (c >= b'A') & (c <= b'Z')
}

/// Converts a letter byte (A-Z) to an index 0-25.
#[inline]
pub fn letter_to_index(c: u8) -> usize {
    (c - b'A') as usize
}

/// Parses a string of words into a vector of unique words of the given length.
pub fn parse_words(content: &str, n_chars: usize) -> HashSet<String> {
    let mut valid_words: HashSet<String> = HashSet::new();
    for word in content.split_whitespace() {
        if word.len() != n_chars {
            warn!("Skipping word with unexpected len: \"{}\"", word);
            continue;
        }
        let word_uppercase = word.to_uppercase();
        if !word_uppercase.chars().all(|c| is_letter_char(c as u8)) {
            warn!("Skipping word with invalid chars: \"{}\"", word);
            continue
        }
        valid_words.insert(word_uppercase);
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
pub fn words_to_arr(words: &HashSet<String>) -> Vec<[u8; N_CHARS]> {
    let mut sorted_words: Vec<&String> = words.iter().collect();
    sorted_words.sort();

    let mut data = Vec::with_capacity(words.len());
    for word in sorted_words {
        assert_eq!(word.len(), N_CHARS, "Unexpected word len: expected {}, found {}", N_CHARS, word.len());
        let bytes = word.as_bytes();
        let mut fixed_arr = [0u8; N_CHARS];
        fixed_arr.copy_from_slice(&bytes[0..N_CHARS]);
        data.push(fixed_arr);
    }
    data
}

pub fn arr_to_word(arr: &[u8; N_CHARS]) -> String {
    str::from_utf8(arr)
        .expect("Byte array must be valid UTF-8")
        .to_string()
}

/// The default list of candidate words.
pub static CANDIDATES: Lazy<HashSet<String>> = Lazy::new(|| {
    debug!("One-time parse: Loading default candidates...");
    let file_contents = include_str!("../words/candidates.txt");
    parse_words(file_contents, N_CHARS)
});

/// The default list of words to guess from.
pub static GUESSES: Lazy<HashSet<String>> = Lazy::new(|| {
    debug!("One-time parse: Loading default guesses...");
    let file_contents = include_str!("../words/guesses.txt");
    parse_words(file_contents, N_CHARS)
});