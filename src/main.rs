use std::collections::HashSet;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use anyhow::{bail, Context};
use bitvec::bitvec;
use bitvec::order::Lsb0;
use bitvec::vec::BitVec;
use log::{info, warn};
use clap::{Parser, Subcommand};
use wordle_solver::cache::{compute_context_hash, just_save_cache, new_cache};
use wordle_solver::graph::generate_comparison_image;
use wordle_solver::strat::pick_optimal;
use wordle_solver::resp::{compute_response_cache, get_resp, response_to_index, ResponseCache, B, CORRECT_IDX, G, Y};
use wordle_solver::sim::{simulate, MaxFreqPolicy, MinRemainingPolicy, Policy, SimStats};
use wordle_solver::solver::compute_optimal_move;
use wordle_solver::words::{arr_to_word, load_words, words_to_arr, CANDIDATES, GUESSES, N_CHARS};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Optional path to custom guesses file
    #[arg(long = "guesses", short = 'g', global = true)]
    guesses_file: Option<PathBuf>,

    /// Optional path to custom candidates file
    #[arg(long = "candidates", short = 'c', global = true)]
    candidates_file: Option<PathBuf>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Play the game using the optimal move cache
    Play {
        /// Path to the cache file
        #[arg(short, long, default_value = "solver_cache.bin")]
        cache: PathBuf,
    },
    /// Generate the optimal move cache
    Generate {
        /// Path to the cache file
        #[arg(short, long, default_value = "solver_cache.bin")]
        cache: PathBuf,
    },
    /// Generate a PNG comparing heuristics
    Compare {
        #[arg(short, long, default_value = "solver_cache.bin")]
        cache: PathBuf,
        #[arg(short, long, default_value = "comparison.svg")]
        output: PathBuf,
    },
}

fn main() -> Result<(), anyhow::Error> {
    env_logger::init();
    let cli = Cli::parse();

    // Load data
    info!("Loading word lists...");
    // Resolve Guesses
    let mut guesses_set = if let Some(path) = &cli.guesses_file {
        info!("Loading guesses from custom file: {:?}", path);
        load_words(path, N_CHARS)?
    } else {
        info!("Using default guesses list.");
        GUESSES.clone()
    };

    // Resolve Candidates
    let candidates_set = if let Some(path) = &cli.candidates_file {
        info!("Loading candidates from custom file: {:?}", path);
        load_words(path, N_CHARS)?
    } else {
        info!("Using default candidates list.");
        CANDIDATES.clone()
    };

    // Ensure Candidates is a subset of Guesses
    let missing: Vec<String> = candidates_set.difference(&guesses_set).cloned().collect();
    if !missing.is_empty() {
        warn!("Adding {} missing candidates into list of guesses.", missing.len());
        guesses_set.extend(missing);
    }

    // Convert into arrays
    let all_candidates = words_to_arr(&candidates_set);
    let all_guesses = words_to_arr(&guesses_set);
    let context_hash = compute_context_hash(&all_guesses, &all_candidates);
    let response_cache = compute_response_cache(&all_guesses, &all_candidates);

    match cli.command {
        Commands::Play { cache } => {
            play(
                &cache,
                context_hash,
                all_candidates,
                all_guesses,
                &response_cache
            )?;
        }
        Commands::Generate { cache } => {
            generate(
                &cache,
                context_hash,
                &all_candidates,
                &all_guesses,
                &response_cache
            )?;
        },
        Commands::Compare { cache, output } => {
             compare_heuristics(
                 &cache, &output,
                 context_hash,
                 &all_candidates,
                 &all_guesses,
                 &response_cache
            )?;
        }
    }

    Ok(())
}

fn generate(
    cache_path: &Path,
    context_hash: u64,
    candidates: &[[u8; N_CHARS]],
    guesses: &[[u8; N_CHARS]],
    response_cache: &ResponseCache,
) -> Result<(), anyhow::Error> {
    info!("Entering GENERATE mode.");

    // Load or Create Cache
    let memo = if cache_path.exists() {
        info!("Loading existing cache from {:?}", cache_path);
        let loaded = wordle_solver::cache::load_cache(cache_path, context_hash)
            .context("Failed to load existing cache. Ensure file is not corrupt and matches word lists.")?;
        Arc::new(loaded)
    } else {
        info!("Creating new cache at {:?}", cache_path);
        Arc::new(new_cache())
    };

    // Ctrl-C Handler
    let memo_signal = memo.clone();
    let path_signal = cache_path.to_path_buf();
    ctrlc::set_handler(move || {
        info!("Received SIGINT. Saving cache...");
        just_save_cache(&memo_signal, &path_signal, context_hash);
        std::process::exit(0);
    }).expect("Error setting Ctrl-C handler");

    info!("Starting solver...");
    let (best_idx, cost) = compute_optimal_move(&candidates, &guesses,
                                                &response_cache, &memo
    );

    let best_word = arr_to_word(&guesses[best_idx]);
    info!("Optimization Complete.");
    info!("Best Start Word: {} (Index {})", best_word, best_idx);
    info!("Minimum Total Cost: {}", cost);

    just_save_cache(&memo, cache_path, context_hash);

    Ok(())
}

fn play(
    cache_path: &Path,
    context_hash: u64,
    candidates: Vec<[u8; N_CHARS]>,
    guesses: Vec<[u8; N_CHARS]>,
    response_cache: &ResponseCache,
) -> Result<(), anyhow::Error> {
    println!("--- Wordle Solver: PLAY Mode ---");

    // Load Cache
    if !cache_path.exists() {
        bail!("Cache file not found at {:?}. Run 'generate' first.", cache_path);
    }
    let memo = wordle_solver::cache::load_cache(cache_path, context_hash)
        .context("Failed to load cache")?;

    let mut current_candidates = bitvec![u64, Lsb0; 1; candidates.len()];
    let n_total_candidates = candidates.len();
    let n_total_guesses = guesses.len();

    loop {
        let count = current_candidates.count_ones();
        assert!(count > 0); // Should be guaranteed by get_user_response

        // Print Remaining Candidates
        print_candidates(&current_candidates, &candidates, None);

        // Get Optimal Guess
        println!("Thinking...");
        let guess_idx = pick_optimal(
            &current_candidates, n_total_guesses,
            &response_cache, &memo
        );

        let guess_idx = match guess_idx {
            Ok(idx) => idx,
            Err(e) => {
                return Err(anyhow::anyhow!(e));
            }
        };

        let guess_word_str = arr_to_word(&guesses[guess_idx]);
        println!("------------------------------------------------");
        println!("OPTIMAL GUESS: {guess_word_str}");
        println!("------------------------------------------------");

        // Request Response
        let resp_idx = get_response(&guesses[guess_idx], &current_candidates, &candidates)?;

        if resp_idx == CORRECT_IDX {
            println!("Congratulations! \u{1F389}"); // Party popper
            return Ok(());
        }

        // Filter Candidates
        let mut next_candidates = bitvec![u64, Lsb0; 0; n_total_candidates];
        let cache_row = response_cache.get_row(guess_idx);
        for c_idx in current_candidates.iter_ones() {
            let actual_resp = cache_row[c_idx] as usize;
            if actual_resp == resp_idx {
                next_candidates.set(c_idx, true);
            }
        }
        current_candidates = next_candidates;
    }
}

const DEFAULT_MAX_PRINT: usize = 512;

fn print_candidates(candidates: &BitVec<u64, Lsb0>, all_candidates: &[[u8; 5]], max_print: Option<usize>) {
    let count = candidates.count_ones();
    println!("\nRemaining Candidates: {}", count);

    let max_print = max_print.unwrap_or(DEFAULT_MAX_PRINT);

    let mut printed = 0;
    for idx in candidates.iter_ones() {
        if printed >= max_print {
            println!("and {} more...", count - printed);
            break;
        }
        print!("{} ", arr_to_word(&all_candidates[idx]));
        printed += 1;
    }
    println!();
}

fn get_response(
    guess: &[u8; 5],
    candidates: &BitVec<u64, Lsb0>,
    all_candidates: &[[u8; 5]]
) -> io::Result<usize> {
    loop {
        print!("Enter response (e.g., BGYBB) or 'exit': ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim().to_uppercase();

        if input == "EXIT" {
            std::process::exit(0);
        }

        if input.len() != N_CHARS {
            println!("Invalid length. Must be {N_CHARS} characters.");
            continue;
        }

        let mut resp_bytes = [0u8; 5];
        let mut invalid_chars: HashSet<char> = HashSet::new();
        for (i, c) in input.chars().enumerate() {
            match c {
                'B' => resp_bytes[i] = B,
                'Y' => resp_bytes[i] = Y,
                'G' => resp_bytes[i] = G,
                _ => { invalid_chars.insert(c); }
            }
        }
        if invalid_chars.len() > 0 {
            println!("Invalid characters: {:?}", invalid_chars);
            continue;
        }

        let resp_idx = response_to_index(&resp_bytes);

        // Validation: Is this response actually possible given remaining candidates?
        // If the partition size for this response is 0, the user made a mistake.
        let mut possible = false;
        for c_idx in candidates.iter_ones() {
            let cand = &all_candidates[c_idx];
            let calculated = get_resp(guess, cand);
            if response_to_index(&calculated) == resp_idx {
                possible = true;
                break;
            }
        }

        if !possible {
            println!("Impossible response! No remaining candidate would generate '{}' for guess '{}'. Check your input.", input, arr_to_word(guess));
            continue;
        }

        return Ok(resp_idx);
    }
}

fn compare_heuristics(
    cache_path: &Path,
    output_path: &Path,
    context_hash: u64,
    all_candidates: &[[u8; N_CHARS]],
    all_guesses: &[[u8; N_CHARS]],
    response_cache: &ResponseCache,
) -> Result<(), anyhow::Error> {
    info!("--- Wordle Solver: COMPARISON Mode ---");

    if !cache_path.exists() {
        bail!("Cache file not found. Run 'generate' first.");
    }
    let _memo = wordle_solver::cache::load_cache(cache_path, context_hash)
        .context("Failed to load cache")?;

    let policies: Vec<(&str, Box<dyn Policy + '_>)> = vec![
        (
            "Max Frequency",
            Box::new(MaxFreqPolicy {
                all_candidates,
                all_guesses,
            })
        ),
        (
            "Minimize Remaining",
            Box::new(MinRemainingPolicy {
                response_cache,
                n_total_candidates: all_candidates.len(),
                n_total_guesses: all_guesses.len(),
            })
        ),
        /*
        (
            "Global Optimal",
            Box::new(OptimalPolicy {
                all_guesses,
                response_cache,
                memo,
                n_total_candidates
            })
        ),
        */
    ];


    // Run simulations for each strategy
    let mut results: Vec<(&str, SimStats)> = Vec::new();
    let initial_candidates = bitvec![u64, Lsb0; 1; all_candidates.len()];

    println!("Starting strategy comparison...");
    for (name, policy) in policies {
        println!("Running simulation for: {}", name);

        // The `simulate` function calculates stats for the given policy
        let stats = simulate(&response_cache, &initial_candidates, &*policy);

        println!("  -> Mean: {:.4} | Total Guesses: {}", stats.mean(), stats.total_guesses());
        results.push((name, stats));
    }


    info!("Generating plot at {:?}", output_path);
    generate_comparison_image(results, output_path)
        .map_err(|e| anyhow::anyhow!("Plotting error: {}", e))?;

    Ok(())
}
