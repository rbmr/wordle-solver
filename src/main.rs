use std::collections::HashSet;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use anyhow::{bail, Context};
use bitvec::bitvec;
use bitvec::order::Lsb0;
use bitvec::vec::BitVec;
use log::{info};
use clap::{Parser, Subcommand};
use wordle_solver::cache::{compute_context_hash, just_save_cache, new_cache};
use wordle_solver::graph::generate_comparison_image;
use wordle_solver::policy::pick_optimal;
use wordle_solver::resp::{compute_response_cache, get_resp, response_to_index, B, CORRECT_IDX, G, Y};
use wordle_solver::sim::{simulate, MAX_FREQUENCY_POLICY, MIN_REMAINING_POLICY};
use wordle_solver::solver::compute_optimal_move;
use wordle_solver::words::{arr_to_word, words_to_arr, CANDIDATES, GUESSES, N_CHARS};


#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
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
        #[arg(short, long, default_value = "comparison.png")]
        output: PathBuf,
    },
}

fn main() -> Result<(), anyhow::Error> {
    env_logger::init();
    let cli = Cli::parse();

    // Load data
    info!("Loading word lists...");
    let candidates_arr = words_to_arr(&CANDIDATES);
    let guesses_arr = words_to_arr(&GUESSES);
    let context_hash = compute_context_hash(&guesses_arr, &candidates_arr);

    // Compute basics
    let response_cache = compute_response_cache(&guesses_arr, &candidates_arr);

    match cli.command {
        Commands::Play { cache } => {
            play(
                &cache,
                context_hash,
                candidates_arr,
                guesses_arr,
                response_cache
            )?;
        }
        Commands::Generate { cache } => {
            generate(
                &cache,
                context_hash,
                candidates_arr,
                guesses_arr,
                response_cache
            )?;
        },
        Commands::Compare { cache, output } => {
             compare_heuristics(
                &cache, &output, context_hash,
                candidates_arr, guesses_arr, response_cache
            )?;
        }
    }

    Ok(())
}

fn generate(
    cache_path: &Path,
    context_hash: u64,
    candidates: Vec<[u8; N_CHARS]>,
    guesses: Vec<[u8; N_CHARS]>,
    response_cache: Box<[u8]>,
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
    let (best_idx, cost) = compute_optimal_move(
        &response_cache, &candidates, &guesses, &memo
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
    response_cache: Box<[u8]>,
) -> Result<(), anyhow::Error> {
    println!("--- Wordle Solver: PLAY Mode ---");

    // Load Cache
    if !cache_path.exists() {
        bail!("Cache file not found at {:?}. Run 'generate' first.", cache_path);
    }
    let memo = wordle_solver::cache::load_cache(cache_path, context_hash)
        .context("Failed to load cache")?;

    let mut current_candidates = bitvec![u64, Lsb0; 1; candidates.len()];
    let n_total = candidates.len();

    loop {
        let count = current_candidates.count_ones();
        assert!(count > 0); // Should be guaranteed by get_user_response

        // Print Remaining Candidates
        print_candidates(&current_candidates, &candidates, None);

        // Get Optimal Guess
        println!("Thinking...");
        let guess_idx = pick_optimal(
            &current_candidates, &guesses,
            n_total, &response_cache, &memo
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
        let mut next_candidates = bitvec![u64, Lsb0; 0; n_total];
        for c_idx in current_candidates.iter_ones() {
            let actual_resp = response_cache[guess_idx * n_total + c_idx] as usize;
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
    candidates: Vec<[u8; N_CHARS]>,
    guesses: Vec<[u8; N_CHARS]>,
    response_cache: Box<[u8]>,
) -> Result<(), anyhow::Error> {
    info!("--- Wordle Solver: COMPARISON Mode ---");

    if !cache_path.exists() {
        bail!("Cache file not found. Run 'generate' first.");
    }
    let memo = wordle_solver::cache::load_cache(cache_path, context_hash)
        .context("Failed to load cache")?;

    let c_idx_to_g_idx = wordle_solver::utils::compute_cidx_to_gidx_map(&candidates, &guesses);
    let initial_candidates = bitvec![u64, Lsb0; 1; candidates.len()];
    let _memo_ref = Some(memo);

    let mut results = Vec::new();

    info!("1/3: Simulating Max Frequency...");
    let stats_max = simulate(
        &initial_candidates, &guesses, &candidates, &response_cache,
        &c_idx_to_g_idx, None, MAX_FREQUENCY_POLICY
    );
    results.push(("Max Frequency", stats_max));

    info!("2/3: Simulating Min Remaining...");
    let stats_min = simulate(
        &initial_candidates, &guesses, &candidates, &response_cache,
        &c_idx_to_g_idx, None, MIN_REMAINING_POLICY
    );
    results.push(("Min Remaining", stats_min));

    // info!("3/3: Simulating Optimal (this uses the cache)...");
    // let stats_opt = simulate(
    //     &initial_candidates, &guesses, &candidates, &response_cache,
    //     &c_idx_to_g_idx, memo_ref, OPTIMAL_CACHE_POLICY
    // );
    // results.push(("Optimal", stats_opt));

    info!("Generating plot at {:?}", output_path);
    generate_comparison_image(results, output_path)
        .map_err(|e| anyhow::anyhow!("Plotting error: {}", e))?;

    Ok(())
}
